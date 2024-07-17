include("hw2_skeleton.jl")

using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters
using StatsBase


@with_kw struct ParticleBelief
    particles::Array{Array{Float64, 1}}
    weights::Array{Float64, 1}
end

function scatterParticles(belief::ParticleBelief, label::String)
    x = [p[1] for p in belief.particles]
    y = [p[2] for p in belief.particles]
    
    w = belief.weights
    
    # Calculate the weighted sum of the particle positions
    weighted_x = sum(x .* w)
    weighted_y = sum(y .* w)

    scatter!(x, y, markersize=w .*50, markercolor=:auto, markerstrokewidth=0, alpha=0.5, label=label)
     # Plot the weighted sum as a distinct point
    scatter!([weighted_x], [weighted_y], color=:auto, markersize=10, marker=:x, label="Weighted particle-$label")
end

function InitParticleBelief(n_particles::Int, μ0::Array{Float64, 1}, Σ0::Array{Float64, 2})::ParticleBelief
    b0 = MvNormal(μ0, Σ0)
    particles = rand(b0, n_particles)
    particles = [particles[:, i] for i in 1:size(particles, 2)]
    weights = fill(1.0 / n_particles, n_particles)
    return ParticleBelief(particles, weights)
    
end

function PropagateBelief(b::ParticleBelief, 𝒫::POMDPscenario, a::Array{Float64, 1})::ParticleBelief
    particles = b.particles
    n = length(particles)
    sizes = [length(v) for v in particles]
    new_particles = [zeros(Float64, sizes[i]) for i in 1:n]
    Σw = 𝒫.Σw
    for (index, particle) in enumerate(particles)
        b_min = MvNormal(particle + a, Σw)
        n_p = vec(rand(b_min, 1))
        new_particles[index] = n_p
    end
    new_weights = b.weights
    return ParticleBelief(new_particles, new_weights)
end

function obs_likelihood(𝒫::POMDPscenario, o::Array{Float64, 1}, x::Array{Float64, 1})::Float64
    beacons = 𝒫.beacons
    # Preallocate an array for distances
    distances = zeros(1, size(beacons, 1))

    # Calculate the distance for each row in X
    for i in 1:size(beacons, 1)
        distances[i] = norm(beacons[i, :] .- x)
    end

    x_b_nearest = nothing
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            x_b_nearest = beacons[index,:]
        end    
    end 

    if x_b_nearest === nothing
        return 0
    end

    μ_z = x - x_b_nearest
    return pdf(MvNormal(μ_z , 𝒫.Σv), o)
end

function PropagateUpdateBelief(b::ParticleBelief, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::ParticleBelief
    pred_b = PropagateBelief(b, 𝒫, a)
    new_particles = pred_b.particles
    new_weights = zeros(length(new_particles))
    weights = pred_b.weights

    for (index, particle) in enumerate(new_particles)
        obs_w = obs_likelihood(𝒫, o, particle)
        new_weights[index] = weights[index] * obs_w
    end
    new_weights = new_weights / sum(new_weights)
    return ParticleBelief(new_particles, new_weights)
end  

function resample(b::ParticleBelief)::ParticleBelief
    particles = b.particles
    new_particles = []
    weights = b.weights
    new_weights = []

    sampled_indexes = sample(collect(1:length(weights)), Weights(weights), length(weights), replace=true)
    index_map = countmap(sampled_indexes)
    n_new = length(index_map)
    for (key, value) in index_map
        s_p = particles[key]
        push!(new_particles, s_p)
        push!(new_weights,1/n_new)
    end
    return ParticleBelief(new_particles, new_weights)
end


function particle()
        # definition of the random number generator with seed 
        rng = MersenneTwister(1)
        μ0 = [0.0,0.0]
        Σ0 = [1.0 0.0; 0.0 1.0]
        d =1.0 
        rmin = 0.1
        # set beacons locations 
        beacons =  [0.0 0.0; 0.0 4.0; 0.0 8.0; 4.0 0.0; 4.0 4.0; 4.0 8.0; 8.0 0.0; 8.0 4.0; 8.0 8.0]
        𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                          Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                          Σv=[1.0 0.0; 0.0 1.0], 
                          rng = rng , beacons=beacons, d=d, rmin=rmin) 
    
        T = 6
    
        # initialize particle belief
        n_particles = 10
        b0 = InitParticleBelief(n_particles, μ0, Σ0)
        gb0 = MvNormal(μ0, Σ0)
    
        # initialize ground truth
        xgt0 = [-0.5, -0.2]           
        ak = [1.5, 1.5]     
        τ = [xgt0]      
    
        # generate motion trajectory
        for i in 1:T-1
            push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
        end 
    
        # generate observation trajectory
        τobsbeacons = []
        for i in 1:T
            push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i], true))
        end  
    
        # plots 
        dr1=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
        scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        # Plot circles around each beacon
        for i in 1:size(beacons, 1)
            plot_circle!(beacons[i, 1], beacons[i, 2])
        end
        savefig(dr1,"p_trajectory.pdf")
    
        # generate beliefs dead reckoning 
        τbp = [b0]
        
        for i in 1:T-1
            push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
        end

        # generate gaus beliefs dead reckoning 
        τgbp = [gb0]
        
        for i in 1:T-1
            push!(τgbp, PropagateBelief(τgbp[end],  𝒫, ak))
        end
        
        #generate posteriors without resampling
        τb = [b0]
        for i in 1:T-1
            if τobsbeacons[i+1] === nothing
                push!(τb, PropagateBelief(τb[end],  𝒫, ak))
            else
                push!(τb, PropagateUpdateBelief(τb[end],  𝒫, ak, τobsbeacons[i+1].obs))
            end
        end
    
        #generate posteriors with resampling
        τbr = [b0]
        for i in 1:T-1
            if τobsbeacons[i+1] === nothing
                push!(τbr, PropagateBelief(τbr[end],  𝒫, ak))
            else
                b = PropagateUpdateBelief(τbr[end],  𝒫, ak, τobsbeacons[i+1].obs)
                b = resample(b)
                push!(τbr, b)
            end
        end

        #generate  gaus posteriors with resampling
        τgb = [gb0]
        for i in 1:T-1
            if τobsbeacons[i+1] === nothing
                push!(τgb, PropagateBelief(τgb[end],  𝒫, ak))
            else
                push!(τgb, PropagateUpdateBeliefBeacon(τgb[end],  𝒫, ak, τobsbeacons[i+1].obs, τobsbeacons[i+1].index, τobsbeacons[i+1].Σv))
            end
        end
        
        # plots 
        dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
        scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        for i in 1:T
            scatterParticles(τbp[i], "$i")
        end
        savefig(dr,"p_dr.pdf")
    
        tr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
        scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        for i in 1:T
            scatterParticles(τb[i], "$i")
        end
        savefig(tr,"p_no_resampling.pdf")
    
        tr2=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
        scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        for i in 1:T
            scatterParticles(τbr[i], "$i")
        end
        savefig(tr2,"p_resampling.pdf")

        gdr=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
        scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        for i in 1:T
            covellipse!(τgbp[i].μ, τgbp[i].Σ, showaxes=true, n_std=3, label="")
        end
        savefig(gdr,"gaus_prop.pdf")
        
        ttt=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
        scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        for i in 1:T
            covellipse!(τgb[i].μ, τgb[i].Σ, showaxes=true, n_std=3 , label="")
        end
        savefig(ttt,"gaus.pdf")
    
end

function theortical()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    d =1.0 
    rmin = 0.1
    # set beacons locations 
    beacons =  [0.0 0.0; 0.0 4.0; 0.0 8.0; 4.0 0.0; 4.0 4.0; 4.0 8.0; 8.0 0.0; 8.0 4.0; 8.0 8.0]
    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                      Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv=[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin) 

    T = 6


    # initialize ground truth
    xgt0 = [-0.5, -0.2]           
    ak = [1.5, 1.5]     
    τ = [xgt0]      
    b0 = MvNormal(μ0, Σ0)

    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end 

    # generate observation trajectory
    τobsbeacons = []
    for i in 1:T
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i], true))
    end  

    # plots 
    dr1=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    # Plot circles around each beacon
    for i in 1:size(beacons, 1)
        plot_circle!(beacons[i, 1], beacons[i, 2])
    end
    savefig(dr1,"trajectory.pdf")

    # generate beliefs dead reckoning 
    τbp = [b0]
    
    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end
    
    #generate posteriors without resampling
    τb = [b0]
    for i in 1:T-1
        if τobsbeacons[i+1] === nothing
            push!(τb, PropagateBelief(τb[end],  𝒫, ak))
        else
            push!(τb, PropagateUpdateBelief(τb[end],  𝒫, ak, τobsbeacons[i+1].obs))
        end
    end

    #generate posteriors with resampling
    τbr = [b0]
    for i in 1:T-1
        if τobsbeacons[i+1] === nothing
            push!(τbr, PropagateBelief(τbr[end],  𝒫, ak))
        else
            b = PropagateUpdateBelief(τbr[end],  𝒫, ak, τobsbeacons[i+1].obs)
            b = resample(b)
            push!(τbr, b)
        end
    end
    
    # plots 
    dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        scatterParticles(τbp[i], "$i")
    end
    savefig(dr,"dr.pdf")

    tr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        scatterParticles(τb[i], "$i")
    end
    savefig(tr,"no_resampling.pdf")

    tr2=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        scatterParticles(τbr[i], "$i")
    end
    savefig(tr2,"resampling.pdf")

end


function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    d =1.0 
    rmin = 0.1
    # set beacons locations 
    beacons =  [0.0 0.0; 0.0 4.0; 0.0 8.0; 4.0 0.0; 4.0 4.0; 4.0 8.0; 8.0 0.0; 8.0 4.0; 8.0 8.0]
    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                      Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv=[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin) 

    T = 6

    # initialize particle belief
    n_particles = 10
    b0 = InitParticleBelief(n_particles, μ0, Σ0)
    gb0 = MvNormal(μ0, Σ0)

    # initialize ground truth
    xgt0 = [-0.5, -0.2]           
    ak = [1.5, 1.5]     
    τ = [xgt0]      

    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end 

    # generate observation trajectory
    τobsbeacons = []
    for i in 1:T
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i], true))
    end  

    # plots 
    dr1=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    # Plot circles around each beacon
    for i in 1:size(beacons, 1)
        plot_circle!(beacons[i, 1], beacons[i, 2])
    end
    savefig(dr1,"trajectory.pdf")

    # generate beliefs dead reckoning 
    τbp = [b0]
    
    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end
    
    # generate gaus beliefs dead reckoning 

    gτbp = [gb0]
    
    for i in 1:T-1
        push!(gτbp, PropagateBelief(gτbp[end],  𝒫, ak))
    end
    
    #generate posteriors without resampling
    τb = [b0]
    for i in 1:T-1
        if τobsbeacons[i+1] === nothing
            push!(τb, PropagateBelief(τb[end],  𝒫, ak))
        else
            push!(τb, PropagateUpdateBelief(τb[end],  𝒫, ak, τobsbeacons[i+1].obs))
        end
    end

    #generate posteriors with resampling
    τbr = [b0]
    for i in 1:T-1
        if τobsbeacons[i+1] === nothing
            push!(τbr, PropagateBelief(τbr[end],  𝒫, ak))
        else
            b = PropagateUpdateBelief(τbr[end],  𝒫, ak, τobsbeacons[i+1].obs)
            b = resample(b)
            push!(τbr, b)
        end
    end

    #generate gaus posteriors 
    gτb = [gb0]
    for i in 1:T-1
        if τobsbeacons[i+1] === nothing
            push!(gτb, PropagateBelief(gτb[end],  𝒫, ak))
        else
            push!(gτb, PropagateUpdateBeliefBeacon(gτb[end],  𝒫, ak, τobsbeacons[i+1].obs, τobsbeacons[i+1].index, τobsbeacons[i+1].Σv))
        end
    end

    
    # plots 
    dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        scatterParticles(τbp[i], "$i")
    end
    savefig(dr,"dr.pdf")

    tr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        scatterParticles(τb[i], "$i")
    end
    savefig(tr,"no_resampling.pdf")

    tr2=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        scatterParticles(τbr[i], "$i")
    end
    savefig(tr2,"resampling.pdf")

    gdr=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        covellipse!(gτbp[i].μ, gτbp[i].Σ, showaxes=false, n_std=3, label="")
    end
    savefig(gdr,"gaus_prop.pdf")
    
    ttt=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        covellipse!(gτb[i].μ, gτb[i].Σ, showaxes=false, n_std=3 , label="")
    end
    savefig(ttt,"gaus.pdf")

end 

main()