using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters
using StatsBase

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    rng::MersenneTwister
    beacons::Array{Float64, 2}
    d::Float64
    rmin::Float64
end

@with_kw struct ParticleBelief
    particles::Array{Array{Float64, 1}}
    weights::Array{Float64, 1}
end

function scatterParticles(belief::ParticleBelief, label::String)
    x = [p[1] for p in belief.particles]
    y = [p[2] for p in belief.particles]
    w = belief.weights
    scatter!(x, y, markersize=w .*50, markercolor=:auto, markerstrokewidth=0, alpha=0.5, label=label)
end

function InitParticleBelief(n_particles::Int, μ0::Array{Float64, 1}, Σ0::Array{Float64, 2})::ParticleBelief
    # add your code here
    return ParticleBelief(particles, weights)
    
end
function PropagateBelief(b::ParticleBelief, 𝒫::POMDPscenario, a::Array{Float64, 1})::ParticleBelief
    # add your code here
    return ParticleBelief(new_particles, new_weights)
end

function obs_likelihood(𝒫::POMDPscenario, o::Array{Float64, 1}, x::Array{Float64, 1})::Float64
    μ_z = # add your code here
    return pdf(MvNormal(μ_z , 𝒫.Σv), o)
end

function PropagateUpdateBelief(b::ParticleBelief, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::ParticleBelief
    # add your code here
    return ParticleBelief(new_particles, new_weights)
end  

function resample(b::ParticleBelief)::ParticleBelief
    # add your code here
    # Hint: use the function sample to sample from the particles and use the flag replace=true to sample with replacement
end

function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    # add your code here
end 


function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1})::Union{NamedTuple, Nothing}
    distances = # add your code here
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            obs = # add your code here
            return (obs=obs, index=index) 
        end    
    end 
    return nothing    
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
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i]))
    end  

    println(τobsbeacons)

    bplot =  scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    savefig(bplot,"beacons.pdf")

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

main()