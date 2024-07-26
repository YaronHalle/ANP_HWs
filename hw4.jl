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


function TransitBeliefMDP(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1})::FullNormal
    p_b = PropagateBelief(b, 𝒫, a)
    x = vec(rand(p_b, 1))
    z = GenerateObservationFromBeacons(𝒫, x, false)
    if z === nothing
        return p_b
    else
        return PropagateUpdateBelief(p_b, 𝒫, a, z.obs)
    end
end

function selectAction(b::FullNormal, d::Int32, lambda::Float64, n::Int32, 𝒫::POMDPscenario, A::Array{Array{Float64, 1}, 1}, x_g::Array{Float64, 1}, discount_factor::Float64)::Tuple{Array{Float64, 1}, Float64}
    if d == 0
        return nothing, 0.0
    b_a, b_v = nothing, Inf
    for a in A
        q = 0.0
        for i in 1:n
            p_b = TransitBeliefMDP(b, 𝒫, a)
            r =  norm(mean(p_b) - x_g) + lambda * det(var(p_b))
            _, v_c = selectAction(p_b, d-1, lambda, n, 𝒫, A, x_g, discount_factor)
            q = q + value(r + discount_factor*v_c) / n
        end
        if q < b_v
            b_a, b_v = a, q
        end
    end
    return b_a, b_v
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
    action = [[1,0], [-1,0], [0,1], [0,-1], [1/sqrt(2),1/sqrt(2)], [-1/sqrt(2),1/sqrt(2)], [1/sqrt(2),-1/sqrt(2)], [-1/sqrt(2),-1/sqrt(2)], [0,0]]
    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                      Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv=[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin) 

    T = 15

    b0 = MvNormal(μ0, Σ0)
    # initialize particle belief
    
    # initialize ground truth
    xgt0 = [-0.5, -0.2]               
    τ = [xgt0]      
    x_goal = [11, 11]  

    # select action Parameters
    deapth = 8
    n = 10
    λ = 1
    discount_factor = 0.9

    # generate motion trajectory
    τp = [b0]
    τobsbeacons = []
    for i in 1:T-1
        ak, _ = selectAction(τp[end], deapth, λ, n, 𝒫, action, x_goal, discount_factor)
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i]))
        if τobsbeacons[i+1] === nothing
            push!(τb, PropagateBelief(τb[end],  𝒫, ak))
        else
            push!(τb, PropagateUpdateBelief(τb[end],  𝒫, ak, τobsbeacons[i+1].obs))
        end
    end 


    bplot =  scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    savefig(bplot,"beacons.pdf")

    
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