include("hw2_skeleton.jl")

using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters
using StatsBase



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

function selectAction(b::FullNormal, d::Int64, lambda::Float64, n::Int64, 𝒫::POMDPscenario, A::Array{Float64, 2}, x_g::Array{Float64, 1}, discount_factor::Float64)::Tuple{Union{Array{Float64, 1}, Nothing}, Float64}
    if d == 0
        return nothing, 0.0
    end
    b_a, b_v = nothing, Inf
    for ai in 1:size(A,1)
        # println("ai: ", ai, d)
        q = 0.0
        for i in 1:n
            p_b = TransitBeliefMDP(b, 𝒫, A[ai, :])
            r =  norm(mean(p_b) - x_g) + lambda * det(cov(p_b))
            _, v_c = selectAction(p_b, d-1, lambda, n, 𝒫, A, x_g, discount_factor)
            q = q + (r + discount_factor*v_c) / n
        end
        if q < b_v
            b_a, b_v = A[ai, :], q
        end
    end
    return b_a, b_v
end

function get_trajectory(xgt0::Array{Float64, 1}, b0::FullNormal, T::Int64, deapth::Int64, λ::Float64, n::Int64, 𝒫::POMDPscenario, action::Array{Float64, 2}, x_goal::Array{Float64, 1}, discount_factor::Float64)
    τ = [xgt0]
    τp = [b0]
    τobsbeacons = []
    for i in 1:T
        ak, _ = selectAction(τp[end], deapth, λ, n, 𝒫, action, x_goal, discount_factor)
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[end], false))
        if τobsbeacons[end] === nothing
            push!(τp, PropagateBelief(τp[end],  𝒫, ak))
        else
            push!(τp, PropagateUpdateBeliefBeacon(τp[end],  𝒫, ak, τobsbeacons[end].obs, τobsbeacons[end].index, τobsbeacons[end].Σv))
        end
    end 
    return τ, τobsbeacons, τp
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
    action = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0; 1/sqrt(2) 1/sqrt(2); -1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2); -1/sqrt(2) -1/sqrt(2); 0.0 0.0;]
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
    x_goal = [11.0, 11.0]  

    # select action Parameters
    deapth = [1, 1, 1, 3, 3, 3, 8, 8, 8, 8]
    n = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4]
    λ = 1.0
    discount_factor = 1.0

    # generate motion trajectory
    for i in 1:10
        τ, τobsbeacons, τp = get_trajectory(xgt0, b0, T, deapth[i], λ, n[i], 𝒫, action, x_goal, discount_factor)

        bplot =  scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        scatter!([x_goal[1]], [x_goal[2]], label="goal", markershape=:star5)
        scatter!([x[1] for x in τ], [x[2] for x in τ], label="gt", markershape=:circle)
        f_o = []
        for j in 1:size(τobsbeacons, 1)
            if τobsbeacons[j] !== nothing
                x_sensor_i = 𝒫.beacons[τobsbeacons[j].index, :]
                push!(f_o, τobsbeacons[j].obs + x_sensor_i)
            end
        end
        scatter!([x[1] for x in f_o], [x[2] for x in f_o], label="observation", markershape=:square)
        savefig(bplot,"trajectory$i.pdf")

        dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
        scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
        scatter!([x_goal[1]], [x_goal[2]], label="goal", markershape=:star5)
        for i in 1:T
            covellipse!(τp[i].μ, τp[i].Σ, showaxes=true, n_std=3, label="step $i")
        end
        savefig(dr,"dr$i.pdf")
    end


end 

main()