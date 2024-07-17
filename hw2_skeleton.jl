using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    rng::MersenneTwister
    beacons::Array{Float64, 2}
    d::Float64
    rmin::Float64
end


function PropagateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    # predict
    μp = F*μb + a 
    Σp = F*Σb*transpose(F) + Σw
    return MvNormal(μp, Σp)
end 



function PropagateUpdateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    # predict
    μp = F*μb + a 
    Σp = F*Σb*transpose(F) + Σw
    # update
    μb′ = μp + Σp*inv(Σp + Σv)*(o - μp)
    Σb′ = Σp - Σp * inv(Σp + Σv) * Σp
    return MvNormal(μb′, Σb′)
end    

function PropagateUpdateBeliefBeacon(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1}, index::Int64, Σv::Array{Float64, 2})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw = 𝒫.Σw
    x_sensor_i = 𝒫.beacons[index, :]
    f_o = o + x_sensor_i
    # predict
    μp = F*μb + a 
    Σp = F*Σb*transpose(F) + Σw
    # update
    μb′ = μp + Σp*inv(Σp + Σv)*(f_o - μp)
    Σb′ = Σp - Σp * inv(Σp + Σv) * Σp
    return MvNormal(μb′, Σb′)
end  


function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    F  = 𝒫.F
    w_dist = MvNormal([0., 0.], 𝒫.Σw)
    w = vec(rand(w_dist, 1))
    return F*x + a + w
end 

function GenerateObservation(𝒫::POMDPscenario, x::Array{Float64, 1})
    v_dist = MvNormal([0., 0.], 𝒫.Σv)
    v = vec(rand(v_dist, 1))
    return x + v
end   


function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1}, fixed_v::Bool)::Union{NamedTuple, Nothing}
    beacons_x = 𝒫.beacons
    rmin = 𝒫.rmin

    # Preallocate an array for distances
    distances = zeros(1, size(beacons_x, 1))

    # Calculate the distance for each row in X
    for i in 1:size(beacons_x, 1)
        distances[i] = norm(beacons_x[i, :] .- x)
    end

    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            if fixed_v
                Σv = 𝒫.Σv
                v_dist = MvNormal([0., 0.], Σv)
            else
                r = max(rmin, distance)
                Σv = (0.01 * r) * (0.01 * r) * [1.0 0.0; 0.0 1.0]
                v_dist = MvNormal([0., 0.], Σv)
            end
            v = vec(rand(v_dist, 1))
            x_sensor_i = beacons_x[index, :]
            obs = x - x_sensor_i + v
            return (obs=obs, index=index, Σv=Σv) 
        end    
    end 
    return nothing    
end    

function plot_circle!(cx, cy)
    θ = range(0, stop=2π, length=100)
    x = cx .+ 1 .* cos.(θ)
    y = cy .+ 1 .* sin.(θ)
    plot!(x, y, label="", lw=2)
end


function q3()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d = 1.0 
    rmin = 0.1
    # set beacons locations 
    beacons = [0.0 0.0;
               4.5 0.0;
               9.0 0.0; 
               0.0 4.5;
               4.5 4.5;
               9.0 4.5;
               0.0 9.0;
               4.5 9.0;
               9.0 9.0]# define array with beacons
    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                      Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv=0.01^2*[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin) 
    
    xgt0 = [-0.5, -0.2]           
    T = 100
    N = 10
    ak = [[0.1, 0.1 * (j / 5)] for j in 1:N]  

    # generating the trajectory
    τ_s = [deepcopy([xgt0]) for _ in 1:N]
    
    # generate motion trajectory
    for j in 1:N
        for i in 1:T-1
            push!(τ_s[j], SampleMotionModel(𝒫, ak[j], τ_s[j][end]))
        end  
    end

    # generate observation trajectory
    τobs = [[] for _ in 1:N]
    for j in 1:N
        for i in 1:T
            push!(τobs[j], GenerateObservationFromBeacons(𝒫, τ_s[j][i], false))
        end  
    end
    
    # generate beliefs dead reckoning 
    τbp = [[deepcopy(b0)] for _ in 1:N]
    
    for j in 1:N
        for i in 1:T-1
            push!(τbp[j], PropagateBelief(τbp[j][end],  𝒫, ak[j]))
        end
    end
    
    #generate posteriors 
    τb = [[deepcopy(b0)] for _ in 1:N]
    for j in 1:N
        for i in 1:T-1
            if τobs[j][i+1] === nothing
                push!(τb[j], PropagateBelief(τb[j][end],  𝒫, ak[j]))
            else
                push!(τb[j], PropagateUpdateBeliefBeacon(τb[j][end],  𝒫, ak[j], τobs[j][i+1].obs, τobs[j][i+1].index, τobs[j][i+1].Σv))
            end
        end
    end


    
    # plots 
    dr=scatter([x[1] for x in τ_s[1]], [x[2] for x in τ_s[1]], label="gt-1")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for j in 2:N
        scatter!([x[1] for x in τ_s[j]], [x[2] for x in τ_s[j]], label="gt-$j")
    end
    savefig(dr,"q3_traj.pdf")

    cost = []
    # plot estimation error
    for j in 1:N
        push!(cost, norm(det(τb[j][end].Σ)))
    end
    pl = scatter(1:N, cost, show=true, label="Cost over trajectory index")
    savefig(pl,"q3_cost.pdf")

end


function plot_q2(τ, beacons, τb, τbp, suffix, T)
    # plots 
    dr1=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    # Plot circles around each beacon
    for i in 1:size(beacons, 1)
        plot_circle!(beacons[i, 1], beacons[i, 2])
    end
    savefig(dr1,"tran_$suffix.pdf")

    dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        covellipse!(τbp[i].μ, τbp[i].Σ, showaxes=true, n_std=3, label="")
    end
    savefig(dr,"qa_dr_$suffix.pdf")
    
    ttt=scatter([x[1] for x in τ], [x[2] for x in τ], label="")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        covellipse!(τb[i].μ, τb[i].Σ, showaxes=true, n_std=3 , label="")
    end
    savefig(ttt,"q2_tr_$suffix.pdf")

    err = []
    tr_cov = []
    # plot estimation error
    for i in 1:T
        push!(err, norm(τb[i].μ - τ[i]))
        push!(tr_cov, sqrt(tr(τb[i].Σ)))
    end
    pl = scatter(1:T, err, show=true, label="estimation errors over time")
    savefig(pl, "q2_squared_norms_$suffix.pdf")

    pl = scatter(1:T, tr_cov, show=true, label=" estimation covariance over time")
    savefig(pl,"q2_trace_$suffix.pdf")

end


function q2()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d = 1.0 
    rmin = 0.1
    # set beacons locations 
    beacons = [0.0 0.0;
               4.5 0.0;
               9.0 0.0; 
               0.0 4.5;
               4.5 4.5;
               9.0 4.5;
               0.0 9.0;
               4.5 9.0;
               9.0 9.0]# define array with beacons
    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                      Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv=0.01^2*[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin) 
    
    xgt0 = [-0.5, -0.2]           
    ak = [0.1, 0.1]   
    T=100        

    τ = [xgt0]
    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end 

    # generate observation trajectory
    τobsbeacons_fixed = []
    τobsbeacons_not_fixed = []
    for i in 1:T
        push!(τobsbeacons_not_fixed, GenerateObservationFromBeacons(𝒫, τ[i], false))
        push!(τobsbeacons_fixed, GenerateObservationFromBeacons(𝒫, τ[i], true))
    end  

    # generate beliefs dead reckoning 
    τbp = [b0]
    
    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end
        
    #generate posteriors 
    τb_fixed = [b0]
    τb_not_fixed = [b0]
    for i in 1:T-1
        if τobsbeacons_fixed[i+1] === nothing
            push!(τb_fixed, PropagateBelief(τb_fixed[end],  𝒫, ak))
            push!(τb_not_fixed, PropagateBelief(τb_not_fixed[end],  𝒫, ak))
        else
            push!(τb_fixed, PropagateUpdateBeliefBeacon(τb_fixed[end],  𝒫, ak, τobsbeacons_fixed[i+1].obs, τobsbeacons_fixed[i+1].index, τobsbeacons_fixed[i+1].Σv))
            push!(τb_not_fixed, PropagateUpdateBeliefBeacon(τb_not_fixed[end],  𝒫, ak, τobsbeacons_not_fixed[i+1].obs, τobsbeacons_not_fixed[i+1].index, τobsbeacons_not_fixed[i+1].Σv))
        end
    end

    # plots 
    plot_q2(τ, beacons, τb_fixed, τbp, "fixed", T)
    plot_q2(τ, beacons, τb_not_fixed, τbp, "changing", T)

end


function q1()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d =1.0 # dummy define define 
    rmin = 0.1 # dummy define define 
    # set beacons locations 
    beacons = [1.0 0.0; 0.0 1.0]# dummy define define 

    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                      Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv=0.01^2*[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin)
                      
    ak = [1.0, 0.0]
    xgt0 = [0.5, 0.2]
    T = 10
    # generating the trajectory
    τ = [xgt0]
    
    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end  
    # generate observation trajectory
    τobs = Array{Float64, 1}[]
    for i in 1:T
        push!(τobs, GenerateObservation(𝒫, τ[i]))
    end  
    
    # generate beliefs dead reckoning 
    τbp = [b0]
    
    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end
    
    #generate posteriors 
    τb = [b0]
    for i in 1:T-1
        push!(τb, PropagateUpdateBelief(τb[end],  𝒫, ak, τobs[i+1]))
    end


    
    # plots 
    dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    for i in 1:T
        covellipse!(τbp[i].μ, τbp[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    savefig(dr,"q1_dr.pdf")

    tr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    for i in 1:T
        covellipse!(τb[i].μ, τb[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    savefig(tr,"q1_tr.pdf")

end


# function main()
#     # definition of the random number generator with seed 
#     q1()
#     q2()
#     q3()
# end 