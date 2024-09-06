using ProgressMeter

using Random

include("POSG.jl")


mutable struct MCInfoState 
    σ::Vector{Float64}
    r::Vector{Float64}
    s::Vector{Float64}
    _tmp_σ::Vector{Float64}
    a_idx::Int
end

function MCInfoState(L::Integer)
    return MCInfoState(
        fill(1/L, L),
        zeros(L),
        fill(1/L,L),
        fill(1/L,L),
        0
    )
end

@with_kw struct ESCFRSolver
    I::Dict{POSGInfoKey, MCInfoState}
    game::TagGame
end

function weighted_sample(rng::AbstractRNG, w::AbstractVector)
    t = rand(rng)
    i = 1
    cw = first(w)
    while cw < t && i < length(w)
        i += 1
        @inbounds cw += w[i]
    end
    return i
end

weighted_sample(w::AbstractVector) = weighted_sample(Random.GLOBAL_RNG, w)

Random.rand(I::MCInfoState) = weighted_sample(I.σ)


function regret_match!(σ::AbstractVector, r::AbstractVector)
    s = 0.0
    for (i,r_i) in enumerate(r)
        if r_i > 0.0
            s += r_i
            σ[i] = r_i
        else
            σ[i] = 0.0
        end
    end
    s > 0.0 ? (σ ./= s) : fill!(σ,1/length(σ))
end

function regret_match!(sol::ESCFRSolver)
    for I in values(sol.I)
        regret_match!(I.σ, I.r)
        I.a_idx = 0
    end
end

function CFR(solver::ESCFRSolver, h, i, t)
    game = solver.game
    current_player = get_player(game, h)

    if isterminal(game, h)
        return utility(game, i, h)
    elseif iszero(current_player) # chance player
        h′ = chance_action_next_h(game, h)
        return CFR(solver, h′, i, t)
    end

    k = infokey(game, h)
    I = infoset(solver, k)
    A = actions(game, k)

    v_σ = 0.0

    if current_player == i
        v_σ_Ia = I._tmp_σ
        for (k,a) in enumerate(A)
            h′ = next_hist(game, h, a)
            v_σ_Ia[k] = CFR(solver, h′, i, t) 
            v_σ += I.σ[k]*v_σ_Ia[k]
        end

        update!(solver, I, v_σ_Ia, v_σ, t)
    else
        a_idx = I.a_idx
        iszero(a_idx) && (a_idx = rand(I))
        I.a_idx = a_idx
        a = A[a_idx]
        h′ = next_hist(game, h, a)
        v_σ = CFR(solver, h′, i, t)
    end

    return v_σ
end

function update!(sol::ESCFRSolver, I, v_σ_Ia, v_σ, t)
    (;α, β, γ) = sol.method
    s_coeff = t^γ
    for k in eachindex(v_σ_Ia)
        r = (1 - I.σ[k])*(v_σ_Ia[k] - v_σ)
        r_coeff = r > 0.0 ? t^α : t^β

        I.r[k] += r_coeff*r
        I.s[k] += s_coeff*I.σ[k]
    end
    return nothing
end


function infoset(solver::ESCFRSolver, k::POSGInfoKey) 
    infotype = eltype(values(solver.I))
    return get!(solver.I, k) do
        infotype(length(actions(solver.game, k)))
    end
end

function train!(solver::ESCFRSolver, N::Int; show_progress::Bool=false)
    regret_match!(solver)
    prog = Progress(N; enabled=show_progress)
    h0 = initiPOSGHist(solver.game.num_player)
    for t in 1:N
        for i in 1:solver.game.num_player
            CFR(solver, h0, i, t)
        end
        regret_match!(solver)
        next!(prog)
    end
    solver
end