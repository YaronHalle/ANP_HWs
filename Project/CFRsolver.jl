using ProgressMeter
using Random
using Plots

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

mutable struct ESCFRSolver
    I::Dict{POSGInfoKey, MCInfoState}
    game::TagGame
    sesstion::Int
    c_regrat_over_time::Array{Float64, 3}
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
        solver.sesstion += 1
        return 0.0
    elseif iszero(current_player) # chance player
        h′ = chance_action_next_h(game, h, solver.sesstion)
        r = c_reward(h′.pb, i)
        return r + CFR(solver, h′, i, t)
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

        update!(solver, I, v_σ_Ia, v_σ, t, i, h)
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

function update!(sol::ESCFRSolver, I, v_σ_Ia, v_σ, t, player, hist)
    #(;α, β, γ) = (1, 1, 1)
    s_coeff = t^1
    for k in eachindex(v_σ_Ia)
        r = (1 - I.σ[k])*(v_σ_Ia[k] - v_σ)
        r_coeff = r > 0.0 ? t^1 : t^1
        
        if hist.action_obs_hist[1][1] == -1 && hist.action_obs_hist[1][2] == -1 && hist.action_obs_hist[1][3] == -1
            println("Player $player, Action $k, Reward $r")
            sol.c_regrat_over_time[t, player, k] = r
        end

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

function train!(solver::ESCFRSolver, N::Int; show_progress::Bool=false, cb=()->())
    regret_match!(solver)
    prog = Progress(N; enabled=show_progress)
    h0 = initiPOSGHist(solver.game.num_player, solver.game.num_particle, solver.game.a)
    #scatterParticles(h0.pb, "T=0", solver.game.num_player, h0.xt)
    for t in 1:N
        #println("t=$t")
        for i in 1:solver.game.num_player
            CFR(solver, h0, i, t)
        end
        regret_match!(solver)
        next!(prog)
    end
    solver
end
############################ Exploit FUNCTION ############################ 
Base.@kwdef mutable struct SuperRoot
    val::Float64 = 0.0
end

struct ExploitabilitySolver{G<:TagGame, K}
    "Game upon which a solver is to be evaluated"
    game        :: G

    "EXPLOITING Player"
    p           :: Int

    "Child information states of being in information state `I` and taking action `a`"
    Ia_children :: Dict{Tuple{K, Int}, Set{K}}

    "Reach probability weighted utility of being in information state `I` and taking action `a`"
    utility     :: Dict{K,Vector{Float64}}

    "π⁻ⁱ(h₀ → I)"
    prob        :: Dict{K,Float64}

    "Information states where player `p` is first allowed to act"
    roots       :: Set{K}

    _super_root :: SuperRoot
end


function reset!(e_sol::ExploitabilitySolver)
    for U in values(e_sol.utility)
        fill!(U, 0.0)
    end
    for I in keys(e_sol.prob)
        e_sol.prob[I] = 0.0
    end
    e_sol._super_root.val = 0.0
    nothing
end

function ExploitabilitySolver(game::TagGame, p::Int)
    I = POSGInfoKey
    e_sol = ExploitabilitySolver(
        game,
        other_player(p),
        Dict{Tuple{I,Int},Set{I}}(),
        Dict{I,Vector{Float64}}(),
        Dict{I,Float64}(),
        Set{I}(),
        SuperRoot()
    )
    return populate!(e_sol)
end

function ExploitabilitySolver(sol::ESCFRSolver, p::Int)
    return ExploitabilitySolver(sol.game, p)
end

function populate!(e_sol::ExploitabilitySolver)
    _populate!(e_sol, initialhist(e_sol.game), nothing, 0)
    return e_sol
end

function _populate!(e_sol::ExploitabilitySolver, h, last_I, last_a_idx::Int)
    game = e_sol.game
    if isterminal(game, h)
        return nothing
    elseif iszero(player(game, h))
        for a in chance_actions(game, h)
            _populate!(e_sol, next_hist(game, h, a), last_I, last_a_idx)
        end
        return nothing
    end

    I = infokey(game, h)
    A = actions(game, I)

    if player(game, h) == e_sol.p
        if isnothing(last_I)
            push!(e_sol.roots, I)
        else
            Ia_children = get(e_sol.Ia_children,(last_I, last_a_idx), nothing)
            if isnothing(Ia_children)
                e_sol.Ia_children[(last_I, last_a_idx)] = Set{typeof(I)}()
            else
                push!(Ia_children, I)
            end
        end
        if !haskey(e_sol.utility, I)
            e_sol.utility[I] = zeros(Float64, length(A))
            e_sol.prob[I] = 0.0
        end
        for (k,a) in enumerate(A)
            _populate!(e_sol, next_hist(game, h, a), I, k)
        end
    else
        for a in A
            _populate!(e_sol, next_hist(game, h, a), last_I, last_a_idx)
        end
    end
    return nothing
end

function utility!(
    e_sol::ExploitabilitySolver,
    solver::ESCFRSolver,
    h               = initialhist(e_sol.game),
    last_I          = e_sol._super_root,
    last_a_idx::Int = 0,
    π_ni::Float64   = 1.0)

    (;game, p) = e_sol
    current_player = player(game, h)
    if isterminal(game, h)
        if last_I isa SuperRoot
            e_sol._super_root.val += π_ni*utility(game, p, h)
        else
            e_sol.utility[last_I][last_a_idx] += π_ni*utility(game, p, h)
        end
        return nothing
    elseif iszero(current_player) # chance player
        A = chance_actions(game, h)
        inv_l_a = inv(length(A))
        for a in A
            utility!(e_sol, solver, next_hist(game, h, a), last_I, last_a_idx, π_ni*inv_l_a)
        end
        return nothing
    end

    I = infokey(game, h)
    A = actions(game, I)

    if current_player == p
        e_sol.prob[I] += π_ni
        for (k,a) in enumerate(A)
            h′ = next_hist(game, h, a)
            utility!(e_sol, solver, h′, I, k, π_ni)
        end
    else
        σ = strategy(solver, I)
        for (k,a) in enumerate(A)
            h′ = next_hist(game, h, a)
            utility!(e_sol, solver, h′, last_I, last_a_idx, π_ni*σ[k])
        end
    end
    return nothing
end

function exploit_utility(e_sol::ExploitabilitySolver, sol::ESCFRSolver)
    reset!(e_sol)
    utility!(e_sol, sol)

    for (I,U) in e_sol.utility
        U ./= e_sol.prob[I]
    end

    u_exp = e_sol._super_root.val
    for I in e_sol.roots
        u_exp += e_sol.prob[I]*best_response(e_sol, I)
    end
    return u_exp
end

function exploitability(e_sol::ExploitabilitySolver, sol::ESCFRSolver)
    u_exp = exploit_utility(e_sol, sol)
    u_current = evaluate(sol, e_sol.p)
    return u_exp - u_current
end

"""
    exploitability(sol::AbstractCFRSolver, p::Int=1)

Calculates exploitability of player `p` given strategy specified by solver `sol`
"""
function exploitability(sol::ESCFRSolver, p::Int=1)
    e_sol = ExploitabilitySolver(sol, p)
    return exploitability(e_sol, sol)
end

function best_response(e_sol::ExploitabilitySolver, I)
    util = e_sol.utility[I]
    π_A = e_sol.prob[I]

    Q_max = -Inf
    for a_i in eachindex(util)
        children = get(e_sol.Ia_children, (I,a_i), nothing)
        Qa_i = 0.0
        Qa_i += π_A*util[a_i]
        if !isnothing(children)
            for I′ in children
                Qa_i += e_sol.prob[I′]*best_response(e_sol, I′)
            end
        end
        if Qa_i > Q_max
            Q_max = Qa_i
        end
    end
    return Q_max/π_A
end

############################ CALLBACK FUNCTION ############################ 

struct ExploitabilityHistory
    x::Vector{Int}
    y::Vector{Float64}
    ExploitabilityHistory() = new(Int[], Float64[])
end

function Base.push!(h::ExploitabilityHistory, x, y)
    push!(h.x, x)
    push!(h.y, y)
end

"""
    ExploitabilityCallback(sol::AbstractCFRSolver, n=1; p=1)

- `sol` :
- `n`   : Frequency with which to query exploitability e.g. `n=10` indicates checking exploitability every 10 CFR iterations
- `p`   : Player whose exploitability is being measured

Usage:
```
using CounterfactualRegret
const CFR = CounterfactualRegret

game = CFR.Games.Kuhn()
sol = CFRSolver(game)
train!(sol, 10_000, cb=ExploitabilityCallback(sol))
```
"""
mutable struct ExploitabilityCallback
    sol::ESCFRSolver
    e_sol::ExploitabilitySolver
    n::Int
    state::Int
    hist::ExploitabilityHistory
end

function ExploitabilityCallback(sol::ESCFRSolver, n::Int=1; p::Int=1)
    e_sol = ExploitabilitySolver(sol, p)
    return ExploitabilityCallback(sol, e_sol, n, 0, ExploitabilityHistory())
end

function (cb::ExploitabilityCallback)()
    if iszero(rem(cb.state, cb.n))
        e = exploitability(cb.e_sol, cb.sol)
        push!(cb.hist, cb.state, e)
    end
    cb.state += 1
end

"""
NashConvCallback(sol::AbstractCFRSolver, n=1)

- `sol` :
- `n`   : Frequency with which to query nash convergence e.g. `n=10` indicates checking exploitability every 10 CFR iterations

Usage:
```
using CounterfactualRegret
const CFR = CounterfactualRegret

game = CFR.Games.Kuhn()
sol = CFRSolver(game)
train!(sol, 10_000, cb=NashConvCallback(sol))
```
"""
mutable struct NashConvCallback
    sol::ESCFRSolver
    e_sols::ExploitabilitySolver
    n::Int
    state::Int
    hist::ExploitabilityHistory
end

function NashConvCallback(sol::ESCFRSolver, n::Int=1)
    e_sols = (ExploitabilitySolver(sol, 1),  ExploitabilitySolver(sol, 2))
    return NashConvCallback(sol, e_sols, n, 0, ExploitabilityHistory())
end

function (cb::NashConvCallback)()
    if iszero(rem(cb.state, cb.n))
        e1 = exploit_utility(cb.e_sols[1], cb.sol)
        e2 = exploit_utility(cb.e_sols[2], cb.sol)
        push!(cb.hist, cb.state, e1 + e2)
    end
    cb.state += 1
end

@recipe function f(hist::ExploitabilityHistory)
    xlabel --> "Training Steps"
    @series begin
        ylabel --> "Exploitability"
        label --> ""
        hist.x, hist.y
    end
end

@recipe f(cb::ExploitabilityCallback) = cb.hist
@recipe f(cb::NashConvCallback) = cb.hist

"""

Wraps a function, causing it to trigger every `n` CFR iterations

```
test_cb = Throttle(() -> println("test"), 100)
```
Above example will print `"test"` every 100 CFR iterations
"""
mutable struct Throttle{F}
    f::F
    n::Int
    state::Int
end

function Throttle(f::Function, n::Int)
    return Throttle(f, n, 0)
end

function (t::Throttle)()
    iszero(rem(t.state, t.n)) && t.f()
    t.state += 1
end


"""
Chain together multiple callbacks

Usage:
```
using CounterfactualRegret
const CFR = CounterfactualRegret


game = CFR.Games.Kuhn()
sol = CFRSolver(game)
exp_cb = ExploitabilityCallback(sol)
test_cb = Throttle(() -> println("test"), 100)
train!(sol, 10_000, cb=CFR.CallbackChain(exp_cb, test_cb))
```
"""
struct CallbackChain{T<:Tuple}
	t::T
	CallbackChain(args...) = new{typeof(args)}(args)
end

Base.iterate(chain::CallbackChain, s=1) = iterate(chain.t, s)

function (chain::CallbackChain)()
	for cb in chain
		cb()
	end
end


# mutable struct MCTSExploitabilityCallback{M<:ISMCTS}
#     mcts::M
#     n::Int
#     eval_iter::Int
#     state::Int
#     hist::ExploitabilityHistory
#     function MCTSExploitabilityCallback(sol::AbstractCFRSolver, n=1; kwargs...)
#         mcts = ISMCTS(sol;kwargs...)
#         new{typeof(mcts)}(mcts, n, mcts.max_iter, 0, ExploitabilityHistory())
#     end
# end

# function exploitability(cb::MCTSExploitabilityCallback)
#     mcts = cb.mcts
#     v_exploit = run(mcts)
#     v_current = approx_eval(mcts.sol, cb.eval_iter, mcts.sol.game, mcts.player)
#     return v_exploit - v_current
# end

# function (cb::MCTSExploitabilityCallback)()
#     if iszero(rem(cb.state, cb.n))
#         push!(cb.hist, cb.state, exploitability(cb))
#     end
#     cb.state += 1
# end

# @recipe f(cb::MCTSExploitabilityCallback) = cb.hist


# mutable struct MCTSNashConvCallback{M}
#     mcts::M
#     n::Int
#     state::Int
#     hist::ExploitabilityHistory
#     function MCTSNashConvCallback(sol::AbstractCFRSolver, n=1; kwargs...)
#         tup = (ISMCTS(sol;kwargs..., player=1), ISMCTS(sol;kwargs..., player=2))
#         new{typeof(tup)}(tup, n, 0, ExploitabilityHistory())
#     end
# end

# function nashconv(cb::MCTSNashConvCallback)
#     u1 = run(cb.mcts[1])
#     u2 = run(cb.mcts[2])
#     return u1 + u2
# end

# function (cb::MCTSNashConvCallback)()
#     if iszero(rem(cb.state, cb.n))
#         push!(cb.hist, cb.state, nashconv(cb))
#     end
#     cb.state += 1
# end

# @recipe f(cb::MCTSNashConvCallback) = cb.hist

# mutable struct ModelSaverCallback{SOL}
#     sol::SOL
#     save_every::Int
#     save_dir::String
#     pad_digits::Int
#     policy_only::Bool
#     state::Int
#     function ModelSaverCallback(sol, save_every; save_dir=joinpath(pwd(),"checkpoints"), pad_digits=9, policy_only=true)
#         new{typeof(sol)}(sol, save_every, save_dir, pad_digits, policy_only, 0)
#     end
# end

# function _fmt_model_str(iter, pad_digits)
#     return "model_"*lpad(iter, pad_digits, '0')*".jld2"
# end

# function _save_model(model, dir, iter, pad_digits)
#     FileIO.save(joinpath(dir, _fmt_model_str(iter, pad_digits)), Dict("model"=>model))
# end

# function (cb::ModelSaverCallback)()
#     if iszero(rem(cb.state, cb.save_every))
#         m = cb.policy_only ? CFRPolicy(cb.sol) : cb.sol
#         _save_model(m, cb.save_dir, cb.state, cb.pad_digits)
#     end
#     cb.state += 1
# end

# load_model(path) = FileIO.load(path)["model"]
