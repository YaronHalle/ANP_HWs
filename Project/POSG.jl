using Parameters
using LinearAlgebra

include("ParticleB.jl")

T = 5
x0 = [(3^0.5)/8 1/8; -(3^0.5)/8 1/8; 0 -0.25]
UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
REWARD_RADIUS = 0.1
REWARD = 1.0

const POSGActionObservationHist = Tuple{Matrix{Int64}, Array{Int64, 3}}# actions of all players, Obervation of all players
const POSGInfoKey = Tuple{Int, Vector{Int64}, Array{Int64, 2}} # player, his preivious action, his observation



@with_kw struct POSGHist
    action_obs_hist::POSGActionObservationHist
    xt::Array{Float64, 2}
    pb::ParticleBelief
    t::Int 
end

function initiPOSGHist(num_player::Int64, num_particle::Int64, a::Float64)
    action_obs_hist = (fill(-1, 1, 3), fill(-1, 1, 3, 2))
    pb = InitParticleBelief(num_particle, a, num_player)
    return POSGHist(action_obs_hist, x0, pb, 0)
end

@with_kw struct TagGame 
    num_particle::Int
    num_player::Int
    a::Float64
    actions = [0 0.1; 0 -0.1 ; -0.1 0; 0.1 0;]
end




function get_player(game::TagGame, h::POSGHist)
    last_t = h.action_obs_hist[1][end,:]
    for i in 1:game.num_player
        if last_t[i] == -1
            return i
        end
    end
    return 0
end

# 1->2->3->1
function c_reward(pb::ParticleBelief, player::Int)
    runining_from = player - 1
    chasing = player + 1 
    if runining_from == 0
        runining_from = 3
    end
    if chasing == 4
        chasing = 1
    end

    avg_r = 0.0
    for i in 1:size(pb.particles, 1)
        weight = pb.weights[i]
        if weight == 0
            continue
        end
        player_pos = pb.particles[i, player, :]
        runining_from_pos = pb.particles[i, runining_from, :]
        chasing_pos = pb.particles[i, chasing, :]
        if norm(player_pos - runining_from_pos) < REWARD_RADIUS 
            avg_r = avg_r - REWARD * weight
        end
        if norm(player_pos - chasing_pos) < REWARD_RADIUS
            avg_r = avg_r + REWARD * weight
        end
    end
    # if avg_r != 0.0
    #     println("player $player get reward $avg_r")
    # end
    return avg_r
end

function chance_action_next_h(game::TagGame, h::POSGHist, sesstion::Int)
        # in case of chance player
        action_obs_hist = deepcopy(h.action_obs_hist)
        pb = deepcopy(h.pb)
        xt = deepcopy(h.xt)
        t = h.t + 1
        last_join_action = action_obs_hist[1][end, :]
        last_join_action_raw = [game.actions[last_join_action[i], :] for i in 1:game.num_player]
        xt1 = xt + vcat(permutedims.(last_join_action_raw)...)
        pb1 = UpdateBelief(pb, vcat(permutedims.(last_join_action_raw)...), xt1)
        #scatterParticles(pb1, "T=$t,sesstion$sesstion", game.num_player, xt1)
        current_obs = GenerateObservation(xt1)
        action1 = vcat(action_obs_hist[1], fill(-1, 1, game.num_player))
        obs1 = vcat(action_obs_hist[2], fill(-1, 1, game.num_player, 2))
        for i in 1:game.num_player
            obs1[end - 1,i,:] = current_obs[i,:]
        end
        return POSGHist((action1, obs1), xt1, pb1, t)
end

function next_hist(game::TagGame, h::POSGHist, a::Int)
    player = get_player(game, h)
    pb = deepcopy(h.pb)
    action_obs_hist = deepcopy(h.action_obs_hist)
    xt = deepcopy(h.xt)
    t = deepcopy(h.t) 
    new_h = POSGHist(action_obs_hist, xt, pb, t)

    new_h.action_obs_hist[1][end, player] = a
    return new_h
end

function isterminal(game::TagGame,  h::POSGHist)
    if size(h.action_obs_hist[2], 1) > T
        return true
    end 
    return false
end

function actions(game::TagGame, k::POSGInfoKey)
    return [UP, DOWN, LEFT, RIGHT]
end

function infokey(g::TagGame, h::POSGHist)
    p = get_player(g,h)
    actions = h.action_obs_hist[1][:, p]
    obs = h.action_obs_hist[2][:,p,:]
    if size(actions, 1) == 1
        return (p, actions, obs)
    end
    actions = [i for i in actions if i!= -1]
    obs = [obs[i, :] for i in 1:size(obs, 1) if obs[i, :]!= [-1, -1]]
    obs = vcat(permutedims.(obs)...)
    return (p, actions, obs) #player, his action, his observation
end



