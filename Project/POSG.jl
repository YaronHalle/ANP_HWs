using Parameters
include("ParticleB.jl")

T = 5
x0 = [(3^0.5)/8 1/8; -(3^0.5)/8 1/8; 0 -0.25]
UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4


const POSGActionObservationHist = Tuple{Matrix{Int64}, Matrix{Int64}}# actions of all players, Obervation of all players
const POSGInfoKey = Tuple{Int, Vector{Int64}, Vector{Int64}} # player, his preivious action, his observation



@with_kw struct POSGHist
    action_obs_hist::POSGActionObservationHist
    xt::Array{Float64, 2}
end

function initiPOSGHist(num_player::Int64)
    action_obs_hist = (fill(-1, 1, 3), fill(-1, 1, 3))
    return POSGHist(action_obs_hist, x0)
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

function chance_action_next_h(game::TagGame, h::POSGHist)
        # in case of chance player
        action_obs_hist = h.action_obs_hist
        xt = h.xt
        last_join_action = action_obs_hist[1][end, :]
        last_join_action_raw = [game.actions[last_join_action[i], :] for i in 1:game.num_player]
        xt1 = xt + vcat(permutedims.(last_join_action_raw)...)
        current_obs = GenerateObservation(xt1)
        action1 = vcat(action_obs_hist[1], fill(-1, 1, game.num_player))
        obs1 = vcat(action_obs_hist[2], fill(-1, 1, game.num_player))
        for i in 1:game.num_player
            obs1[end - 1,i] = current_obs[i]
        end
        return POSGHist((action1, obs1), xt1)
end

function next_hist(game::TagGame, h::POSGHist, a::Int)
    player = get_player(game, h)
    action_obs_hist = h.action_obs_hist
    xt = h.xt
    new_h = POSGHist(action_obs_hist, xt)

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
    obs = h.action_obs_hist[2][:, p]
    if size(actions, 1) == 1
        return (p, actions, obs)
    end
    actions = [i for i in actions if i!= -1]
    obs = [i for i in obs if i!= -1]
    return (p, actions, obs) #player, his action, his observation
end



