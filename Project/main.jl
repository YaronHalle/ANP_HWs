include("POSG.jl")
include("CFRsolver.jl")

x0 = [(sqrt(3)/8) 1/8; -(sqrt(3)/8) 1/8; 0 -0.25]

function show_most_pro_policy(solver::ESCFRSolver)
    colors = [:red, :green, :blue]
    coord_dict = Dict{Tuple{Int64, Float64, Float64}, Vector{Float64}}()
    l = zeros(3)
    dr=scatter([x0[i,1] for i in 1:size(x0, 1)], [x0[i,2] for i in 1:size(x0, 1)], label="gt")
    prog = Progress(length(solver.I); enabled=true)
    for (k, v) in solver.I
        player_num = k[1]
        player_pos = x0[player_num, :]
        pre_act = k[2]

        if pre_act[1] != -1
            for i in 1:length(pre_act)
                player_pos = solver.game.actions[pre_act[i], :] + player_pos
                # print("Player ", player_num, "Action", solver.game.actions[pre_act[i]], " ", player_pos)
            end
            # print("Player start", x0[player_num, :])
            # println(player_pos)
            # println(pre_act)
        end

        pol = v.s/sum(v.s)
        key = (player_num, player_pos[1], player_pos[2])
        if haskey(coord_dict, key)
            coord_dict[key] = coord_dict[key] .+ pol
        else
            coord_dict[key] = pol
        end
        next!(prog)
    end

    
    for player_num in 1:3
        cur_pos = x0[player_num, :]
        for j in 1:5
            color = colors[player_num]
            pol = coord_dict[(player_num, cur_pos[1], cur_pos[2])]
            pol = pol / sum(pol)
            best_act = argmax(pol)
            new_pos = cur_pos + solver.game.actions[best_act, :]
            w = pol[best_act]
            if l[player_num] == 0  
                plot!([cur_pos[1], new_pos[1]], [cur_pos[2], new_pos[2]], arrow=true, lw=w, color=color, label="Player $player_num")
                l[player_num] = 1
            else
                plot!([cur_pos[1], new_pos[1]], [cur_pos[2], new_pos[2]], arrow=true, lw=w, color=color, label=false)
            end
            cur_pos = new_pos
        end
    end
    savefig(dr,"PB2.pdf")
    print("Done")
    end


function show_policy(solver::ESCFRSolver)
    colors = [:red, :green, :blue]
    coord_dict = Dict{Tuple{Int64, Float64, Float64}, Vector{Float64}}()
    l = zeros(3)
    dr=scatter([x0[i,1] for i in 1:size(x0, 1)], [x0[i,2] for i in 1:size(x0, 1)], label="gt")
    prog = Progress(length(solver.I); enabled=true)
    for (k, v) in solver.I
        player_num = k[1]
        player_pos = x0[player_num, :]
        pre_act = k[2]

        if pre_act[1] != -1
            for i in 1:length(pre_act)
                player_pos = solver.game.actions[pre_act[i], :] + player_pos
                # print("Player ", player_num, "Action", solver.game.actions[pre_act[i]], " ", player_pos)
            end
            # print("Player start", x0[player_num, :])
            # println(player_pos)
            # println(pre_act)
        end

        pol = v.s/sum(v.s)
        key = (player_num, player_pos[1], player_pos[2])
        if haskey(coord_dict, key)
            coord_dict[key] = coord_dict[key] .+ pol
        else
            coord_dict[key] = pol
        end
        next!(prog)
    end

    
    for (k, v) in coord_dict
        player_num = k[1]
        player_pos = [k[2], k[3]]
        color = colors[player_num]
        v = exp.(v / 0.1)
        pol = v / sum(v)
        println(pol)
        for i in 1:4
            # Get the direction and weight
            dir = solver.game.actions[i, :]
            if pol[i] < 0.1
                continue
            end
            w = pol[i] 
        
            # Define the end point for the direction
            end_point = player_pos .+ dir
        
            # Plot the line with thickness proportional to the weight
            if l[player_num] == 0  
                plot!([player_pos[1], end_point[1]], [player_pos[2], end_point[2]], arrow=true, lw=w, color=color, label="Player $player_num")
                l[player_num] = 1
            else
                plot!([player_pos[1], end_point[1]], [player_pos[2], end_point[2]], arrow=true, lw=w, color=color, label=false)
            end
        end
    end

    savefig(dr,"PB.pdf")
    print("Done")
end

function main()
    # pb = InitParticleBelief(100, 0.008, 3)
    # x0 = [(3^0.5)/8 1/8; -(3^0.5)/8 1/8; 0 -0.25]
    # x1 = x0 + [0.1 0;  0 0.1; 0.1 0]
    # pb = UpdateBelief(pb, [0.1 0;  0 0.1; 0.1 0], x1)
    # x1 = x0 + [0.1 0;  -0.1 0; 0.1 0]
    # pb = UpdateBelief(pb, [0.1 0;  -0.1 0; 0.1 0], x1)
    # uniform distribution range [-a, a]
    game = TagGame(num_particle = 5000, num_player = 3, a = 0.001)
    T = 1000
    solver = ESCFRSolver(Dict{POSGInfoKey, MCInfoState}(), game, 0, fill(0, (T, 3 ,4)))
    train!(solver, show_progress=true, T)
    show_most_pro_policy(solver)
end


main()


