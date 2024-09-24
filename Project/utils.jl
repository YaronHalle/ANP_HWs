using LinearAlgebra
using Plots


x0 = [(sqrt(3)/8) 1/8; -(sqrt(3)/8) 1/8; 0 -0.25]

function scatterParticles(belief::ParticleBelief, label::String, num_player::Int, xt::Array{Float64, 2})
    dr=scatter([xt[i,1] for i in 1:size(xt, 1)], [xt[i,2] for i in 1:size(xt, 1)], label="gt")
    for p_num in 1:num_player
        x = [p for p in belief.particles[:, p_num, 1]]
        y = [p for p in belief.particles[:, p_num, 2]]

        w = belief.weights

        # Calculate the weighted sum of the particle positions
        weighted_x = sum(x .* w)
        weighted_y = sum(y .* w)

        scatter!(x, y, markersize=w .*50, markercolor=:auto, markerstrokewidth=0, alpha=0.5, label=label)
         # Plot the weighted sum as a distinct point
        scatter!([weighted_x], [weighted_y], color=:auto, markersize=10, marker=:x, label="Weighted particle-$label")
    end
    savefig(dr,"PB$label.pdf")
end

