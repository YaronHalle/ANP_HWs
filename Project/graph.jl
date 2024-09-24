using Plots, Distributions, LinearAlgebra
using StatsPlots

# Function to draw 3 Gaussian distributions in 2D space and save the plot as a PDF
function plot_gaussian_distributions()
    # Define the means for each distribution
    means = [(sqrt(3)/8) 1/8; -(sqrt(3)/8) 1/8; 0 -0.25]

    # Covariance matrix (variance = I * 0.001)
    covariance = 0.001 * I(2)  # I(2) is the 2x2 identity matrix

    # Define Gaussian distributions for each agent
    agent1_dist = MvNormal(means[1, :], covariance)
    agent2_dist = MvNormal(means[2, :], covariance)
    agent3_dist = MvNormal(means[3, :], covariance)

    # Define agent names
    agents = ["Agent 1", "Agent 2", "Agent 3"]

    # Create a grid for plotting
    grid_x = range(-0.5, 0.5, length=100)
    grid_y = range(-0.5, 0.5, length=100)

    # Initialize the plot
    p = plot(title="Initialize belief of the Agents", xlabel="X", ylabel="Y", legend=:topleft)

    # Function to plot contours for each distribution
    function plot_distribution!(dist, color, label)
        covellipse!(dist.μ, dist.Σ, showaxes=false, n_std=3 , label=label, color = color)
        #z = [pdf(dist, [y, x]) for x in grid_x, y in grid_y]
        #contour!(p, grid_x, grid_y, z, levels=5, color=color, label=label)
    end

    # Plot each distribution
    plot_distribution!(agent1_dist, :red, agents[1])
    plot_distribution!(agent2_dist, :blue, agents[2])
    plot_distribution!(agent3_dist, :green, agents[3])

    # Add arrows from agent 1 to 2, 2 to 3, and 3 to 1
    # Add arrows using quiver! (1 to 2, 2 to 3, and 3 to 1)
    quiver!([means[1, 1]], [means[1, 2]], quiver=([means[2, 1] - means[1, 1]], [means[2, 2] - means[1, 2]]), arrow=:closed, color=:black, label=false)  # 1 to 2
    quiver!([means[2, 1]], [means[2, 2]], quiver=([means[3, 1] - means[2, 1]], [means[3, 2] - means[2, 2]]), arrow=:closed, color=:black, label=false)  # 2 to 3
    quiver!([means[3, 1]], [means[3, 2]], quiver=([means[1, 1] - means[3, 1]], [means[1, 2] - means[3, 2]]), arrow=:closed, color=:black, label=false)  # 3 to 1

        
    # Save the plot as a PDF
    savefig(p, "gaussian_distributions.pdf")
end

# Call the function to plot and save
plot_gaussian_distributions()
