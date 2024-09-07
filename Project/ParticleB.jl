using Parameters
using Distributions


@with_kw struct ParticleBelief
    particles::Array{Float64, 3}
    weights::Array{Float64, 1}
end

x0 = [(sqrt(3)/8) 1/8; -(sqrt(3)/8) 1/8; 0 -0.25]


function InitParticleBelief(n_particles::Int, a::Float64, num_player::Int)::ParticleBelief
    # Initialize an array to hold the samples
    particles = Array{Float64}(undef, n_particles, 3, 2)
    cov_matrix = [1.0 0.0; 0.0 1.0] * a
    # Loop over each point in x0, and sample from the corresponding Gaussian distribution
    for i in 1:3
        # Create a multivariate normal distribution with mean x0[i, :] and given covariance
        mvn = MvNormal(x0[i, :], cov_matrix)
        
        # Sample n_samples points from the distribution and store in samples array
        particles[:, i, :] = rand(mvn, n_particles)'  # Transpose to fit shape (100, 2)
    end
    weights = fill(1.0 / n_particles, n_particles)
    return ParticleBelief(particles, weights)
end

function PropagateBelief(b::ParticleBelief,  a::Array{Float64, 2})::ParticleBelief
    particles = b.particles
    new_weights = b.weights
    new_particles = reshape(a, 1, size(a, 1), size(a, 2)) .+ particles 
    # for i in 1:size(particles, 1)
    #     new_particles[i,:,:] = a + particles[i,:,:]
    # end
    return ParticleBelief(new_particles, new_weights)
end

quantized_angles = [0, pi/2, pi, 3*pi/2]

function calculate_angle_and_quntized(row)
    # Assuming row is a 2D vector [x, y]
    x = row[1]
    y = row[2]
    if x >= 0 && y > 0
        return 1
    end
    if x < 0 && y >= 0
        return 2
    end
    if x <= 0 && y < 0
        return 3
    end
    if x > 0 && y <= 0
        return 4
    end
    return 1
end

function GenerateObservation(x_gt::Array{Float64, 2})::Array{Int64, 2}
    num_player = size(x_gt, 1)
    os = zeros(num_player, 2)
    for i::Int in 1:num_player
        k = 1
        for j::Int in 1:num_player
            if i == j
                continue
            end
            reletive_pos = x_gt[j, :] .- x_gt[i, :]
            os[i,k] = calculate_angle_and_quntized(reletive_pos) 
            k = k + 1
        end
    end
    return os
end

function UpdateBelief(b::ParticleBelief,  a::Array{Float64, 2}, x_gt::Array{Float64, 2})::ParticleBelief
    propogated_belief = PropagateBelief(b, a)
    os = GenerateObservation(x_gt)
    new_particles = propogated_belief.particles
    new_weights = propogated_belief.weights
    for i::Int in 1:size(new_particles, 1)
        if propogated_belief.weights[i] == 0
            continue
        end
        if os != GenerateObservation(new_particles[i, :, :])
            new_weights[i] = 0.
        end
    end
    new_weights = new_weights / sum(new_weights)
    if maximum(new_weights) > 1
        throw(ArgumentError("All Particles are out!"))
    end
    return ParticleBelief(new_particles, new_weights)
end