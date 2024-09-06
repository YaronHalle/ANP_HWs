using Parameters

@with_kw struct ParticleBelief
    particles::Array{Float64, 3}
    weights::Array{Float64, 1}
end

# Sample a initial particle belief from [-a, a] x [-a, a] for all the players in the game (the state size is [2, num_player])
function InitParticleBelief(n_particles::Int, a::Float64, num_player::Int)::ParticleBelief
    particles = (-a) .+ 2a .* rand(n_particles, num_player, 2)
    weights = fill(1.0 / n_particles, n_particles)
    return ParticleBelief(particles, weights)
end

function PropagateBelief(b::ParticleBelief,  a::Array{Float64, 2})::ParticleBelief
    particles = b.particles
    new_weights = b.weights
    new_particles = reshape(a, 1, size(a, 1), size(a, 2)) .+ particles 
    return ParticleBelief(new_particles, new_weights)
end

quantized_angles = [0, pi/2, pi, 3*pi/2]

function calculate_angle_and_quntized(row)
    # Assuming row is a 2D vector [x, y]
    x = row[1]
    y = row[2]
    angle = atan(y, x) 
    if angle < 0
        angle += 2 * pi
    end
    return argmin(abs.(quantized_angles .- angle))
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
        if os != GenerateObservation(new_particles[i, :, :])
            new_weights[i] = 0.
        end
    end
    new_weights = new_weights / sum(new_weights)
    return ParticleBelief(new_particles, new_weights)
end