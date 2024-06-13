using Revise #essential tool for tracking changes 
using Plots
using Random
using Distributions
includet("./utils.jl") #include and track changes 




function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1) 

    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0) #multivariate gaussian
    T = 1000
    squared_norms = []
    for i in 1:T
        new_sample = rand(rng, b0)
        squared_norm = new_sample[1]^2+new_sample[2]^2 
        push!(squared_norms, squared_norm)
    end 
    pl = scatter(1:T, squared_norms, show=true, label="squared Euclidean norms")
    savefig(pl,"squared_norms.pdf")


end 

main() #enter for entering the debugger
