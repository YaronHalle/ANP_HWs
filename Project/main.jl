include("POSG.jl")
include("CFRsolver.jl")

function main()
    # uniform distribution range [-a, a]
    game = TagGame(num_particle = 100, num_player = 3, a = 0.25)
    solver = ESCFRSolver(Dict{POSGInfoKey, MCInfoState}(), game)
    train!(solver, 1000)
end


main()