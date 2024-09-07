include("POSG.jl")
include("CFRsolver.jl")

function main()
    # pb = InitParticleBelief(100, 0.008, 3)
    # x0 = [(3^0.5)/8 1/8; -(3^0.5)/8 1/8; 0 -0.25]
    # x1 = x0 + [0.1 0;  0 0.1; 0.1 0]
    # pb = UpdateBelief(pb, [0.1 0;  0 0.1; 0.1 0], x1)
    # x1 = x0 + [0.1 0;  -0.1 0; 0.1 0]
    # pb = UpdateBelief(pb, [0.1 0;  -0.1 0; 0.1 0], x1)
    # uniform distribution range [-a, a]
    game = TagGame(num_particle = 100, num_player = 3, a = 0.001)
    solver = ESCFRSolver(Dict{POSGInfoKey, MCInfoState}(), game, 0)
    train!(solver, 1000)
end


main()