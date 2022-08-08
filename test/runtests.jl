# TODO: Create tests.

include("../src/PotentialLearning.jl")
using .PotentialLearning
using Test

@testset "PotentialLearning.jl" begin

    @testset "IO Tests.jl" begin
        include("io/extxyz_test.jl")
        include("io/lammps_test.jl")
    end
end
