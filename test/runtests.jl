# TODO: Create tests.

using PotentialLearning
using Test

@testset "PotentialLearning.jl" begin

    @testset "IO Tests.jl" begin
        include("io/extxyz_test.jl")
        include("io/yaml_test.jl")
    end
    @testset "Kernel Tests.jl" begin
        include("kernels/kernel_tests.jl")
    end
end
