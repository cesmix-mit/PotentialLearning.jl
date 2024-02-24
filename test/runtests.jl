# TODO: Create tests.

using PotentialLearning
using Test

@testset "PotentialLearning.jl" begin

    @testset "IO Tests" begin
        include("io/extxyz_test.jl")
        include("io/yaml_test.jl")
    end
    @testset "Kernel Tests" begin
        include("kernels/kernel_tests.jl")
    end
    @testset "Subset Selector Tests" begin
        include("subset_selector/subset_selector_tests.jl")
    end
    @testset "Dimension Reduction Tests" begin
        include("dimension_reduction/dimension_reduction_tests.jl")
    end
    @testset "Learning Tests" begin
        include("learning/linear_tests.jl")
    end
    @testset "Data Tests" begin 
        include("data/data_utils_tests.jl")
    end
end
