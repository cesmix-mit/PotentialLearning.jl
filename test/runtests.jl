include("../src/PotentialLearning.jl")
using .PotentialLearning: load_learning_params, load_dft_data, learn, validate
using Test

include("GaN-SNAP-LAMMPS-Test.jl")
