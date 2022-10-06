using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using CairoMakie
using JLD
using DPP
include("./src/PotentialLearning.jl")
using .PotentialLearning

#################### Importing Data ###################
# Import Raw Data

ds, thermos = load_data("examples/Sodium/data/empirical_sodium_2d.yaml", YAML(u"eV", u"Å"));
ds = ds[2:end]
r = [ustrip(norm(r[1] - r[2])) for r in get_positions.(ds)]
e2d = get_values.(get_energy.(ds))
f2d = [get_values(vi)[1][1] for vi in get_forces.(ds)]

ds, thermos = load_data("examples/Sodium/data/empirical_sodium_3d.yaml", YAML(u"eV", u"Å"));
ds = ds[2:end]
θ = 1.0:0.02:π 
e3d = get_values.(get_energy.(ds))
f3d = [norm(get_values(vi)) for vi in get_forces.(ds)]

size_inches = (12, 12)
size_pt = 72 .* size_inches
fig = Figure(resolution = size_pt, fontsize = 16)
ax1 = Axis(fig[1, 1], xlabel = "r (Å)", ylabel = "Energy (eV)", title = "Pair Potential for Sodium")
ax2 = Axis(fig[1, 2], xlabel = "r (Å)", ylabel = "Force (eV / Å)", title = "Force Pair Potential for Sodium")
ax3 = Axis(fig[2, 1], xlabel = "θ (radian)", ylabel = "Energy (eV)", title = "3 Body Potential for Sodium")
ax4 = Axis(fig[2, 2], xlabel = "θ (radian)", ylabel = "|Force| (eV / Å)", title = "3 Body Potential for Sodium")
lines!(ax1, r, e2d, color = :black)
lines!(ax2, r, f2d, color = :black)
lines!(ax3, θ, e3d, color = :black)
lines!(ax4, θ, f3d, color = :black)

save("examples/Sodium/figures/2d_3d_potential_sodium.pdf", fig)
