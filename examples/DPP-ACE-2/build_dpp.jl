using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using CairoMakie
using JLD
using DPP
using PotentialLearning

#################### Importing Data ###################
# Import Raw Data
energies, descriptors = JLD.load(
    "examples/aHfO2/data/aHfO2_diverse_descriptors_3600.jld",
    "energies",
    "descriptors",
)

# Create structs for data types 
energies = Energy.(energies, (u"eV",))
descriptors = LocalDescriptors.(descriptors)


# Create configurations  (EnergyConfigurations)
configs = Configuration.(energies, descriptors)
ds = DataSet(configs)

################# Kernel #####################
# Choose feature 
gm = GlobalMean()

# Construct Kernel 
k_dot = DotProduct()

# KDPP
dpp = kDPP(ds, gm, k_dot; batch_size = 100)

# Access Kernel Matrix: dpp.K.l

# Calculate inclusion probabilities
ips = get_inclusion_prob(dpp)
describe(ips)

################### Plots #####################

size_inches = (12, 8)
size_pt = 72 .* size_inches
fig = Figure(resolution = size_pt, fontsize = 12)
ax1 = Axis(
    fig[1, 1],
    xlabel = "Inclusion Probability",
    title = "Distribution of DPP Inclusion Probabilities",
)
hist!(ax1, ips, bins = 30)

ax2 = Axis(
    fig[1, 2],
    xlabel = "Number of Oxygen Atoms",
    ylabel = "Energy (eV)",
    title = "DPP Inclusion Probabilities vs. Configuration Energy",
)
scatter!(
    ax2,
    [length(c.B) - 32 for c in configs],
    [e.d for e in energies],
    color = ips,
    markersize = 8.0,
)

ax3 = Axis(
    fig[2, 1],
    ylabel = "Inclusion Probability",
    xlabel = "Energy (eV)",
    title = "DPP Inclusion Probabilities vs. Configuration Energy",
)
scatter!(ax3, [e.d for e in energies], ips, markersize = 4.0)

ax4 = Axis(
    fig[2, 2],
    ylabel = "Inclusion Probability",
    xlabel = "Number of Atoms",
    title = "DPP Inclusion Probabilities vs. Number of Oxygen Atoms",
)
scatter!(ax4, [length(c.B) - 32 for c in configs], ips, markersize = 4.0)

save("examples/aHfO2/figures/aHfO2_dpp_inclusion_probabilities.pdf", fig)
