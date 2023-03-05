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
energies1, descriptors1 = JLD.load(
    "examples/aHfO2/data/aHfO2_diverse_descriptors_3601_6000.jld",
    "energies",
    "descriptors",
)
energies = [energies; energies1];
descriptors = [descriptors; descriptors1]
# Create structs for data types 
energies = Energy.(energies, (u"eV",))
descriptors = LocalDescriptors.(descriptors)


# Create configurations  (EnergyConfigurations)
configs = Configuration.(energies, descriptors)
ds = DataSet(configs)

ds_ = ds[1:100:end]
################# Kernel #####################
# Precompute Features for simplicity
gms = compute_features(ds_, GlobalMean());

C = Symmetric(pinv(cov(gms) + 1e-3 * I(716), 1e-3) + 1e-3 * I(716))
cms = compute_features(ds_, CorrelationMatrix());

# Global Mean, Dot Product Kernel Matrix 
dpp_gm_dp = kDPP(gms, DotProduct(); batch_size = 20)
dpp_gm_rbf = kDPP(gms, RBF(Euclidean(C)); batch_size = 20)
dpp_cm_dp = kDPP(cms, DotProduct(); batch_size = 20)
dpp_cm_rbf = kDPP(cms, RBF(Euclidean(C)); batch_size = 20)
dpp_cm_fo = kDPP(cms, RBF(Forstner(716)); batch_size = 20)


dpps = [dpp_gm_dp, dpp_gm_rbf, dpp_cm_dp, dpp_cm_fo]
ips = get_inclusion_prob.(dpps)
ks = [dppi.K.L[triu(trues(360, 360), 1)] for dppi in dpps]

################### Plots #####################


size_inches = (16, 16)
size_pt = 72 .* size_inches
fig = Figure(resolution = size_pt, fontsize = 12)
labels = [L"D: GM+DP", L"D: GM+RBF", L"D: CM+DP", L"D: CM+FO"]
iplabels = [L"IP: $GM+DP$", L"IP: $GM+RBF$", L"IP: $CM+DP$", L"IP: $CM+FO$"]
for i = 1:4
    for j = 1:(i-1)
        ax = Axis(fig[i, j], xlabel = labels[j], ylabel = labels[i])
        scatter!(ax, ks[j], ks[i], color = (:red, 0.2), markersize = 1.0)
    end
    ax = Axis(fig[i, i], xlabel = labels[i], xlabelcolor = :red, ylabelcolor = :red)
    ax1 = Axis(
        fig[i, i],
        xlabel = iplabels[i],
        xlabelcolor = :blue,
        xaxisposition = :top,
        yaxisposition = :right,
    )
    hist!(ax, ks[i], color = (:red, 0.4))
    hist!(ax1, ips[i], color = (:blue, 0.4))

    for j = i+1:4
        ax = Axis(fig[i, j], xlabel = iplabels[j], ylabel = iplabels[i])
        scatter!(ax, ips[j], ips[i], color = (:blue, 0.2), markersize = 4.0)
    end

end

save("examples/aHfO2/figures/aHfO2_multiple_dpp_distances_inclusion_probabilities.pdf", fig)
