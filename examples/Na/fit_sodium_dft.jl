using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic, StaticArrays
using InteratomicPotentials, InteratomicBasisPotentials
using CairoMakie
using JLD
using Determinantal
push!(Base.LOAD_PATH, dirname(@__DIR__))
using PotentialLearning

#################### Importing Data ###################
# Import Raw Data

# Import configurations 
ds, thermo = load_data("examples/Sodium/data/liquify_sodium.yaml", YAML(:Na, u"eV", u"Å"));
ds, thermo = ds[220:end], thermo[220:end];
systems = get_system.(ds);

# Get configurations 
n_body = 4  # 2-body
max_deg = 8 # 8 degree polynomials
r0 = 1.0 # minimum distance between atoms
rcutoff = 5.0 # cutoff radius 
wL = 1.0 # Defaults, See ACE.jl documentation 
csp = 1.0 # Defaults, See ACE.jl documentation
ace = ACE([:Na], n_body, max_deg, wL, csp, r0, rcutoff)
local_descriptors = compute_local_descriptors.(systems, (ace,))
# force_descriptors = compute_force_descriptors.(systems, (ace,))
lb = LBasisPotential(ace)

# If we already have local_descriptors
# local_descriptors = JLD.load("examples/Sodium/data/sodium_empirical_full.jld", "descriptors");

# If we need to compute local_descriptors
local_descriptors = LocalDescriptors.(local_descriptors);
ds = ds .+ local_descriptors;

## Set up Train / Test split
ds_train, ds_test = DataSet(ds[1:1000]), DataSet(ds[1001:end]);

dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 200)


dpp_inds = get_random_subset(dpp)
lb, Σ = learn!(lb, ds_train[dpp_inds]; α = 1e-6)

############################################################################################################
####################################### Compute Errors ################################################
############################################################################################################
# Train Errors 
train_features = sum.(get_values.(get_local_descriptors.(ds_train)))
ê = dot.(train_features, (lb.β,) ./ 108.0)
e_true_train = get_values.(get_energy.(ds_train))
errors_train = (ê - e_true)
rel_errors_train = errors_train ./ e_true_train
describe(100.0 .* rel_errors_train)

# Test Errors
test_features = sum.(get_values.(get_local_descriptors.(ds_test)))
ê = dot.(test_features, (lb.β,) ./ 108.0)
e_true_test = get_values.(get_energy.(ds_test))
errors_test = (ê - e_true)
rel_errors_test = errors_test ./ e_true_test
describe(100.0 .* rel_errors_test)

dpp_inds2 = get_random_subset(dpp; batch_size = 20)
size_inches = (12, 8)
size_pt = 72 .* size_inches
fig = Figure(resolution = size_pt, fontsize = 16)
ax1 = Axis(fig[1, 1], xlabel = "Energy (eV)", ylabel = "Error (eV)")
# ax2 = Axis(fig[1,2], xlabel = "Energy (eV)", ylabel = "Relative Error (%)")
scatter!(ax1, e_true_train, errors_train, label = "Training", markersize = 5.0)
scatter!(ax1, e_true_test, errors_test, label = "Test", markersize = 5.0)
scatter!(
    ax1,
    e_true_train[dpp_inds2],
    errors_train[dpp_inds2],
    color = :darkred,
    label = "DPP Samples",
    markersize = 5.0,
)
# scatter!(ax2, e_true_train, -100.0*rel_errors_train, color = :blue, label = "Training", markersize = 3.0)
# scatter!(ax2, e_true_test, -100.0.*rel_errors_test, color = :red, label = "Test", markersize = 3.0)
axislegend(ax1)
save("examples/Sodium/figures/energy_error_training_test_scatter.pdf", fig)



######################## ACE MD data ######################

# Import configurations 
ds_ace, thermo_ace = load_data("examples/Sodium/data/liquify_sodium_ace.yaml", YAML(:Na, u"eV", u"Å"));
ds_ace, thermo_ace = ds_ace[220:end], thermo_ace[220:end]
systems_ace = get_system.(ds_ace);
local_descriptors_ace =
    LocalDescriptors.(
        JLD.load("examples/Sodium/data/sodium_ace_md_descriptors.jld", "local_descriptors")
    )
ds_ace = ds_ace .+ local_descriptors_ace
ace_energies = get_values.(get_energy.(ds_ace))
ace_features = mean.(get_values.(get_local_descriptors.(ds_ace)))
ê = dot.(ace_features, (lb.β,))

###
# GP 
### 


gmx = compute_features(ds, GlobalMean())
gmy = compute_features(ds_ace, GlobalMean())
fx = [errors_train; errors_test]

C = inv(sqrt.(Diagonal(cov(gmx))))
k = RBF(Euclidean(73); ℓ = 1.0)
Kxx = KernelMatrix(gmx, k)
Kxy = KernelMatrix(gmx, gmy, k)
Kyy = KernelMatrix(gmy, k)

γ = mean(abs2, errors_test)
L = Kxy' * pinv(Kxx + γ * I(2092))
fŷ = L * fx
σŷ = sqrt.(abs.(diag(Kyy - Kxy' * L')))


size_inches = (12, 8)
size_pt = 72 .* size_inches
fig = Figure(resolution = size_pt, fontsize = 16)
ax1 = Axis(fig[1, 1], xlabel = "Energy (eV)")
scatter!(ax1, ace_energies, markersize = 3.0)
save("examples/Sodium/figures/energy_error_prediction_scatter.pdf", fig)

####

em_energies = get_values.(get_energy.(ds))
ace_energies = get_values.(get_energy.(ds_ace))
em_temps = map(x -> x[2]["data"][5], thermo)
ace_temps = map(x -> x[2]["data"][5], thermo_ace)

size_inches = (12, 8)
size_pt = 72 .* size_inches
fig = Figure(resolution = size_pt, fontsize = 16)
ax1 = Axis(fig[1, 1], xlabel = "Temperature (K)", ylabel = "Energy (eV)")
ax2 = Axis(
    fig[1, 1],
    yaxisposition = :right,
    ylabel = "Probability of Retraining",
    ylabelcolor = :red,
)
scatter!(ax1, em_temps, em_energies, markersize = 3.0, label = "Data")
scatter!(ax1, ace_temps, ace_energies, markersize = 3.0, label = "ACE MD")
lines!(ax2, mids[1:end-15], reprob[1:end-15], color = :red, linewidth = 5.0)
axislegend(ax1, position = :rb)
ax2.xticklabelsvisible = false
ax2.xticklabelsvisible = false
ax2.xlabelvisible = false
ax2.ygridvisible = false

linkxaxes!(ax1, ax2)
save("examples/Sodium/figures/energy_temps_empirical_ace_scatter.pdf", fig)

bins = minimum(ace_temps):10:maximum(ace_temps)
mids = 0.5 * (bins[1:end-1] + bins[2:end])
probs = zeros(length(bins) - 1)
sigs = zeros(length(bins) - 1)
for i in eachindex(bins[1:end-1])
    probs[i] = mean(fŷ[bins[i].<ace_temps.<bins[i+1]])
    sigs[i] = mean(σŷ[bins[i].<ace_temps.<bins[i+1]])
end
mids = mids[.~isnan.(probs)]
sigs = sigs[.~isnan.(probs)]
probs = probs[.~isnan.(probs)]
reprob = (1.0 .- exp.(-0.5 * probs .^ 2 ./ sigs)) .^ 2
reprob[reprob.<0.4] .= reprob[reprob.<0.4] .^ 2
reprob[0.4 .< reprob .< 0.5] .= reprob[0.4 .< reprob .< 0.5] .^ 1.5









import JuLIP, ACE1pack
rpi = get_rpi(ace)
IP = JuLIP.MLIPs.combine(rpi, lb.β ./ 108.0);
ACE1pack.Export.export_ace("examples/Sodium/data/parameters.ace", IP);





ace = ACE(params...)
ds, thermo = load_data("examples/Sodium/data/liquify_sodium.yaml", YAML(:Na, u"eV", u"Å"));
lb = LBasisPotential(ace)
local_descriptors = compute_local_descriptors(systems, ace)
force_descriptors = compute_force_descriptors(systems, ace)
ds = ds + local_descriptors + force_descriptors;
## Set up Train / Test split
ds_train, ds_test = ds[1:1000], ds[1001:end];

# Build a DPP Model
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 200)
dpp_mode = get_dpp_mode(dpp)

# Fit with dpp_mode
lb, Σ = learn!(lb, ds_train[dpp_inds]; α = 1e-6)

# Fit with Batch Gradient Descent 
lb, Σ = learn!(lb, ds_train, dpp; α = 1e-6)
