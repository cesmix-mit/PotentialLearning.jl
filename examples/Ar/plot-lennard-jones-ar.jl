push!(Base.LOAD_PATH, "../../")

using Unitful, UnitfulAtomic
using AtomsBase, InteratomicPotentials, PotentialLearning
using LinearAlgebra, CairoMakie

# Load dataset: Lennard-Jones + Argon
ds, thermo = load_data("data/lj-ar.yaml", YAML(:Ar, u"eV", u"Å"))

# Filter first configuration (zero energy)
ds = ds[2:end]

# Compute distance from origin, LJ energies, and time range
systems = get_system.(ds)
n_atoms = length(first(systems)) # Note: in this dataset all systems contain the same no. of atoms
positions = position.(systems)
dists_origin = map(x->ustrip.(norm.(x)), positions)
energies = get_values.(get_energy.(ds))
time_range = 0.5:0.5:5000

# Plot distance from origin vs time, and LJ energies vs time
size_inches = (12, 10)
size_pt = 72 .* size_inches
fig = Figure(resolution = size_pt, fontsize = 16)
ax1 = Axis(fig[1,1], xlabel = "τ | ps", ylabel = "Distance from origin | Å")
ax2 = Axis(fig[2,1], xlabel = "τ | ps", ylabel = "Lennard Jones energy | eV")
for i = 1:n_atoms
    lines!(ax1, time_range, map(x->x[i], dists_origin))
end
lines!(ax2, time_range, energies)
save("figures/dist2origin-ljenergy-time.pdf", fig)

