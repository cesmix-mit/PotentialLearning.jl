using Unitful, UnitfulAtomic
using AtomsBase, InteratomicPotentials, PotentialLearning
using LinearAlgebra, Plots

# Load dataset: Lennard-Jones + Argon
path = joinpath(dirname(pathof(PotentialLearning)), "../examples/LJ-Ar")
ds, thermo = load_data("$path/../data/LJ-AR/lj-ar.yaml", YAML(:Ar, u"eV", u"Å"))

# Filter first configuration (zero energy)
ds = ds[2:end]

# Compute distance from origin, LJ energies, and time range
systems = get_system.(ds)
n_atoms = length(first(systems)) # Note: in this dataset all systems contain the same no. of atoms
positions = position.(systems)
dists_origin = map(x->ustrip.(norm.(x)), positions)
energies = get_values.(get_energy.(ds))
time_range = 0.5:0.5:5000

# Plot distance from origin vs time
p = plot(xlabel = "τ | ps",
         ylabel = "Distance from origin | Å", 
         dpi = 300, fontsize = 12)
for i = 1:n_atoms
    plot!(time_range, map(x->x[i], dists_origin), label="")
end
p

# Plot LJ energies vs time
plot(time_range, energies,
     xlabel = "τ | ps",
     ylabel = "Lennard Jones energy | eV",
     dpi = 300, fontsize = 12)

