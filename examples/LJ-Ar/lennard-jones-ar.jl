# # Load Ar dataset with energies computed by Lennard-Jones and postprocess

# ## a. Load packages and define paths.

# Load packages.
using Unitful, UnitfulAtomic
using AtomsBase, InteratomicPotentials, PotentialLearning
using LinearAlgebra, Plots, DisplayAs

# Define paths.
base_path = match(r"^(.*/PotentialLearning/)", @__DIR__).match
ds_path   = "$base_path/examples/data/LJ-AR/lj-ar.yaml"

# ## b. Load atomistic dataset.
ds, thermo = load_data(ds_path, YAML(:Ar, u"eV", u"Å"))
ds = @views ds[2:end] # Filter first configuration (zero energy)

# ## c. Compute distance from origin, extract LJ energies, and define time range.

# Get atom positions and compute distance from origin.
systems = get_system.(ds)
n_atoms = length(first(systems)) # Note: in this dataset all systems contain the same no. of atoms
positions = position.(systems)
dists_origin = map(x->ustrip.(norm.(x)), positions)

# Extract LJ energies from dataset.
energies = get_values.(get_energy.(ds))

# Define time range.
time_range = 0.5:0.5:5000

# ## d. Post-process data.

# Plot distance from origin vs time.
p = plot(xlabel = "τ | ps",
         ylabel = "Distance from origin | Å", 
         dpi = 300, fontsize = 12)
for i = 1:n_atoms
    plot!(time_range, map(x->x[i], dists_origin), label="")
end
DisplayAs.PNG(p)

# Plot LJ energies vs time.
p = plot(time_range, energies,
         xlabel = "τ | ps",
         ylabel = "Lennard Jones energy | eV",
         dpi = 300, fontsize = 12)
DisplayAs.PNG(p)
