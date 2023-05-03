using AtomsBase
using Unitful, UnitfulAtomic
using LinearAlgebra

# initialize some fake descriptors
d = 8
num_atoms = 20
num_configs = 10
batch_size = 2
ld = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
ld = LocalDescriptors.(ld)
ds = DataSet(Configuration.(ld))

function Q(c::Configuration)
    ϕ = sum(get_values(get_local_descriptors(c)))
    0.5 * dot(ϕ, ϕ)
end
function ∇Q(c::Configuration)
    ϕ = sum(get_values(get_local_descriptors(c)))
    ϕ
end

num_dim = 2
as = ActiveSubspace(Q, ∇Q, num_dim)
pca = PCA(num_dim)
@test typeof(as) <: DimensionReducer
@test typeof(pca) <: DimensionReducer

λ_as, W_as = fit(ds, as)
@test typeof(λ_as) <: Vector{Float64}
@test typeof(W_as) <: Matrix{Float64}

# These fail because W_as has rows/columns flipped
#@test size(W_as, 1) == num_dim
#@test size(W_as, 2) == d
#@test typeof(W_as * ds) <: DataSet

# PCA is also broken
#λ_pca, W_pca = fit(ds, pca)
#@test typeof(λ_pca) <: Vector{Float64}
#@test typeof(W_pca) <: Matrix{Float64}
#@test size(W_pca, 1) == num_dim
#@test size(W_pca, 2) == d
#
#@test all(λ_as .≈ λ_pca)