push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random, Statistics, Zygote
using CairoMakie

# Load atomic configurations
confs, _ = load_data("data/lj-ar.yaml", YAML(:Ar, u"eV", u"Å"))

# Define ACE
ace = ACE(species = [:Ar],         # species
          body_order = 2,          # 2-body
          polynomial_degree = 8,   # 8 degree polynomials
          wL = 1.0,                # Defaults, See ACE.jl documentation 
          csp = 1.0,               # Defaults, See ACE.jl documentation 
          r0 = 1.0,                # minimum distance between atoms
          rcutoff = 5.0)           # cutoff radius 

# Update training dataset by adding energy (local) and force descriptors
println("Computing local descriptors of training dataset")
e_descr_train = compute_local_descriptors(confs, ace)

ds = DataSet(confs .+ e_descr_train)

## Score Matching 
## E[ tr(∇x(S(x;θ)) + 0.5|s(x;θ)|^2 ]
# let e(x) = 0.5*x'*A*x + b*x 
# then s(x) = A*x + b 

# Define new linear type
struct l
    A::Any
    b::Any
end
(ℓ::l)(x) = ℓ.A * x .+ ℓ.b

n = length(first(first(e_descr_train))) # no. of local descriptors
ll = l(I(n) + zeros(n, n), zeros(n))

# Define loss function
tr∇s(x) = n # ?
loss(ll, x) = tr∇s(x) + 0.5 * norm(ll(x))^2
∇θloss(ll, x) = first(gradient(ell -> loss(ell, x), ll))
function loss(ll::l, c::Configuration)
    ld = get_values(get_local_descriptors(c))
    l = =
    g = ∇θloss.((ll,), ld)
    gA = Symmetric(mean(map(x -> x.A, g)))
    gb = mean(map(x -> x.b, g))
    l, gA, gb
end

# Learn
γ = 5e-2
for i = 1:200
    batch_inds = randperm(length(ds))[1:100]
    l_temp = 0.0
    gA = zeros(n, n)
    gb = zeros(n)
    for c in ds[batch_inds]
        temp_l, gA_temp, gb_temp = loss(ll, c)
        l_temp += temp_l / 100.0
        gA += gA_temp / 100.0
        gb += gb_temp / 100.0
    end

    if i % 1 == 0
        println(
            "i = $i  l = $(round(l_temp, digits=3)) |gA| = $(norm(gA)) |gb| = $(norm(gb))",
        )
    end
    broadcast!(-, ll.A, ll.A, γ * gA)
    broadcast!(-, ll.b, ll.b, γ * gb)
end
