using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic, StaticArrays
using InteratomicPotentials, InteratomicBasisPotentials
using CairoMakie
using JLD
using DPP
push!(Base.LOAD_PATH, dirname(@__DIR__))
using PotentialLearning

using Flux, Zygote
using Polynomials, SpecialPolynomials

#################### Importing Data ###################
# Import Raw Data

# Import configurations 
ds, thermo = load_data("examples/LJ/data/lj.yaml", YAML(:Ar, u"eV", u"Å"));
ds = ds[3:end];
systems = get_system.(ds);
positions = position.(systems)

# plot distances 
# size_inches = (12, 10)
# size_pt = 72 .* size_inches
# fig = Figure(resolution = size_pt, fontsize =16)
# ax1 = Axis(fig[1,1], xlabel = "τ", ylabel = "Distance from Origin (Å)")
# ax2 = Axis(fig[2,1], xlabel = "τ", ylabel = "Energy (eV)")
# dists_origin = map( x->ustrip.(norm.(x)), positions )
# for i = 1:13
#     lines!(ax1, 0.005:0.005*100:(0.005*1E6), map(x->x[i], dists_origin))
# end
# lines!(ax2, 0.005:0.005*100:(0.005*1E6), get_values.(get_energy.(ds)))
# save("examples/LJ/figures/lj_dist_from_origin_energy.pdf", fig)

# Get configurations 
n_body = 2  # 2-body
max_deg = 8 # 8 degree polynomials
r0 = 1.0 # minimum distance between atoms
rcutoff = 5.0 # cutoff radius 
wL = 1.0 # Defaults, See ACE.jl documentation 
csp = 1.0 # Defaults, See ACE.jl documentation
ace = ACE([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)
local_descriptors = compute_local_descriptors.(systems, (ace,))
force_descriptors = compute_force_descriptors.(systems, (ace,))
lb = LBasisPotential(ace)
ds = ds .+ LocalDescriptors.(local_descriptors) .+ ForceDescriptors.(force_descriptors)
ds = DataSet(ds)

## Score Matching 
## E[ tr(∇x(S(x;θ)) + 0.5|s(x;θ)|^2 ]
# let e(x) = 0.5*x'*A*x + b*x 
# then s(x) = A*x + b 

struct l
    A 
    b
end
(ℓ::l)(x) = ℓ.A * x .+ ℓ.b
ll = l(I(8) + zeros(8, 8), zeros(8))

tr∇s(x) = 8.0
loss(ll, x) = tr∇s(x) + 0.5*norm(ll(x))^2 
∇θloss(ll, x) = first(gradient(ell->loss(ell, x), ll))
function loss(ll::l, c::Configuration) 
    ld = get_values(get_local_descriptors(c))
    l = mean( loss.( (ll, ), ld)  )
    g = ∇θloss.( (ll, ), ld)
    gA = Symmetric(mean(map(x->x.A, g)))
    gb = mean(map(x->x.b, g))
    l, gA, gb
end

γ = 5e-2
for i = 1:50
    batch_inds = randperm(length(ds))[1:100]
    l_temp = 0.0
    gA = zeros(8, 8)
    gb = zeros(8)
    for c in ds[batch_inds]
        temp_l, gA_temp, gb_temp = loss(ll, c)
        l_temp += temp_l / 100.0
        gA += gA_temp / 100.0
        gb += gb_temp / 100.0
    end

    if i%1 == 0
        println("i = $i  l = $(round(l_temp, digits=3)) |gA| = $(norm(gA)) |gb| = $(norm(gb))")
    end

    broadcast!(-, ll.A, ll.A, γ*gA)
    broadcast!(-, ll.b, ll.b, γ*gb)
end







using Flux
# Range
xrange = 0:π/99:π
xs = [ Float32.([x1, x2]) for x1 in xrange for x2 in xrange]
# NN model: multi-layer perceptron (MLP)
mlp = Chain(Dense(2,4, Flux.σ),Dense(4,1))
ps_mlp = Flux.params(mlp)
E_mlp(x) = sum(mlp(x))
dE_mlp(x) = first(ForwarDiff.gradient(E_mlp, x))
# Computing "nested" gradient

Zygote.gradient(x->sum(dE_mlp(x)), ps_mlp)