using AtomsBase
using Unitful, UnitfulAtomic
using LinearAlgebra
using Distributions

# initialize some fake descriptors
d = 8
num_atoms = 20
num_configs = 10

ld = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
ld = LocalDescriptors.(ld)
ds = DataSet(Configuration.(ld))

@test typeof(ld[1]) <: LocalDescriptors
@test typeof(ld[1].b[1]) <: LocalDescriptor
@test length(ld) == num_configs
@test length(ld[1]) == num_atoms
@test typeof(ds) <: DataSet
@test length(ds) == num_configs

## features
gm = GlobalMean()
cm = CorrelationMatrix()
@test typeof(gm) <: Feature
@test typeof(cm) <: Feature

f_gm = compute_feature.(ld, (gm,))
f_cm = compute_feature.(ld, (cm,))
@test typeof(f_gm[1]) <: Vector{Float64}
@test typeof(f_cm[1]) <: Symmetric{Float64,Matrix{Float64}}
@test compute_features(ds, gm) == f_gm
@test compute_features(ds, cm) == f_cm

mvn = MvNormal(zeros(d), I(d))
f_mvn = [rand(mvn) for j = 1:100]

## distances 
fo = Forstner(1e-16)
e = Euclidean(d)
@test typeof(fo) <: Distance
@test typeof(e) <: Distance
@test compute_distance(f_gm[1], f_gm[1], e) < eps()
@test compute_distance(f_gm[1], f_gm[2], e) > 0.0
@test compute_distance(f_cm[1], f_cm[1], e) < eps()
@test compute_distance(f_cm[1], f_cm[2], e) > 0.0
@test compute_distance(f_cm[1], f_cm[1], fo) < eps()
@test compute_distance(f_cm[1], f_cm[2], fo) > 0.0

@test abs(sum(compute_gradx_distance(f_gm[1], f_gm[1], e))) < eps()
@test abs(sum(compute_gradx_distance(f_gm[1], f_gm[2], e))) > 0.0
@test abs(sum(compute_grady_distance(f_gm[1], f_gm[1], e))) < eps()
@test abs(sum(compute_grady_distance(f_gm[1], f_gm[2], e))) > 0.0
@test compute_gradxy_distance(f_gm[1], f_gm[1], e) == -2.0*I(d)
@test compute_gradxy_distance(f_gm[1], f_gm[2], e) == -2.0*I(d)

## kernels 
dp = DotProduct()
rbf_e = RBF(e)
rbf_fo = RBF(fo)

@test typeof(dp) <: Kernel
@test typeof(rbf_e) <: Kernel
@test typeof(rbf_fo) <: Kernel

@test compute_kernel(f_gm[1], f_gm[1], dp) ≈ 1.0
@test compute_kernel(f_gm[1], f_gm[1], rbf_e) ≈ 1.0
@test compute_kernel(f_cm[1], f_cm[1], dp) ≈ 1.0
@test compute_kernel(f_cm[1], f_cm[1], rbf_e) ≈ 1.0
@test compute_kernel(f_cm[1], f_cm[1], rbf_fo) ≈ 1.0

@test compute_kernel(f_gm[1], f_gm[2], dp) > 0
@test compute_kernel(f_gm[1], f_gm[2], rbf_e) > 0
@test compute_kernel(f_cm[1], f_cm[2], dp) > 0
@test compute_kernel(f_cm[1], f_cm[2], rbf_e) > 0
@test compute_kernel(f_cm[1], f_cm[2], rbf_fo) > 0

@test abs(sum(compute_gradx_kernel(f_gm[1], f_gm[1], rbf_e))) < eps()
@test abs(sum(compute_gradx_kernel(f_gm[1], f_gm[2], rbf_e))) > 0
@test abs(sum(compute_grady_kernel(f_gm[1], f_gm[1], rbf_e))) < eps()
@test abs(sum(compute_grady_kernel(f_gm[1], f_gm[2], rbf_e))) > 0
@test compute_gradxy_kernel(f_gm[1], f_gm[1], rbf_e) == I(8)
@test abs(sum(compute_gradxy_kernel(f_gm[1], f_gm[2], rbf_e))) > 0

@test typeof(KernelMatrix(f_gm, dp)) <: Symmetric{Float64,Matrix{Float64}}
@test typeof(KernelMatrix(f_cm, dp)) <: Symmetric{Float64,Matrix{Float64}}
@test typeof(KernelMatrix(f_gm, rbf_e)) <: Symmetric{Float64,Matrix{Float64}}
@test typeof(KernelMatrix(f_cm, rbf_e)) <: Symmetric{Float64,Matrix{Float64}}
@test typeof(KernelMatrix(f_cm, rbf_fo)) <: Symmetric{Float64,Matrix{Float64}}


# divergences
std_gauss_score(x) = -x
ksd = KernelSteinDiscrepancy(score=std_gauss_score, kernel=rbf_e)

@test typeof(ksd) <: Divergence
@test compute_divergence(f_mvn, ksd) < compute_divergence(f_gm, ksd)

