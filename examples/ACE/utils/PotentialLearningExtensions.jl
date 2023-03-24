import PotentialLearning

# New function to read XYZ files ###############################################
include("xyz.jl")



# New function to Split datasets ###############################################

function Base.split(ds, n, m)
    ii = randperm(length(ds))
    return @views ds[first(ii, n)], ds[last(ii, m)]
end

# New functions to reduce dimension of dataset descriptors #####################
# TODO: adapt these functions to current interfaces

function fit_pca(d, tol)
    m = [mean(d[:,i]) for i in 1:size(d)[2]]
    dc = reduce(hcat,[d[:,i] .- m[i] for i in 1:size(d)[2]])
    Q = Symmetric(mean(dc[i,:]*dc[i,:]' for i in 1:size(dc,1)))
    λ, ϕ = eigen(Q)
    λ, ϕ = λ[end:-1:1], ϕ[:, end:-1:1] # reorder by column
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    W = ϕ[1:tol, :] # W = ϕ[:, Σ .> tol]
    return λ, W, m
end

function get_dim_red_pars(ds, tol)
    lll = get_values.(get_local_descriptors.(ds))
    lll_mat = Matrix(hcat(vcat(lll...)...)')
    λ_l, W_l, m_l = fit_pca(lll_mat, tol)

    fff = get_values.(get_force_descriptors.(ds))
    fff_mat = Matrix(hcat(vcat(vcat(fff...)...)...)')
    λ_f, W_f, m_f = fit_pca(fff_mat, tol)

    return λ_l, W_l, m_l, lll, λ_f, W_f, m_f, fff
end

function reduce_desc(λ_l, W_l, m_l, lll, λ_f, W_f, m_f, fff)
    e_descr = [LocalDescriptors([((l .- m_l)' * W_l')' for l in ll ]) for ll in lll]
    f_descr = [ForceDescriptors([[((fc .- m_f)' * W_f')' for fc in f] for f in ff]) for ff in fff]
    return e_descr, f_descr
end


# LBasisPotential is not exported in InteratomicBasisPotentials.jl / basis_potentials.jl
# These functions should be removed once export issue is fixed.
struct LBasisPotential
    basis
    β
end
function LBasisPotential(basis)
    return LBasisPotential(basis, zeros(length(basis)))
end


# New learning function based on weigthed least squares ########################
function learn!(lb::LBasisPotential, ds::DataSet; w_e = 1.0, w_f = 1.0)

    lp = PotentialLearning.LinearProblem(ds)
    
    @views B_train = reduce(hcat, lp.B)'
    @views dB_train = reduce(hcat, lp.dB)'
    @views e_train = lp.e
    @views f_train = reduce(vcat, lp.f)
    
    # Calculate A and b.
    @views A = [B_train; dB_train]
    @views b = [e_train; f_train]

    # Calculate coefficients β.
    Q = Diagonal([w_e * ones(length(e_train));
                  w_f * ones(length(f_train))])
    β = (A'*Q*A) \ (A'*Q*b)

    #copyto!(lp.β, β)
    #copyto!(lp.σe, w_e)
    #copyto!(lp.σf, w_f)
    copyto!(lb.β, β)
end




## Dont erase


# λ_pca, W_pca = fit(ds_train_1, PCA()) # Current implementation

#using MultivariateStats, Statistics
#s = open("a-Hfo2-300K-NVT-6000-NACE/locdesc.dat") do file
#    read(file, String)
#end
#a = eval(Meta.parse(s))
#b = reduce(hcat,[a[:,i] .- mean(a[:,i]) for i in 1:size(a)[2]])
#M = MultivariateStats.fit(MultivariateStats.PCA, c)
#R = predict(M, c)


#using MultivariateStats, Statistics
#a = vcat(get_values.(get_local_descriptors.(ds_train_1))...)
#b = Matrix(hcat(vcat(get_values.(get_local_descriptors.(ds_train_1))...)...)')
##c = reduce(hcat,[b[:,1] .- mean(b[:,1]) for i in 1:size(b)[2]])

#Q = Symmetric(mean(di*di' for di in foreachrow(a)))

#using LinearAlgebra

#Qa = Symmetric(Symmetric(mean(a[i,:]*a[i,:]' for i in 1:size(a,1))))
#Qc = Symmetric(Symmetric(mean(c[i,:]*c[i,:]' for i in 1:size(c,1))))

#λ, ϕ = eigen(Qa)
#λ, ϕ = λ[end:-1:1], ϕ[end:-1:1, :] # reorder

#Σ = 1.0 .- cumsum(λ) / sum(λ)
#tol = 0.00001
#W = ϕ[Σ .> tol, :]
#λ, W

