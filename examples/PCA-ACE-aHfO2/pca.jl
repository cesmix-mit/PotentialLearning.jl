# New functions to reduce dimension of dataset descriptors #####################

mutable struct PCAState <: DimensionReducer
    tol
    λ
    W
    m
end

function PCAState(; tol = 0.01, λ = [], W = [], m = [])
    PCAState(tol, λ, W, m)
end

function fit!(ds::DataSet, pca::PCAState)
    d = try
        #vcat(get_values.(get_local_descriptors.(ds))...) # use local desc
        sum.(get_values.(get_local_descriptors.(ds))) # use global desc
    catch
        error("No local descriptors found in DataSet")
    end
    d = try
        f = get_values.(get_force_descriptors.(ds))
        ff = vcat(vcat(fd...)...)
        return vcat(d, ff)
    catch
        d
    end
    if pca.m == []
        pca.m = sum(d) / length(d)
    end
    dm = d .- [pca.m] # center desc
    pca.λ, pca.W = select_eigendirections(dm, pca.tol)
    nothing
end

function transform!(ds::DataSet, dr::DimensionReducer)
    ds̃ = try
        ldc = get_values.(get_local_descriptors.(ds))
        ml = dr.m / length(ldc[1]) # compute local mean
        ldc_new = [LocalDescriptors([(dr.W' * (l .- ml)) for l in ld])
                   for ld in ldc]
        ds .+ ldc_new
    catch
        ds
    end
    ds̃ = try
        fdc = get_values.(get_force_descriptors.(ds))
        fdc_new = [ForceDescriptors([[(dr.W' * (fc .- dr.m)) for fc in f] for f in fd])
                   for fd in fdc]
        ds̃ .+ fdc_new
    catch
        ds̃
    end
    ds̃ = DataSet(ds̃)
    copyto!(ds.Configurations, ds̃.Configurations)
end

function PotentialLearning.select_eigendirections(d::Vector{T}, tol::Int) where {T<:Vector{<:Real}}
    λ, ϕ = PotentialLearning.compute_eigen(d)
    λ, ϕ = λ[end:-1:1], ϕ[:,end:-1:1] # reorder by columns instead of rows
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    W = ϕ[:, 1:tol]
    return λ, W
end

## Do not delete


#ld = vcat(get_values.(get_local_descriptors.(ds_train))...)
#m = sum(ld) / length(ld)
#ldm = ld .- [m]
#λ, W = select_eigendirections(ldm, 2) # this functions order phi with column-major order

#using MultivariateStats
#ld_mat = hcat(ld...)
#M = MultivariateStats.fit(MultivariateStats.PCA, ld_mat; maxoutdim=2)

#W
#M.proj

#x = rand(26)
#predict(M, x)
#W' * (x - m)

#    function Q(c::Configuration)
#        ϕ = sum(get_values(get_local_descriptors(c)))
#        return 0.5 * dot(ϕ, ϕ)
#    end
#    function ∇Q(c::Configuration)
#        ϕ = sum(get_values(get_local_descriptors(c)))
#        return ϕ
#    end
#    as = ActiveSubspace(Q, ∇Q, n_desc)
#    λ_l, W_l = fit(ds_train, as)
#    ds_train
#    ds_train = W_l * ds_train

#function fit_pca(d, tol)
#    m = [mean(d[:,i]) for i in 1:size(d)[2]]
#    dc = reduce(hcat,[d[:,i] .- m[i] for i in 1:size(d)[2]])
#    Q = Symmetric(mean(dc[i,:]*dc[i,:]' for i in 1:size(dc,1)))
#    λ, ϕ = eigen(Q)
#    λ, ϕ = λ[end:-1:1], ϕ[:, end:-1:1] # reorder by column
#    Σ = 1.0 .- cumsum(λ) / sum(λ)
#    W = ϕ[1:tol, :] # W = ϕ[:, Σ .> tol]
#    return λ, W, m
#end

#function get_dim_red_pars(ds, tol)
#    lll = get_values.(get_local_descriptors.(ds))
#    lll_mat = Matrix(hcat(vcat(lll...)...)')
#    λ_l, W_l, m_l = fit_pca(lll_mat, tol)

#    fff = get_values.(get_force_descriptors.(ds))
#    fff_mat = Matrix(hcat(vcat(vcat(fff...)...)...)')
#    λ_f, W_f, m_f = fit_pca(fff_mat, tol)

#    return λ_l, W_l, m_l, lll, λ_f, W_f, m_f, fff
#end

#function reduce_desc(λ_l, W_l, m_l, lll, λ_f, W_f, m_f, fff)
#    e_descr = [LocalDescriptors([((l .- m_l)' * W_l')' for l in ll ]) for ll in lll]
#    #f_descr = [ForceDescriptors([[((fc .- m_f)' * W_f')' for fc in f] for f in ff]) for ff in fff]
#    f_descr = [ForceDescriptors([[(fc' * W_l')' for fc in f] for f in ff]) for ff in fff]
#    return e_descr, f_descr
#end


## Definition of a linear problem. Changes descriptors calculations (only one line)
#"""
#    LinearProblem(ds::DatasSet; T = Float64)

#Construct a LinearProblem by detecting if there are energy descriptors and/or force descriptors and construct the appropriate LinearProblem (either Univariate, if only a single type of descriptor, or Covariate, if there are both types).
#"""
#function PotentialLearning.LinearProblem(ds::DataSet; T = Float64)
#    d_flag, descriptors, energies = try
#        #true,  compute_features(ds, GlobalSum()), get_values.(get_energy.(ds))
#        true, sum.(get_values.(get_local_descriptors.(ds))), get_values.(get_energy.(ds))
#        
#        #true, compute_feature.(get_local_descriptors.(ds), [GlobalSum()]), get_values.(get_energy.(ds))
#    catch 
#        false, 0.0, 0.0 
#    end
#    fd_flag, force_descriptors, forces = try  
#        true, [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds], get_values.(get_forces.(ds))
#    catch
#        false, 0.0, 0.0
#    end
#    if d_flag & ~fd_flag 
#        dim = length(descriptors[1])
#        β = zeros(T, dim)

#        p = UnivariateLinearProblem(descriptors, 
#                energies, 
#                β, 
#                [1.0],
#                Symmetric(zeros(dim, dim)))
#    elseif ~d_flag & fd_flag 
#        dim = length(force_descriptors[1][1])
#        β = zeros(T, dim)

#        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
#        p = UnivariateLinearProblem(force_descriptors,
#            [reduce(vcat, fi) for fi in forces], 
#            β, 
#            [1.0], 
#            Symmetric(zeros(dim, dim))
#        )
#        
#    elseif d_flag & fd_flag 
#        dim_d = length(descriptors[1])
#        dim_fd = length(force_descriptors[1][1])

#        if  (dim_d != dim_fd) 
#            error("Descriptors and Force Descriptors have different dimension!") 
#        else
#            dim = dim_d
#        end

#        β = zeros(T, dim)
#        forces =  [reduce(vcat, fi) for fi in forces]
#        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]

#        p = PotentialLearning.CovariateLinearProblem(energies,
#                [reduce(vcat, fi) for fi in forces], 
#                descriptors, 
#                force_descriptors, 
#                β, 
#                [1.0], 
#                [1.0], 
#                Symmetric(zeros(dim, dim)))

#    else 
#        error("Either no (Energy, Descriptors) or (Forces, Force Descriptors) in DataSet")
#    end
#    p
#end



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

