# Linear learning problem data types and constructors #########################

"""
abstract type LinearProblem{T<:Real} <: AbstractLearningProblem end

An abstract type to specify linear potential inference problems. 
"""
abstract type LinearProblem{T<:Real} <: AbstractLearningProblem end

"""
struct UnivariateLinearProblem{T<:Real} <: LinearProblem{T}
    iv_data::Vector
    dv_data::Vector
    β::Vector{T}
    β0::Vector{T}
    σ::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
end

A UnivariateLinearProblem is a linear problem in which there is only 1 type of independent variable / dependent variable. Typically, that means we are either only fitting energies or only fitting forces. When this is the case, the solution is available analytically and the standard deviation, σ, and covariance, Σ, of the coefficients, β, are computable. 
"""
struct UnivariateLinearProblem{T<:Real} <: LinearProblem{T}
    iv_data::Vector
    dv_data::Vector
    β::Vector{T}
    β0::Vector{T}
    σ::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
end
Base.show(io::IO, u::UnivariateLinearProblem{T}) where {T} =
    print(io, "UnivariateLinearProblem{T, $(u.β), $(u.σ)}")

"""
struct CovariateLinearProblem{T<:Real} <: LinearProblem{T}
    e::Vector
    f::Vector{Vector{T}}
    B::Vector{Vector{T}}
    dB::Vector{Matrix{T}}
    β::Vector{T}
    β0::Vector{T}
    σe::Vector{T}
    σf::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
end

A CovariateLinearProblem is a linear problem in which we are fitting energies and forces using both descriptors and their gradients (B and dB, respectively). When this is the case, the solution is not available analytically and must be solved using some iterative optimization proceedure. In the end, we fit the model coefficients, β, standard deviations corresponding to energies and forces, σe and σf, and the covariance Σ. 
"""
struct CovariateLinearProblem{T<:Real} <: LinearProblem{T}
    e::Vector
    f::Vector{Vector{T}}
    B::Vector{Vector{T}}
    dB::Vector{Matrix{T}}
    β::Vector{T}
    β0::Vector{T}
    σe::Vector{T}
    σf::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
end

Base.show(io::IO, u::CovariateLinearProblem{T}) where {T} =
    print(io, "CovariateLinearProblem{T, $(u.β), $(u.σe), $(u.σf)}")

"""
function LinearProblem(
    ds::DataSet;
    T = Float64
)

Construct a LinearProblem by detecting if there are energy descriptors and/or force descriptors and construct the appropriate LinearProblem (either Univariate, if only a single type of descriptor, or Covariate, if there are both types).
"""
function LinearProblem(
    ds::DataSet
)
    d_flag, descriptors, energies = try
        true, sum.(get_values.(get_local_descriptors.(ds))), get_values.(get_energy.(ds))
    catch
        false, 0.0, 0.0
    end
    fd_flag, force_descriptors, forces = try
        true,
        [reduce(vcat, get_values(get_force_descriptors(dsi))) for dsi in ds],
        get_values.(get_forces.(ds))
    catch
        false, 0.0, 0.0
    end
    if d_flag & ~fd_flag
        dim = length(descriptors[1])
        β = zeros(dim)
        β0 = zeros(1)

        p = UnivariateLinearProblem(
            descriptors,
            energies,
            β,
            β0,
            [1.0],
            Symmetric(zeros(dim, dim)),
        )
    elseif ~d_flag & fd_flag
        dim = length(force_descriptors[1][1])
        β = zeros(dim)
        β0 = zeros(1)

        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
        p = UnivariateLinearProblem(
            force_descriptors,
            [reduce(vcat, fi) for fi in forces],
            β,
            β0,
            [1.0],
            Symmetric(zeros(dim, dim)),
        )

    elseif d_flag & fd_flag
        dim_d = length(descriptors[1])
        dim_fd = length(force_descriptors[1][1])

        if (dim_d != dim_fd)
            error("Descriptors and Force Descriptors have different dimension!")
        else
            dim = dim_d
        end

        β = zeros(dim)
        β0 = zeros(1)
        forces = [reduce(vcat, fi) for fi in forces]
        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]

        p = CovariateLinearProblem(
            energies,
            [reduce(vcat, fi) for fi in forces],
            descriptors,
            force_descriptors,
            β,
            β0,
            [1.0],
            [1.0],
            Symmetric(zeros(dim, dim)),
        )

    else
        error("Either no (Energy, Descriptors) or (Forces, Force Descriptors) in DataSet")
    end
    p
end

# Linear learning functions common to OLS and WLS implementations #############

"""
function learn!(
    iap::InteratomicPotentials.LinearBasisPotential,
    ds::DataSet,
    args...
)

Learning dispatch function, common to ordinary and weghted least squares implementations.
"""
function learn!(
    lb::InteratomicPotentials.LinearBasisPotential,
    ds::DataSet,
    args...
)
    lp = LinearProblem(ds)
    learn!(lp, args...)

    resize!(lb.β, length(lp.β))
    lb.β .= lp.β
    lb.β0 .= lp.β0
    return lp
end
