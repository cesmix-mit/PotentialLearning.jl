struct LBasisPotentialExt{T} <: LinearBasisPotential{NamedTuple{(:β, :β0)}, NamedTuple{()}}
    β
    β0
    basis
end

function LBasisPotentialExt(basis :: BasisSystem; T = Float64)
    return LBasisPotentialExt{T}(zeros(length(basis)), zeros(1), basis)
end
