## Discrepancies
""" 
    Divergence

    A struct of abstract type Divergence produces a measure of discrepancy between two probability distributions. Discepancies may take as argument analytical distributions or sets of samples representing empirical distributions.
"""
abstract type Divergence end


"""
    KernelSteinDiscrepancy <: Divergence
        score :: Function
        knl :: Kernel

    Computes the kernel Stein discrepancy between distributions p (from which samples are provided) and q (for which the score is provided) based on the RKHS defined by kernel k.
"""
struct KernelSteinDiscrepancy <: Divergence
    score :: Function
    kernel :: Kernel
end

function KernelSteinDiscrepancy(; score, kernel)
    return KernelSteinDiscrepancy(score, kernel)
end


function compute_divergence(
    x :: Vector{T},
    div :: KernelSteinDiscrepancy,
) where T <: Union{Real, Vector{<:Real}}

    N = length(x)
    sq = div.score.(x)
    k = div.kernel

    ksd = 0.0
    for i = 1:N
        for j = i:N
            m = (i == j) ? 1 : 2
            sks = sq[i]' * compute_kernel(x[i], x[j], k) * sq[j]
            sk = sq[i]' * compute_grady_kernel(x[i], x[j], k)
            ks = compute_gradx_kernel(x[i], x[j], k)' * sq[j]
            trk = tr(compute_gradxy_kernel(x[i], x[j], k))

            ksd += m * (sks + sk + ks + trk) / (N*(N-1.0))
        end
    end
    return ksd
end