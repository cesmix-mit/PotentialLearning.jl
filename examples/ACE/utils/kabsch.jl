# Source: https://github.com/hng/BiomolecularStructures.jl/blob/master/docs/kabsch.md


# Calculate root mean square deviation of two matrices A, B
# http://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions
function rmsd(A::Array{Float64,2}, B::Array{Float64,2})

    RMSD::Float64 = 0.0

    # D pairs of equivalent atoms
    D::Int = size(A)[1]::Int # <- oddly _only_ changing this to Int makes it work on 32-bit systems.
    # N coordinates
    N::Int = length(A)::Int

    for i::Int64 = 1:N
        RMSD += (A[i]::Float64 - B[i]::Float64)^2
    end
    return sqrt(RMSD / D)
end

# calculate a centroid of a matrix
function calc_centroid(m::Array{Float64,2})

    sum_m::Array{Float64,2} = sum(m, dims=1)
    size_m::Int64 = size(m)[1]

    return map(x -> x/size_m, sum_m)
end

# Translate P, Q so centroids are equal to the origin of the coordinate system
# Translation der Massenzentren, so dass beide Zentren im Ursprung des Koordinatensystems liegen
function translate_points(P::Array{Float64,2}, Q::Array{Float64,2})
    # Calculate centroids P, Q
    # Die Massenzentren der Proteine
    centroid_p::Array{Float64,2} = calc_centroid(P)
    centroid_q::Array{Float64,2} = calc_centroid(Q)
    
    P = broadcast(-,P, centroid_p)
    
    Q = broadcast(-,Q, centroid_q)

    return P, Q, centroid_p, centroid_q
end

# Input: Two sets of points: reference, coords as Nx3 Matrices (so)
# returns optimally rotated matrix 
function kabsch(reference::Array{Float64,2}, coords::Array{Float64,2})

    centered_reference::Array{Float64,2}, centered_coords::Array{Float64,2}, centroid_p::Array{Float64,2}, centroid_q::Array{Float64,2}  = translate_points(reference, coords)
    # Compute covariance matrix A
    A::Array{Float64,2} = *(centered_coords', centered_reference)        

    # Calculate Singular Value Decomposition (SVD) of A
    u::Array{Float64,2}, d::Array{Float64,1}, vt::Array{Float64,2} = svd(A)

    # check for reflection
    f::Int64 = sign(det(vt) * det(u))
    m::Array{Int64,2} = [1 0 0; 0 1 0; 0 0 f]

    # Calculate the optimal rotation matrix _and_ superimpose it
    return broadcast(+, *(centered_coords, u, m, vt'), centroid_p)

end

# directly return RMSD for matrices P, Q for convenience
function kabsch_rmsd(P::Array{Float64,2}, Q::Array{Float64,2})
    return rmsd(P,kabsch(P,Q))
end
