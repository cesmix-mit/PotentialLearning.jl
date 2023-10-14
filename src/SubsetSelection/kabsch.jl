"""
The following source code is based on BiomolecularStructures.jl.

See https://github.com/hng/BiomolecularStructures.jl/blob/a8c8970f2cbbdf4ec05bd1245a61e3ddab2a6380/src/KABSCH/kabsch.jl

The MIT License (MIT)
Copyright (c) [2015] [Simon Malischewski Henning Schumann]
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
    function rmsd(
        A::Array{Float64,2},
        B::Array{Float64,2}
    )

Calculate root mean square deviation of two matrices A, B.
See http://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions
"""
function rmsd(
    A::Array{Float64,2},
    B::Array{Float64,2}
)

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

"""
    function calc_centroid(
        m::Array{Float64,2}
    )
    
Calculate a centroid of a matrix.
"""
function calc_centroid(
    m::Array{Float64,2}
)

    sum_m::Array{Float64,2} = sum(m, dims=1)
    size_m::Int64 = size(m)[1]

    return map(x -> x/size_m, sum_m)
end

"""
    function translate_points(
        P::Array{Float64,2},
        Q::Array{Float64,2}
    )

Translate P, Q so centroids are equal to the origin of the coordinate system
Translation der Massenzentren, so dass beide Zentren im Ursprung des Koordinatensystems liegen
"""
function translate_points(
    P::Array{Float64,2},
    Q::Array{Float64,2}
)
    # Calculate centroids P, Q
    # Die Massenzentren der Proteine
    centroid_p::Array{Float64,2} = calc_centroid(P)
    centroid_q::Array{Float64,2} = calc_centroid(Q)
    
    P = broadcast(-,P, centroid_p)
    
    Q = broadcast(-,Q, centroid_q)

    return P, Q, centroid_p, centroid_q
end

"""
    function kabsch(
        reference::Array{Float64,2},
        coords::Array{Float64,2}
    )

Input: two sets of points: reference, coords as Nx3 Matrices (so) 
Returns optimally rotated matrix 
"""
function kabsch(
    reference::Array{Float64,2},
    coords::Array{Float64,2}
)

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

"""
    function kabsch_rmsd(
        P::Array{Float64,2},
        Q::Array{Float64,2}
    )

Directly return RMSD for matrices P, Q for convenience.
"""
function kabsch_rmsd(
    P::Array{Float64,2},
    Q::Array{Float64,2}
)
    return rmsd(P,kabsch(P,Q))
end
