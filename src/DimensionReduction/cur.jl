"""
The CUR matrix decomposition is a dimension reduction method. It approximates a given matrix by 
selecting a few of its columns (C), a few of its rows (R), and a small intersection matrix (U), 
such that the product of these three matrices closely approximates the original matrix. 
This technique is particularly useful for dimensionality reduction in large datasets because it 
retains a subset of the original data's structure, making it interpretable and efficient for 
large-scale data analysis.

Three varients of CUR are implemented in PotentialLearning.jl: LinearTimeCUR, DEIMCUR, and LSCUR.

"""

struct CUR{T<:Real} <: DimensionReducer
    rows::Vector{Int64}
    cols::Vector{Int64}
end

function CUR(rows::Vector{Int64}, cols::Vector{Int64})
    CUR(rows, cols)
end

function LinearTimeCUR(A::Matrix{T}, k::Int64) where {T<:Number}
    m, n = size(A)
    C = zeros(T, m, k)
    R = zeros(T, k, n)

    fsq_normA = norm(A)^2
    colsq_norm = [ norm(A[:, j])^2 for j in range(1, n)] 
    rowsq_norm = [ norm(A[i, :])^2 for i in range(1, m)] 
    
    col_p = [ (colsq_norm[j]/fsq_normA) for j in range(1, n)] 
    row_p = [ (rowsq_norm[j]/fsq_normA) for j in range(1, m)] 
    
    #computing C and R based on uniform random sampling of rows and cols of matrix A 
    cols = Vector{Int64}(undef, k)
    rows = Vector{Int64}(undef, k)

    for i in range(1, k)
        cols[i] = sample(1:n, ProbabilityWeights(col_p))
        C[:, i] =  A[:, cols[i]] ./ sqrt(k*col_p[cols[i]])
        rows[i] = sample(1:m, ProbabilityWeights(row_p))
        R[i, :] =   A[rows[i], :] ./ sqrt(k*row_p[rows[i]])
    end

    return rows, cols
end

function DEIMCUR(A::Matrix{T}, k::Int64) where {T<:Number}

    m, n = size(A)
    C = zeros(T, m, k)
    R = zeros(T, k, n)
    u, s, vh = svd(A)

    U = u[:, 1:k]
    V = vh[:, 1:k]


    rows = Vector{Int64}(undef, k)
    cols = Vector{Int64}(undef, k)

    for i in range(1, k)
        rows[i] = first.(Tuple.(findall(x -> x==maximum(abs.(U[:,i])), abs.(U))))[1]
        cols[i] = first.(Tuple.(findall(x -> x==maximum(abs.(V[:,i])), abs.(V))))[1]

        @time U_p = pinv(U[rows[i], 1:i])'
        @time mul!(U[:, i+1:k], U[:, 1:i], U_p * U[rows[i],i+1:k]')

        @time V_p = pinv(V[cols[i], 1:i])'
        @time mul!(V[:, i+1:k], V[:, 1:i], V_p * V[cols[i],i+1:k]')
    end

    return rows, cols  
end

function LSCUR_ColSelect(::Type{T}, A::Matrix{T}, k::Int64) where {T<:Number}

    m, n = size(A)
    F = zeros(T, m, k)

    m, n = size(A)
    u, s, vh = svd(A)
    

    V = vh[:, 1:k]

    V = (V.^2)

    prob = [sum(V[i, :]) for i in range(1, m)]

    idx = Vector{Int64}(undef, k)

    for i in range(1, k)
        idx[i] = sample(1:n, ProbabilityWeights(prob))
    end

    F = A[:, idx]

    return F, idx
end

function LSCUR(A::Matrix{T}, k::Int64) where {T<:Number}

    m, n = size(A)
    C = zeros(T, m, k)
    R = zeros(T, k, n)

    C, cols = LSCUR_ColSelect(T, A, k)

    R, rows = LSCUR_ColSelect(T, Matrix(transpose(A)), k)

    return rows, cols   
end
