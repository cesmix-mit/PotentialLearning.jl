using Clustering

include("kabsch.jl")

"""
    struct DBSCANSelector <: SubsetSelector
        clusters
        eps
        minpts
        sample_size
    end

Definition of the type DBSCANSelector, a subselector based on the clustering method DBSCAN.
"""
struct DBSCANSelector <: SubsetSelector
    clusters
    eps
    minpts
    sample_size
end

"""
    function DBSCANSelector(
        ds::DataSet,
        eps,
        minpts,
        sample_size
    )

Constructor of DBSCANSelector based on the atomic configurations in `ds`, the DBSCAN params `eps` and `minpts`, and the sample size `sample_size`.
"""
function DBSCANSelector(
    ds::DataSet,
    eps,
    minpts,
    sample_size
)
    return DBSCANSelector(get_clusters(ds, eps, minpts), eps, minpts, sample_size)
end

"""
    function get_random_subset(
        s::DBSCANSelector,
        batch_size = s.sample_size
    )
    
Returns a random subset of indexes composed of samples of size `batch_size ÷ length(s.clusters)` elements from each cluster in `s`.
"""
function get_random_subset(
    s::DBSCANSelector,
    batch_size = s.sample_size
)
    inds = reduce(vcat, sample.(s.clusters, [batch_size ÷ length(s.clusters)]))
    return inds
end

"""
    function sample(
        c,
        batch_size
    )
    
Select from cluster `c` a sample of size `batch_size`.
"""
function sample(
    c,
    batch_size
)
    return c[rand(1:length(c), batch_size)]
end

"""
    function get_clusters(
        ds,
        eps,
        minpts
    )

Computes clusters from the configurations in `ds` using DBSCAN with parameters `eps` and `minpts`.
"""
function get_clusters(
    ds,
    eps,
    minpts
)
    # Create distance matrix
    if any(boundary_conditions(get_system(ds[1])) .== [Periodic()])
        d = Symmetric(distance_matrix_periodic(ds))
    else
        d = Symmetric(distance_matrix_kabsch(ds))
    end
    # Create clusters using dbscan
    c = dbscan(d, eps, minpts)
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    return clusters
end

"""
    function periodic_rmsd(
        p1::Array{Float64,2},
        p2::Array{Float64,2},
        box_lengths::Array{Float64,1}
    )
    
Calculates the RMSD between atom positions of two configurations taking into account the periodic boundaries.
"""
function periodic_rmsd(
    p1::Array{Float64,2},
    p2::Array{Float64,2},
    box_lengths::Array{Float64,1}
)
    n_atoms = size(p1, 1)
    distances = zeros(n_atoms)
    for i in 1:n_atoms
        d = p1[i, :] - p2[i, :]
        # If d is larger than half the box length subtract box length
        d = d .- round.(d ./ box_lengths) .* box_lengths
        distances[i] = norm(d)
    end
    return sqrt(mean(distances .^2))
end

"""
    function distance_matrix_periodic(
        ds::DataSet
    )

Calculates a matrix of distances between atomic configurations taking into account the periodic boundaries.
"""
function distance_matrix_periodic(ds::DataSet)
    n = length(ds); d = zeros(n, n)
    box = bounding_box(get_system(ds[1]))
    box_lengths = [get_values(box[i])[i] for i in 1:3]
    Threads.@threads for i in 1:n
        if bounding_box(get_system(ds[i])) != box
            error("Periodic box must be the same for all configurations.")
        end
        pi = Matrix(hcat(get_values.(get_positions(ds[i]))...)')
        for j in i+1:n
            pj = Matrix(hcat(get_values.(get_positions(ds[j]))...)')
            d[i,j] = periodic_rmsd(pi, pj, box_lengths)
            d[j,i] = d[i,j]
        end
    end
    return d
end

"""
    function distance_matrix_kabsch(
        ds::DataSet
    )

Calculate a matrix of distances between atomic configurations using KABSCH method.
"""
function distance_matrix_kabsch(
    ds::DataSet
)
    n = length(ds); d = zeros(n, n)
    Threads.@threads for i in 1:n
        p1 = Matrix(hcat(get_values.(get_positions(ds[i]))...)')
        for j in i+1:n
            p2 = Matrix(hcat(get_values.(get_positions(ds[j]))...)')
            d[i,j] = kabsch_rmsd(p1, p2)
            d[j,i] = d[i,j]
        end
    end
    return d
end


