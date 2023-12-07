
function PotentialLearning.get_all_energies(graphs::Vector, gnn::GNNChain)
    return [first(gnn(g, g.x)) for g in graphs]
end

function dist(x, y)
    return sqrt(sum((x .- y).^2))
end

function compute_adjacency_matrix(
    positions,
    bounding_box;
    rcutoff = 5.0u"Å",
    normalize = true
)
    n = length(positions)
    adjacency = Matrix{Bool}(undef, n, n)
    degrees = zeros(n) # place to store the diagonal of degree matrix
    for i in 1:n
        @simd for j in i:n
            adjacency[i, j] = minimum([
                dist(positions[i], positions[j]),
                dist((positions[i] + bounding_box[1]), positions[j]),
                dist((positions[i] + bounding_box[2]), positions[j]),
                dist((positions[i] + bounding_box[3]), positions[j])
            ]) < rcutoff
            adjacency[j, i] = adjacency[i, j]
            degrees[i] += adjacency[i, j]
            degrees[j] += adjacency[i, j]
        end
    end
    if normalize # see https://arxiv.org/abs/1609.02907
        adjacency = Diagonal(degrees)^(-1/2)*adjacency*Diagonal(degrees)^(-1/2)
    end
    return Symmetric(adjacency)
end

function compute_graphs(
    ds::DataSet;
    rcutoff = 5.0u"Å",
    normalize = true
)
    graphs = []
    for c in ds
        adj = compute_adjacency_matrix(get_positions(c),
                                       bounding_box(get_system(c)),
                                       rcutoff = rcutoff, 
                                       normalize = normalize)
        #at_ids = atomic_number(get_system(c))
        #at_mat = reduce(hcat, [OneHotAtom(i) for i in at_ids])
        ld_mat = hcat(get_values(get_local_descriptors(c))...)
        g = GNNGraph(adj, ndata = (; x = ld_mat), #, y = at_ids),
                          gdata = (; z = get_values(get_energy(c)))) |> device
        push!(graphs, g)
    end
    return graphs
end

