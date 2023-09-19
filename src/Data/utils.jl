# Get all energies and forces ##################################################

"""
function get_all_energies(
    ds::DataSet
)

"""
function get_all_energies(
    ds::DataSet
)
    return [get_values(get_energy(ds[c])) for c in 1:length(ds)]
end

"""
    function get_all_forces(
        ds::DataSet
    )

"""
function get_all_forces(
    ds::DataSet
)
    return reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c]))
                                    for c in 1:length(ds)]))
end

"""
    function get_all_energies(
        ds::DataSet,
        lb::LinearBasisPotential
    )

"""
function get_all_energies(
    ds::DataSet,
    lb::LinearBasisPotential
)
    Bs = sum.(get_values.(get_local_descriptors.(ds)))
    return lb.β0[1] .+ dot.(Bs, [lb.β])
end

"""
function get_all_forces(
    ds::DataSet,
    lb::LinearBasisPotential
)

"""
function get_all_forces(
    ds::DataSet,
    lb::LinearBasisPotential
)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    return vcat([lb.β0[1] .+  dB' * lb.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
end

# Compute local and force descriptors ##########################################

"""
function compute_local_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true,
    T = Float64
)

Compute local descriptors of a basis system and dataset using threads.
"""
function compute_local_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true,
    T = Float64
)
    iter = collect(enumerate(get_system.(ds)))
    if pbar
        iter = ProgressBar(iter)
    end
    e_des = Vector{LocalDescriptors}(undef, length(ds))
    Threads.@threads for (j, sys) in iter
        e_des[j] = LocalDescriptors([T.(d) for d in compute_local_descriptors(sys, basis)])
    end
    return e_des
end

"""
function compute_force_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true,
    T = Float64
)

Compute force descriptors of a basis system and dataset using threads
"""
function compute_force_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true,
    T = Float64
)
    iter = collect(enumerate(get_system.(ds)))
    if pbar
        iter = ProgressBar(iter)
    end
    f_des = Vector{ForceDescriptors}(undef, length(ds))
    Threads.@threads for (j, sys) in iter
        f_des[j] = ForceDescriptors([[ T.(fi[i, :]) for i = 1:3] 
                                     for fi in compute_force_descriptors(sys, basis)])
    end
    return f_des
end

