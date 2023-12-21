# Get all energies and forces from a dataset ###################################

"""
function get_all_energies(
    ds::DataSet
)

`ds`: dataset.
"""
function get_all_energies(
    ds::DataSet
)
    return [get_values(get_energy(ds[c]))
            for c in 1:length(ds)]
end

"""
    function get_all_forces(
        ds::DataSet
    )

`ds`: dataset.
"""
function get_all_forces(
    ds::DataSet
)
    return reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c]))
                                    for c in 1:length(ds)]))
end

# Get energies and forces of a dataset and basis potential #####################

"""
    function get_all_energies(
        ds::DataSet,
        bp::BasisPotential
    )

`ds`: dataset.
`bp`: basis potential.
"""
function get_all_energies(
    ds::DataSet,
    bp::BasisPotential
)
    return [potential_energy(ds[i], bp)
            for i in 1:length(ds)]
end

"""
function get_all_forces(
    ds::DataSet,
    bp::BasisPotential
)

`ds`: dataset.
`bp`: basis potential.
"""
function get_all_forces(
    ds::DataSet,
    bp::BasisPotential
)
    return reduce(vcat,reduce(vcat, [force(ds[c], bp)
                                     for c in 1:length(ds)]))
end


# Get energies and forces for a configuration and basis potential ##############

"""
function potential_energy(
    c::Configuration,
    bp::BasisPotential
)

`c`: atomic configuration.
`bp`: basis potential.
"""
function potential_energy(
    c::Configuration,
    bp::BasisPotential
)
    B = get_values(get_local_descriptors(c))
    e = InteratomicPotentials.potential_energy(B, bp)
    return e
end

"""
function force(
    c::Configuration,
    bp::BasisPotential
)

`c`: atomic configuration.
`bp`: basis potential.
"""
function force(
    c::Configuration,
    bp::BasisPotential
)
    dB = get_values(get_force_descriptors(c))
    f = InteratomicPotentials.force(dB, bp)
    return f
end

# Get energies and forces for a configuration and neural basis potential #######

"""
function potential_energy(
    c::Configuration,
    nnbp::NNBasisPotential
)

`c`: atomic configuration.
`nnbp`: neural network basis potential.
"""
function potential_energy(
    c::Configuration,
    nnbp::NNBasisPotential
)
    B = get_values(get_local_descriptors(c))
    s = atomic_symbol.(get_system(c).particles) # mutable version: atomic_symbol(get_system(c))
    e = InteratomicPotentials.potential_energy(B, s, nnbp)
    return e
end

"""
function force(
    c::Configuration,
    nnbp::NNBasisPotential
)

`c`: atomic configuration.
`nnbp`: neural network basis potential.
"""
function force(
    c::Configuration,
    nnbp::NNBasisPotential
)
    dB = get_values(get_force_descriptors(c))
    s = atomic_symbol.(get_system(c).particles) # mutable version: atomic_symbol(get_system(c))
    f = InteratomicPotentials.force(dB, s, nnbp)
    return f
end


# Compute local and force descriptors for a dataset and basis system ###########

"""
function compute_local_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true,
    T = Float64
)

`ds`: dataset.
`basis`: basis system (e.g. ACE)
`pbar`: progress bar
`T`: descriptor number type

Compute local descriptors of a basis system and dataset using threads.
"""
function compute_local_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true
)
    iter = collect(enumerate(get_system.(ds)))
    if pbar
        iter = ProgressBar(iter)
    end
    e_des = Vector{LocalDescriptors}(undef, length(ds))
    Threads.@threads for (j, sys) in iter
        e_des[j] = LocalDescriptors(compute_local_descriptors(sys, basis))
    end
    return e_des
end

"""
function compute_force_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true
)

Compute force descriptors of a basis system and dataset using threads.
"""
function compute_force_descriptors(
    ds::DataSet,
    basis::BasisSystem;
    pbar = true
)
    iter = collect(enumerate(get_system.(ds)))
    if pbar
        iter = ProgressBar(iter)
    end
    f_des = Vector{ForceDescriptors}(undef, length(ds))
    Threads.@threads for (j, sys) in iter
        f_des[j] = ForceDescriptors([[ fi[i, :] for i = 1:3]
                                     for fi in compute_force_descriptors(sys, basis)])
    end
    return f_des
end

