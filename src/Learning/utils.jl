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
