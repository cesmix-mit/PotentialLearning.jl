using Glob
using NaturalSort

struct POD <: BasisSystem
    pod_params
end

function POD(;args...)
    return POD(args)
end

function compute_descriptors(
    confs::DataSet,
    pod::POD;
    ds_path = "../data/HfO2/",
    lammps_path = "../../POD/lammps/build/lmp"
)

    run(`rm -f $(ds_path)/train/\*bin`)
    run(`rm -f $(ds_path)/test/\*bin`)

    params = pod.pod_params
    data = OrderedDict()
    data[:file_format] = "extxyz"
    data[:file_extension] = "extxyz"
    data[:path_to_training_data_set] = "train"
    data[:path_to_test_data_set] = "test"
    data[:compute_pod_descriptors] = 2

    run(`mkdir -p $ds_path`)

    # Create data.pod
    open("$ds_path/data.pod", "w") do io
        [println(io, "$k $v") for (k, v) in data]
    end

    # Create params.pod
    open("$ds_path/params.pod", "w") do io
        [println(io, "$k $v") for (k, v) in params]
    end
    
    # Create fit.pod
    open("$ds_path/fit.pod", "w") do io
        println(io, "fitpod params.pod data.pod")
    end
    
    # fit pod using lammps
    run(Cmd(`mpirun -n 1 $lammps_path -in fit.pod`, dir=ds_path))

end

function load_global_descriptors(
    confs::DataSet,
    pod::POD;
    ds_path = "../data/HfO2/train/"
)
    file_names = sort(glob("$ds_path/globaldescriptors_config*.bin"), lt=natural)
    e_des = []
    for (j, file_desc) in enumerate(file_names)
        row_data = reinterpret(Float64, read(file_desc))
        n_atoms = convert(Int, row_data[1])
        n_desc = convert(Int, row_data[2])
        gd = row_data[3:end]
        push!(e_des, gd)
    end
    return stack(e_des)'
end

function load_local_descriptors(
    confs::DataSet,
    pod::POD;
    ds_path = "../data/HfO2/train/"
)
    file_names = sort(glob("$ds_path/localdescriptors_config*.bin"), lt=natural)
    e_des = Vector{LocalDescriptors}(undef, length(confs))
    for (j, file_desc) in enumerate(file_names)
        raw_data = reinterpret(Float64, read(file_desc))
        n_atoms = convert(Int, raw_data[1])
        n_desc = convert(Int, raw_data[2])
        ld = reshape(raw_data[3:end], n_atoms, n_desc)
        e_des[j] = PotentialLearning.LocalDescriptors([Float64.(ld_i) for ld_i in eachrow(ld)])
    end
    return e_des
end

#function compute_global_descriptors(
#    B::Vector{Vector{T}},
#    sys::AbstractSystem,
#    bp::BasisPotential
#) where T<: Real
#    species = unique(atomic_symbol.(sys))
#    species_id = Dict(s => i for (i, s) in enumerate(species))
#    n_desc = length(B[1])
#    desc = Dict(s => zeros(n_desc) for s in species)
#    for (i, s) in enumerate(atomic_symbol.(sys))
#        println(i,", ", s)
#        desc[s] = desc[s] + B[i]
#    end
#    gd = vcat(values(desc)...)
#    return gd
#end
