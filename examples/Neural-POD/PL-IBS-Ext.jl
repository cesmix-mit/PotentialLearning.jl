using Glob
using NaturalSort

struct POD <: BasisSystem
    pod_params
end

function POD(;args...)
    return POD(args)
end

function compute_local_descriptors(
    confs::DataSet,
    pod::POD;
    T = Float32, 
    ds_path = "../data/HfO2/",
    lammps_path = "../../POD/lammps/build/lmp"
)
    params = pod.pod_params
    data = OrderedDict()
    data[:file_format] = "extxyz"
    data[:file_extension] = "extxyz"
    data[:path_to_training_data_set] = "train"
    data[:path_to_test_data_set] = "test"
    data[:compute_pod_descriptors] = 1

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

# Load local descriptors
function load_local_descriptors(
    confs::DataSet,
    pod::POD;
    T = Float32, 
    ds_path = "../data/HfO2/train/"
)
    file_names = sort(glob("$ds_path/localdescriptors_config*.bin"), lt=natural)
    e_des = Vector{LocalDescriptors}(undef, length(confs))
    for (j, file_desc) in enumerate(file_names)
        row_data = reinterpret(Float64, read(file_desc))
        n_atoms = convert(Int, row_data[1])
        n_desc = convert(Int, row_data[2])
        ld = reshape(row_data[3:end], n_atoms, n_desc)
        e_des[j] = PotentialLearning.LocalDescriptors([T.(ld_i) for ld_i in eachrow(ld)])
    end
    return e_des
end

#ds_train_1 = ds_train[rand(1:length(ds_train), 200)]
#ds_train = ds_train[rand(1:length(ds_train), length(ds_train))]

# Load global energy descriptors
#gd = []
#open("global_energy_descriptors.dat") do f
#    linecounter = 0
#    for l in eachline(f)
#        d = parse.(Float32, split(replace(l, "\n" => ""), " "))
#        push!(gd, d)
#        linecounter += 1
#    end
#end
#n_desc = length(gd[1])using Glob


