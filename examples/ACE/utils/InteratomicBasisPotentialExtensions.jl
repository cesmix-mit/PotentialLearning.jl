include("ace.jl")
include("NNIAP.jl")


Base.size(fd::ForceDescriptors) = (length(fd), )

# Compute descriptors of a basis system and dataset
function compute_local_descriptors(ds::DataSet, basis::BasisSystem)
    e_des = Vector{LocalDescriptors}(undef, length(ds))
    #for (j, sys) in ProgressBar(zip(1:length(ds), get_system.(ds)))
    Threads.@threads for (j, sys) in ProgressBar(collect(enumerate(get_system.(ds))))
        e_des[j] = LocalDescriptors(compute_local_descriptors(sys, basis))
    end
    return e_des
end

function compute_force_descriptors(ds::DataSet, basis::BasisSystem)
    f_des = Vector{ForceDescriptors}(undef, length(ds))
    #for (j, sys) in ProgressBar(zip(1:length(ds), get_system.(ds)))
    Threads.@threads for (j, sys) in ProgressBar(collect(enumerate(get_system.(ds))))
        f_des[j] = ForceDescriptors([[fi[i, :] for i = 1:3] 
                                     for fi in compute_force_descriptors(sys, basis)])
    end
    return f_des
end


