include("ace.jl")
include("NNIAP.jl")


Base.size(fd::ForceDescriptors) = (length(fd), )

# Compute descriptors of a basis system and dataset
function compute_local_descriptors(ds::DataSet, basis::BasisSystem; T = Float64)
    e_des = Vector{LocalDescriptors}(undef, length(ds))
    #for (j, sys) in ProgressBar(zip(1:length(ds), get_system.(ds)))
    Threads.@threads for (j, sys) in ProgressBar(collect(enumerate(get_system.(ds))))
        e_des[j] = LocalDescriptors([T.(d) for d in compute_local_descriptors(sys, basis)])
    end
    return e_des
end

function compute_local_descriptors_unthreaded(ds::DataSet, basis::BasisSystem; T = Float64)
    e_des = Vector{LocalDescriptors}(undef, length(ds))
    #for (j, sys) in ProgressBar(zip(1:length(ds), get_system.(ds)))
    Threads.@threads for (j, sys) in collect(enumerate(get_system.(ds)))
        e_des[j] = LocalDescriptors([T.(d) for d in compute_local_descriptors(sys, basis)])
    end
    return e_des
end

function compute_force_descriptors(ds::DataSet, basis::BasisSystem; T = Float64)
    f_des = Vector{ForceDescriptors}(undef, length(ds))
    #for (j, sys) in ProgressBar(zip(1:length(ds), get_system.(ds)))
    Threads.@threads for (j, sys) in ProgressBar(collect(enumerate(get_system.(ds))))
        f_des[j] = ForceDescriptors([[ T.(fi[i, :]) for i = 1:3] 
                                     for fi in compute_force_descriptors(sys, basis)])
    end
    return f_des
end

function compute_force_descriptors_unthreaded(ds::DataSet, basis::BasisSystem; T = Float64)
    f_des = Vector{ForceDescriptors}(undef, length(ds))
    Threads.@threads for (j, sys) in collect(enumerate(get_system.(ds)))
        f_des[j] = ForceDescriptors([[ T.(fi[i, :]) for i = 1:3] for fi in compute_force_descriptors(sys, basis)])
    end
    return f_des
end

function compute_all_descriptors(ds::DataSet, basis::BasisSystem; T = Float64)
    task₁ = Threads.@spawn compute_local_descriptors_unthreaded(ds, basis; T = T)
    task₂ = Threads.@spawn compute_force_descriptors_unthreaded(ds, basis; T = T)

    local_descriptors = fetch(task₁)
    force_descriptors = fetch(task₂)

    return local_descriptors, force_descriptors
end



