
# Compute local descriptors of a basis system and dataset using threads
function compute_local_descriptors(ds::DataSet, basis::BasisSystem; pbar = true, T = Float64)
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

# Compute force descriptors of a basis system and dataset using threads
function compute_force_descriptors(ds::DataSet, basis::BasisSystem; pbar = true, T = Float64)
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


