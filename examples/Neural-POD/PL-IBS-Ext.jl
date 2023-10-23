using Glob
using NaturalSort

struct POD <: BasisSystem
    # chemical element symbols
    species
    # inner cut-off radius
    rin
    # outer cut-off radius
    rcut
    # polynomial degrees for radial basis functions
    bessel_polynomial_degree
    inverse_polynomial_degree
    # one-body potential
    onebody
    # two-body linear POD potential
    twobody_number_radial_basis_functions
    # three-body linear POD potential
    threebody_number_radial_basis_functions
    threebody_angular_degree
    # four-body linear POD potential
    fourbody_number_radial_basis_functions
    fourbody_angular_degree
    true4BodyDesc
    # five-body linear POD potential
    fivebody_number_radial_basis_functions
    fivebody_angular_degree
    # six-body linear POD potential
    sixbody_number_radial_basis_functions
    sixbody_angular_degree
    # seven-body linear POD potential
    sevenbody_number_radial_basis_functions
    sevenbody_angular_degree
end

function POD(;
    species = [:Hf, :O],
    rin = 1.0,
    rcut = 7.5,
    bessel_polynomial_degree = 4,
    inverse_polynomial_degree = 10,
    onebody = 1,
    twobody_number_radial_basis_functions = 3,
    threebody_number_radial_basis_functions = 3,
    threebody_angular_degree = 3,
    fourbody_number_radial_basis_functions = 0,
    fourbody_angular_degree = 0,
    true4BodyDesc = 0,
    fivebody_number_radial_basis_functions = 0,
    fivebody_angular_degree = 0,
    sixbody_number_radial_basis_functions = 0,
    sixbody_angular_degree = 0,
    sevenbody_number_radial_basis_functions = 0,
    sevenbody_angular_degree = 0)
    return  POD(species,
                rin,
                rcut,
                bessel_polynomial_degree,
                inverse_polynomial_degree,
                onebody,
                twobody_number_radial_basis_functions,
                threebody_number_radial_basis_functions,
                threebody_angular_degree,
                fourbody_number_radial_basis_functions,
                fourbody_angular_degree,
                true4BodyDesc,
                fivebody_number_radial_basis_functions,
                fivebody_angular_degree,
                sixbody_number_radial_basis_functions,
                sixbody_angular_degree,
                sevenbody_number_radial_basis_functions,
                sevenbody_angular_degree)
end


# Harcoded function to compute local descriptors
function compute_local_descriptors(
    confs::DataSet,
    pod::POD;
    T = Float32, 
    path = "../../../POD/get_descriptors/train/"
)
    file_names = sort(glob("$path/localdescriptors_config*.bin"), lt=natural)
    e_des = Vector{LocalDescriptors}(undef, length(confs))
    for (j, file_desc) in enumerate(file_names)
        row_data = reinterpret(Float64, read(file_desc))
        n_atoms = convert(Int, row_data[1])
        n_desc = convert(Int, row_data[2])
        #ld = reshape(row_data[3:end], n_desc, n_atoms)
        ld = reshape(row_data[3:end], n_atoms, n_desc)
        e_des[j] = PotentialLearning.LocalDescriptors([T.(ld_i) for ld_i in eachrow(ld)])
    end
    return e_des
end


#function compute_local_descriptors(
#    ds::DataSet,
#    basis::BasisSystem;
#    pbar = true,
#    T = Float64
#)
#    iter = collect(enumerate(get_system.(ds)))
#    if pbar
#        iter = ProgressBar(iter)
#    end
#    e_des = Vector{LocalDescriptors}(undef, length(ds))
#    Threads.@threads for (j, sys) in iter
#        e_des[j] = LocalDescriptors([T.(d) for d in compute_local_descriptors(sys, basis)])
#    end
#    return e_des
#end



#function PotentialLearning.get_all_energies(
#    ds::DataSet,
#    nniap::NNIAP
#)
#    return [nniap.nn(gd[c])[1] for c in 1:length(gd)]
#end
pen_l2(x::AbstractArray) = sum(abs2, x)/2
function energy_loss(
    nn::Chain,
    iap::BasisSystem,
    ds::DataSet,
    args...
)
    nniap = NNIAP(nn, iap)
    #penalty = sum(pen_l2, Flux.params(nn))
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nniap)
    return Flux.mse(es_pred, es) #+ 1e-8 * penalty
end


