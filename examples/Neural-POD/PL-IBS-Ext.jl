struct POD
    # chemical element symbols
    species
    # periodic boundary conditions
    pbc
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


function compute_local_descriptors(confs, pod::POD, T = Float32)
    

end

function energy_loss(
    nn::Chain,
    iap::BasisSystem,
    ds::DataSet
)
    nniap = NNIAP(nn, iap)
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nniap)
    return Flux.mse(es_pred, es)
end


function PotentialLearning.learn!(
    nniap::NNIAP,
    ds::DataSet,
    opt::Flux.Optimise.AbstractOptimiser,
    epochs::Int,
    loss::Function,
    batch_size::Int,
    log_step::Int
)
    optim = Flux.setup(opt, nniap.nn)  # will store optimiser momentum, etc.
    ∇loss(nn, iap, ds, w_e, w_f) = Flux.gradient((nn) -> energy_loss(nn, iap, ds), nn)
    losses = []
    n_batches = length(ds) ÷ batch_size
    for epoch in 1:epochs
        for _ in 1:n_batches
            # Compute gradient with current parameters and update model
            batch_inds = rand(1:length(ds), batch_size)
            grads = ∇loss(nniap.nn, nniap.iap, ds[batch_inds], w_e, w_f)
            Flux.update!(optim, nniap.nn, grads[1])
        end
        # Logging
        if epoch % log_step == 0
            curr_loss = loss(nniap.nn, nniap.iap, ds, w_e, w_f)
            push!(losses, curr_loss)
            println("Epoch: $epoch, loss: $curr_loss")
        end
        GC.gc()
    end
end


