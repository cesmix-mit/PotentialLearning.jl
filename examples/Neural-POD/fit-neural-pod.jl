# Run this script:
#   $ cd examples/Neural-POD
#   $ julia --project=../ --threads=4
#   julia> include("fit-neural-pod.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")
include("PL-IBS-Ext.jl")
include("../PCA-ACE/pca.jl")


# Setup experiment #############################################################

# Experiment folder
path = "HfO2-NeuralPOD/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)


# Define training and test configuration datasets ##############################

#ds_path = "../data/HfO2_large/"
ds_path = "../data/HfO2/"

# Load complete configuration dataset
#ds_train_path = "$(ds_path)/train/a-HfO2-300K-NVT-6000-train.extxyz"
#ds_train_path = "$(ds_path)/train/HfO2_figshare_form_sorted_train.extxyz"
ds_train_path = "$(ds_path)/train/HfO2_mp352_ads_form_sorted.extxyz"
conf_train = load_data(ds_train_path, uparse("eV"), uparse("Å"))

#ds_test_path = "$(ds_path)/test/a-HfO2-300K-NVT-6000-test.extxyz"
#ds_test_path = "$(ds_path)/test/HfO2_figshare_form_sorted_test.extxyz"
ds_test_path = "$(ds_path)/test/Hf_mp103_ads_form_sorted.extxyz"
conf_test = load_data(ds_test_path, uparse("eV"), uparse("Å"))

n_train, n_test = length(conf_train), length(conf_test)

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in conf_train]...))

# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
#dataset_selector = RandomSelector(length(conf_train); batch_size = 100)

# Subselector, option 2: DBSCANSelector
#ε, min_pts, sample_size = 0.05, 5, 3
#dataset_selector = DBSCANSelector(  conf_train,
#                                    ε,
#                                    min_pts,
#                                    sample_size)

# Subselector, option 3: kDPP + ACE (requires calculation of energy descriptors)
#basis = POD(  species = [:Hf, :O],
#            rin = 1.0,
#            rcut = 7.5,
#            bessel_polynomial_degree = 4,
#            inverse_polynomial_degree = 10,
#            onebody = 1,
#            twobody_number_radial_basis_functions = 2,
#            threebody_number_radial_basis_functions = 2,
#            threebody_angular_degree = 2,
#            fourbody_number_radial_basis_functions = 0,
#            fourbody_angular_degree = 0,
#            true4BodyDesc = 0,
#            fivebody_number_radial_basis_functions = 0,
#            fivebody_angular_degree = 0,
#            sixbody_number_radial_basis_functions = 0,
#            sixbody_angular_degree = 0,
#            sevenbody_number_radial_basis_functions = 0,
#            sevenbody_angular_degree = 0)
#path = "../../../POD/get_descriptors/train/"
#e_descr = compute_local_descriptors(conf_train, basis, T = Float32, path = path)
#conf_train_kDPP = DataSet(conf_train .+ e_descr)
#dataset_selector = kDPP(  conf_train_kDPP,
#                          GlobalMean(),
#                          DotProduct();
#                          batch_size = 75)

# Subsample trainig dataset
#inds = PotentialLearning.get_random_subset(dataset_selector)
#conf_train = conf_train[inds]
#GC.gc()


# Define IAP model #############################################################

# Define POD
basis = POD(  species                                 = "Hf O",
            rin                                     = 1.0,
            rcut                                    = 5.0,
            bessel_polynomial_degree                = 4,
            inverse_polynomial_degree               = 10,
            onebody                                 = 1,
            twobody_number_radial_basis_functions   = 4,
            threebody_number_radial_basis_functions = 2,
            threebody_angular_degree                = 1,
            fourbody_number_radial_basis_functions  = 0,
            fourbody_angular_degree                 = 0,
            true4BodyDesc                           = 0,
            fivebody_number_radial_basis_functions  = 0,
            fivebody_angular_degree                 = 0,
            sixbody_number_radial_basis_functions   = 0,
            sixbody_angular_degree                  = 0,
            sevenbody_number_radial_basis_functions = 0,
            sevenbody_angular_degree                = 0)
@save_var path basis

# Update training dataset by adding energy descriptors
println("Computing energy descriptors of training dataset...")
lammps_path = "../../../../POD/lammps/build/lmp"
compute_local_descriptors(conf_train,
                          basis,
                          T = Float32,
                          ds_path = ds_path,
                          lammps_path = lammps_path)
e_descr_train = load_local_descriptors(conf_train,
                                       basis,
                                       T = Float32,
                                       ds_path = "$ds_path/train")
ds_train = DataSet(conf_train .+ e_descr_train)

#ds_train = ds_train[rand(1:n_train, 200)]

n_desc = length(e_descr_train[1][1])

# Define neural network model
nns = Dict()
for s in species
    nns[s] = Chain( Dense(n_desc,128,σ; init = Flux.glorot_uniform(gain=-1.43)),
                    Dense(128,128,σ; init = Flux.glorot_uniform(gain=-1.43)),
                    Dense(128,1; init = Flux.glorot_uniform(gain=-1.43), bias = false))
end
nnbp = NNBasisPotential(nns, basis)

# Learn
println("Learning energies...")

opt = Adam(1f-2)
n_epochs = 50
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-8
learn!(nnbp,
       ds_train,
       opt,
       n_epochs,
       energy_loss,
       w_e,
       w_f,
       reg,
       batch_size,
       log_step
)

opt = Adam(1e-4)
n_epochs = 100
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-4
learn!(nnbp,
       ds_train,
       opt,
       n_epochs,
       energy_loss,
       w_e,
       w_f,
       reg,
       batch_size,
       log_step
)

# Save current NN parameters
ps1, _ = Flux.destructure(nnbp.nns[:Hf])
ps2, _ = Flux.destructure(nnbp.nns[:O])
@save_var path ps1
@save_var path ps2


# Post-process output: calculate metrics, create plots, and save results #######

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = load_local_descriptors(conf_test,
                                      basis,
                                      T = Float32,
                                      ds_path = "$ds_path/test")
ds_test = DataSet(conf_test .+ e_descr_test)
GC.gc()


# Get true and predicted values
n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                        get_all_energies(ds_train, nnbp) ./ n_atoms_train
@save_var path e_train
@save_var path e_train_pred

e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                      get_all_energies(ds_test, nnbp) ./ n_atoms_test
@save_var path e_test
@save_var path e_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train,
                              e_train_pred,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
@save_dict path e_train_metrics

e_test_metrics = get_metrics(e_test,
                             e_test_pred,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
@save_dict path e_test_metrics

# Plot and save results
e_train_plot = plot_energy(e_train, e_train_pred)
@save_fig path e_train_plot

e_test_plot = plot_energy(e_test, e_test_pred)
@save_fig path e_test_plot

e_plot = plot_energy(e_train, e_train_pred,
                     e_test, e_test_pred)
@save_fig path e_plot

