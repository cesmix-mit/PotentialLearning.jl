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


# Setup experiment #############################################################

# Experiment folder
path = "HfO2-NPOD/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)


# Define training and test configuration datasets ##############################

# Load complete configuration dataset
ds_train_path = "HfO2_mp352_ads_form_sorted.xyz"
conf_train = load_data(ds_path, uparse("eV"), uparse("Å"))

ds_train_path = "Hf_mp103_ads_form_sorted.xyz"
conf_test = load_data(ds_path, uparse("eV"), uparse("Å"))

n_train, n_test = length(conf_train), length(conf_test)


# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
dataset_selector = RandomSelector(length(conf_train); batch_size = 100)

# Subselector, option 2: DBSCANSelector
#ε, min_pts, sample_size = 0.05, 5, 3
#dataset_selector = DBSCANSelector(  conf_train,
#                                    ε,
#                                    min_pts,
#                                    sample_size)

# Subselector, option 3: kDPP + ACE (requires calculation of energy descriptors)
#ace = ACE(species           = [:Hf, :O],
#          body_order        = 2,
#          polynomial_degree = 3,
#          wL                = 1.0,
#          csp               = 1.0,
#          r0                = 1.0,
#          rcutoff           = 5.0)
#e_descr = compute_local_descriptors(conf_train,
#                                    ace,
#                                    T = Float32)
#conf_train_kDPP = DataSet(conf_train .+ e_descr)
#dataset_selector = kDPP(  conf_train_kDPP,
#                          GlobalMean(),
#                          DotProduct();
#                          batch_size = 100)

# Subsample trainig dataset
inds = PotentialLearning.get_random_subset(dataset_selector)
conf_train = conf_train[inds]
GC.gc()


# Define IAP model #############################################################

# Define POD
pod = POD(  species = [:Hf, :O],
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
@save_var path pod

# Update training dataset by adding energy descriptors
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train, pod, T = Float32)
ds_train = DataSet(conf_train .+ e_descr_train)


# Dimension reduction of energy descriptors of training dataset ######
reduce_descriptors = false
n_desc = length(e_descr_train[1][1])
if reduce_descriptors
    n_desc = n_desc / 2
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end


# Define neural network model
nn = Chain( Dense(n_desc,8,Flux.relu),
            Dense(8,1))
npod = NNIAP(nn, pod)

# Learn
println("Learning energies...")

opt = Adam(5e-4, (.9, .8))
n_epochs = 1000
log_step = 10
batch_size = 4
learn!(npod,
       ds_train,
       opt,
       n_epochs,
       energy_loss,
       batch_size,
       log_step
)

@save_var path Flux.params(npod.nn)


# Post-process output: calculate metrics, create plots, and save results #######

# Dimension reduction of energy descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, pod, T = Float32)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test)

# Get true and predicted values
e_train, e_train_pred = get_all_energies(ds_train),
                        get_all_energies(ds_train, npod)
@save_var path e_train
@save_var path e_train_pred

e_test, e_test_pred = get_all_energies(ds_test),
                      get_all_energies(ds_test, npod)
@save_var path e_test
@save_var path e_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train_pred, e_train,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
@save_dict path e_train_metrics

e_test_metrics = get_metrics(e_test_pred, e_test,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
test_metrics = merge(e_test_metrics)
@save_dict path test_metrics

# Plot and save results
e_train_plot = plot_energy(e_train_pred, e_train)
@save_fig path e_train_plot

e_test_plot = plot_energy(e_test_pred, e_test)
@save_fig path e_test_plot



