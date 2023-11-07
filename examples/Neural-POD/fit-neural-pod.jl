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

ds_path = "../../../data/HfO2/"
#ds_path = "../../../data/HfO2_large/"

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
#pod = POD(  species = [:Hf, :O],
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
#e_descr = compute_local_descriptors(conf_train, pod, T = Float32, path = path)
#conf_train_kDPP = DataSet(conf_train .+ e_descr)
#dataset_selector = kDPP(  conf_train_kDPP,
#                          GlobalMean(),
#                          DotProduct();
#                          batch_size = 100)

# Subsample trainig dataset
#inds = PotentialLearning.get_random_subset(dataset_selector)
#conf_train = conf_train[inds]
#GC.gc()


# Define IAP model #############################################################

# Define POD
pod = POD(  species                                 = "Hf O",
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
@save_var path pod

# Update training dataset by adding energy descriptors
println("Computing energy descriptors of training dataset...")
lammps_path = "../../POD/lammps/build/lmp"
compute_local_descriptors(conf_train,
                          pod,
                          T = Float32,
                          ds_path = ds_path,
                          lammps_path = lammps_path)
e_descr_train = load_local_descriptors(conf_train,
                                       pod,
                                       T = Float32,
                                       ds_path = "$ds_path/train")
ds_train = DataSet(conf_train .+ e_descr_train)

n_desc = length(e_descr_train[1][1])

# Define neural network model
nns = Dict()
for s in species
    nns[s] = Chain( Dense(n_desc,128,σ; init = Flux.glorot_uniform(gain=-1.43)),
                    Dense(128,128,σ; init = Flux.glorot_uniform(gain=-1.43)),
                    Dense(128,1; init = Flux.glorot_uniform(gain=-1.43)))
end
npod = NNIAP(nns, pod)

# Learn
println("Learning energies...")

opt = Adam(1f-2)
n_epochs = 50
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-8
learn!(npod,
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

opt = Adam(1e-4, (.9, .8))
n_epochs = 100
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-8
learn!(npod,
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
ps1, _ = Flux.destructure(npod.nn[:Hf])
ps2, _ = Flux.destructure(npod.nn[:O])
@save_var path ps1
@save_var path ps2


# Post-process output: calculate metrics, create plots, and save results #######

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = load_local_descriptors(conf_test,
                                      pod,
                                      T = Float32,
                                      ds_path = "$ds_path/test")
ds_test = DataSet(conf_test .+ e_descr_test)
GC.gc()


# Get true and predicted values
n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train, e_train_pred = get_all_energies(ds_train),
                        get_all_energies(ds_train, npod)
@save_var path e_train
@save_var path e_train_pred

e_test, e_test_pred = get_all_energies(ds_test),
                      get_all_energies(ds_test, npod)
@save_var path e_test
@save_var path e_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train_pred ./ n_atoms_train,
                              e_train ./ n_atoms_train,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
@save_dict path e_train_metrics

e_test_metrics = get_metrics(e_test_pred ./ n_atoms_test,
                             e_test ./ n_atoms_test,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
@save_dict path e_test_metrics

# Plot and save results
e_train_plot = plot_energy(e_train_pred, e_train)
@save_fig path e_train_plot

e_test_plot = plot_energy(e_test_pred, e_test)
@save_fig path e_test_plot

#e_plot = plot_energy(e_train_pred[1:50:5000], e_train[1:50:5000], 
#                     e_test_pred[1:10:1000], e_test[1:10:1000])
e_plot = plot_energy(e_train_pred, e_train,
                     e_test_pred, e_test)
@save_fig path e_plot




########### Linear POD

## Load global energy descriptors of training dataset
#file_names = sort(glob("$(ds_path)/train/globaldescriptors_config*.bin"), lt=natural)
#e_des = []
#for file_desc in file_names
#    raw_data = reinterpret(Float64, read(file_desc))
#    n_atom = Int64(raw_data[1])
#    n_desc = Int64(raw_data[2])
#    gd = raw_data[3:2+n_desc]
#    push!(e_des, gd)
#end
## Solve linear system and compute MAE
#D_train = hcat(e_des...)'
#e_train = get_all_energies(ds_train)
#X = D_train \ e_train
##X = [3.48399741, -7.03350751, -1.79394608, -0.93452429, 0.27688678, 0.01539386, 0.10474397, -0.30322716, 1.07867254, -0.29984347, -0.04422705, -2.29990985, -44.37757193, -1.18013986, -0.45650380, -0.42703254, -0.11620421, -0.09195364, -1.30636940, 0.80070304, -0.13061357, -6.31673190, -1.99364747, 0.07016861, 3.63996064, 0.89533892, -0.25031364, -1.15630429, -0.63267034, 0.40492066, -2.16708676, 0.82338380, -0.06937984, 0.12862848, 0.01934535, -0.03835228, 0.44050895, 0.63986887, -0.62861567, -17.33978357, 1.31278972, -0.12427187, 4.71461571, -0.58520052, 0.42564980, -0.23821002, -0.62710306, 0.18275406, 16.90529974, -5.18348382, 0.00445705, -3.50024956, 0.41487702, -0.32454629, 0.45598998, 0.51386513, 0.13951806, -0.65328817, 0.50033585, 0.00497733, -0.32944733, 0.05827797, 0.02158471, -0.01736381, -0.10467845, 1.67835002, -43.61105677, 0.01692523, 5.53876999, 0.03547309, 7.80999969, 0.07059559, -0.44040350, -0.40925159, 2.19738778, 0.01968470, -0.28207942, 0.17949702, -0.81133262, -0.02340036, 0.06552525, 0.02980010, -2.74351799, -0.01136194, -0.41782946, 0.18610338, 6.36324292, 0.01827037, -0.28958442, -0.05450477, -7.70357714, -0.00484099, 0.51651897, -0.05596215, 0.37432485, 0.00477489, 0.05163665]
#e_train_mae = Flux.mae(D_train * X ./ n_atoms_train, e_train ./ n_atoms_train)
#println("e_train_mae: $e_train_mae")

## Load global energy descriptors of test dataset
#file_names = sort(glob("$(ds_path)/test/globaldescriptors_config*.bin"), lt=natural)
#e_des = []
#for file_desc in file_names
#    raw_data = reinterpret(Float64, read(file_desc))
#    n_atom = Int64(raw_data[1])
#    n_desc = Int64(raw_data[2])
#    gd = raw_data[3:2+n_desc]
#    push!(e_des, gd)
#end
## Compare previous results with test dataset
#D_test = hcat(e_des...)'
#e_test = get_all_energies(ds_test)
#e_test_mae = Flux.mae(D_test * X ./ n_atoms_test, e_test ./ n_atoms_test)
#println("e_test_mae: $e_test_mae")

## Plot
#plot_energy(D_train * X, e_train, D_test * X, e_test)

