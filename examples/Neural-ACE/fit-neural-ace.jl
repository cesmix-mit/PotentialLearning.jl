# Run this script:
#   $ cd examples/Neural-ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-neural-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")


# Setup experiment #############################################################

# Experiment folder
path = "a-HfO2-NACE/" #"Ta-NACE/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)


# Define training and test configuration datasets ##############################

# Load complete configuration dataset
ds_path = "../data/a-HfO2/a-Hfo2-300K-NVT-6000.extxyz" # "../data/Ta.extxyz"
ds = load_data(ds_path, uparse("eV"), uparse("Å"))

# Split configuration dataset into training and test
n_train, n_test = 150, 400 #300, 63
conf_train, conf_test = split(ds, n_train, n_test, f = 0.827)

# Free memory
ds = nothing
GC.gc()


# Define dataset generator #####################################################
dataset_generator = Nothing


# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
dataset_selector = RandomSelector(length(conf_train); batch_size = 150)

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

# Define ACE
ace = ACE(species           = [:Hf, :O], #[:Ta],
          body_order        = 3,
          polynomial_degree = 3,
          wL                = 1.0,
          csp               = 1.0,
          r0                = 1.0,
          rcutoff           = 5.0)
@save_var path ace

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train,
                                          ace,
                                          T = Float32)
println("Computing force descriptors of training dataset...")
f_descr_train = compute_force_descriptors(conf_train,
                                          ace,
                                          T = Float32)
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)


# Dimension reduction of energy and force descriptors of training dataset ######
reduce_descriptors = false
n_desc = length(e_descr_train[1][1])
if reduce_descriptors
    n_desc = n_desc / 2
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end

######### Site energy d
using ACE1, JuLIP

function compute_local_descriptors_d(A::AbstractSystem, ace::ACE)
    return [ACE1.site_energy_d(ace.rpib, InteratomicBasisPotentials.convert_system_to_atoms(A), i) for i = 1:length(A)]
end

#c = ds_train[1]
#d = compute_local_descriptors_d(get_system(c), ace)

#atom_j, coor, i = 10, 3, 1
#[[d[i][k][atom_j][coor] for k in 1:26] for i in 1:96]

dbdr_c = Dict()
for c_ind in 1:length(ds_train)
    c = ds_train[c_ind]
    d = compute_local_descriptors_d(get_system(c), ace)
    dbdr = Array{Any}(undef, 3, 96)
    for coor in 1:3
        for atom_j in 1:96
            dbdr[coor, atom_j] = reduce(hcat, [[d[i][k][atom_j][coor]
                                               for k in 1:n_desc] for i in 1:96])
            dbdr_c[c] = dbdr
        end
    end
end

# Define neural network model
#nn = Chain( Dense(n_desc,64,Flux.sigmoid; init = Flux.glorot_uniform),
#            Dense(64,64,Flux.sigmoid; init = Flux.glorot_uniform),
#            Dense(64,1; init = Flux.glorot_uniform))
#nn = Chain( Dense(n_desc,60,Flux.sigmoid;init=Flux.glorot_normal),
#            Dense(60,60,Flux.sigmoid;init=Flux.glorot_normal),
#            Dense(60,1;init=Flux.glorot_normal))
#nn = Chain( Dense(n_desc,8,Flux.sigmoid; init=Flux.glorot_normal),
#            Dense(8,1; init=Flux.glorot_normal))
nn = Chain( Dense(n_desc,8,Flux.relu),
            Dense(8,1))
nace = NNIAP(nn, ace)

# Learn
println("Learning energies and forces...")

#opt = BFGS()
#n_epochs = 50
#w_e, w_f = 1.0, 1.0 #0.65 # 0.01, 1.0
#learn!(nace,
#       ds_train,
#       opt,
#       n_epochs,
#       loss,
#       w_e,
#       w_f
#)

#(W2 sigmoid(W1 X + b1) + b2)

#opt = Adam(.1, (.9, .8)) # opt = Adam(0.001) #Adam(5e-4, (.9, .8)) # Adam(5e-4)
#opt = Adam(5e-4, (.9, .8)) # 
opt = Adam(5e-4)
n_epochs = 3000
log_step = 10
batch_size = 4
w_e, w_f =  1.0, 1.0 #0.01, 1.0 # 1.0, 0.65
learn!(nace,
       ds_train,
       opt,
       n_epochs,
       loss,
       w_e,
       w_f,
       batch_size,
       log_step
)


@save_var path Flux.params(nace.nn)


# Post-process output: calculate metrics, create plots, and save results #######

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test,
                                         ace,
                                         T = Float32)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test,
                                         ace,
                                         T = Float32)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Get true and predicted values
e_train, e_train_pred = get_all_energies(ds_train),
                        get_all_energies(ds_train, nace)
f_train, f_train_pred = get_all_forces(ds_train),
                        get_all_forces(ds_train, nace)
@save_var path e_train
@save_var path e_train_pred
@save_var path f_train
@save_var path f_train_pred

e_test, e_test_pred = get_all_energies(ds_test),
                      get_all_energies(ds_test, nace)
f_test, f_test_pred = get_all_forces(ds_test),
                      get_all_forces(ds_test, nace)
@save_var path e_test
@save_var path e_test_pred
@save_var path f_test
@save_var path f_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train_pred, e_train,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
f_train_metrics = get_metrics(f_train_pred, f_train,
                              metrics = [mae, rmse, rsq, mean_cos],
                              label = "f_train")
train_metrics = merge(e_train_metrics, f_train_metrics)
@save_dict path train_metrics

e_test_metrics = get_metrics(e_test_pred, e_test,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
f_test_metrics = get_metrics(f_test_pred, f_test,
                             metrics = [mae, rmse, rsq, mean_cos],
                             label = "f_test")
test_metrics = merge(e_test_metrics, f_test_metrics)
@save_dict path test_metrics

# Plot and save results
e_train_plot = plot_energy(e_train_pred, e_train)
f_train_plot = plot_forces(f_train_pred, f_train)
f_train_cos  = plot_cos(f_train_pred, f_train)
@save_fig path e_train_plot
@save_fig path f_train_plot
@save_fig path f_train_cos

e_test_plot = plot_energy(e_test_pred, e_test)
f_test_plot = plot_forces(f_test_pred, f_test)
f_test_cos  = plot_cos(f_test_pred, f_test)
@save_fig path e_test_plot
@save_fig path f_test_plot
@save_fig path f_test_cos

