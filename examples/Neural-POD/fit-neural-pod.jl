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
path = "HfO2-NPOD/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)


# Define training and test configuration datasets ##############################

# Load complete configuration dataset
ds_train_path = "data/HfO2_mp352_ads_form_sorted.extxyz"
conf_train = load_data(ds_train_path, uparse("eV"), uparse("Å"))

ds_test_path = "data/Hf_mp103_ads_form_sorted.extxyz"
conf_test = load_data(ds_test_path, uparse("eV"), uparse("Å"))

n_train, n_test = length(conf_train), length(conf_test)


# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
# dataset_selector = RandomSelector(length(conf_train); batch_size = 100)

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
#            onebody = 2,
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

## Subsample trainig dataset
#inds = PotentialLearning.get_random_subset(dataset_selector)
#conf_train = conf_train[inds]
#GC.gc()


# Define IAP model #############################################################

# Define POD
pod = POD(  species = [:Hf, :O],
            rin = 1.0,
            rcut = 7.5,
            bessel_polynomial_degree = 4,
            inverse_polynomial_degree = 10,
            onebody = 2,
            twobody_number_radial_basis_functions = 2,
            threebody_number_radial_basis_functions = 2,
            threebody_angular_degree = 2,
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
descr_path = "../../../POD/get_descriptors/train/"
e_descr_train = compute_local_descriptors(conf_train, pod, T = Float32, path = descr_path)
ds_train = DataSet(conf_train .+ e_descr_train)

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
#n_desc = length(gd[1])


# Dimension reduction of energy descriptors of training dataset ######
reduce_descriptors = false
n_desc = length(e_descr_train[1][1])
if reduce_descriptors
    n_desc = n_desc ÷ 2
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end


# Define neural network model
nn = Chain( Dense(n_desc,32,Flux.tanh_fast; init = Flux.glorot_normal),
            Dense(32,32,Flux.tanh_fast; init = Flux.glorot_normal),
            Dense(32,1; init = Flux.glorot_normal))
npod = NNIAP(nn, pod)

ps, re = Flux.destructure(nn)
ps = Float32[0.16015099, 0.033158828, 0.37109163, -0.10890404, 0.30580732, -0.16939954, 0.01745141, 0.22640733, 0.11346262, -0.3619112, 0.022905013, 0.016541764, 0.07273872, -0.10549682, -0.10199306, 0.17143102, 0.28734162, -0.18761113, 0.15979838, -0.3352327, -0.026833903, 0.29160684, -0.22943445, 0.34329984, 0.40671653, -0.37174094, -0.38599378, 0.44523606, 0.2592528, -0.2382685, 0.39838815, -0.11071494, 0.17390612, 0.37082618, -0.43802848, -0.4023025, -0.48864242, 0.16717269, 0.30507907, 0.36084625, -0.44483423, 0.2831938, 0.17297249, -0.17960554, 0.28801745, -0.22551166, 0.34111196, 0.13309461, -0.109212145, 0.40053007, -0.04556034, -0.34694383, -0.36323196, 0.20577627, -0.1475614, -0.26531297, -0.17225423, 0.044197276, 0.08621969, -0.105466045, -0.29099983, 0.31725758, 0.3027398, 0.26546746, -0.3821627, -0.23487706, 0.103157684, 0.0932141, 0.2167541, 0.3272981, -0.11879683, 0.17789719, -0.31525454, 0.0027613216, -0.13729386, -0.22362468, -0.49114123, 0.050551202, 0.11269456, -0.40485525, -0.0005971552, -0.058692455, 0.18833442, -0.32317492, -0.06659354, -0.02805326, -0.08840453, 0.09135173, 0.30985132, -0.18567632, -0.2708247, 0.3605935, -0.41191322, 0.18515335, 0.3103248, 0.010269687, 0.01050629, 0.34139964, 0.3344249, 0.16910353, -0.37002227, 0.010623839, -0.32343522, 0.2968478, -0.34272614, -0.022638233, -0.26957977, -0.3010477, -0.16629544, -0.079699084, 0.023561515, -0.41153657, -0.18462841, 0.026638694, 0.042388942, -0.042200387, -0.25579816, 0.06070482, 0.007152858, 0.03528882, -0.15126291, -0.087036856, 0.37019157, -0.014743509, -0.057367995, -0.22371341, 0.46672356, -0.108151406, -0.20234579, -0.12138925, 0.28114173, -0.444108, -0.2284079, -0.07923253, -0.01969011, 0.047939952, -0.43740937, 0.091353334, -0.4496412, 0.13239014, 0.35463908, 0.32364857, 0.27656087, 0.28496614, -0.07782867, 0.3685957, 0.15287036, 0.29790533, -0.051223, 0.24355714, 0.49899682, 0.095605426, 0.19417623, -0.20705725, 0.047458433, 0.15096079, 0.16648497, -0.087318696, 0.022845013, 0.15884268, 0.10961744, 0.40177202, 0.013228595, -0.13041934, 0.18768501, 0.28935707, 0.3604813, 0.3447494, 0.10851231, -0.028270053, -0.43346834, -0.2367126, -0.15806356, 0.46609905, 0.4083501, 0.22188488, 0.175657, 0.11877816, -0.023380734, -0.2650006, 0.20380946, -0.17941305, -0.30819535, -0.27364713, -0.020222226, 0.061383776, -0.01893904, 0.06318603, -0.055978213, 0.060771503, 0.06759761, -0.057151157, 0.49377, -0.38752568, 0.64362216, -0.25375435, 0.3938452, -0.48203218, -0.16366911, 0.7057886, -0.05828085]
nn = re(ps)
npod = NNIAP(nn, pod)

# Learn
println("Learning energies...")


opt = Adam(1f-3)
n_epochs = 100
log_step = 10
batch_size = 32
w_e, w_f = 1.0, 0.0
learn!(npod,
       ds_train,
       opt,
       n_epochs,
       energy_loss,
       w_e,
       w_f,
       batch_size,
       log_step
)

opt = Adam(1f-5, (.9, .8))
n_epochs = 2000
log_step = 10
batch_size = 32
w_e, w_f = 1.0, 0.0
learn!(npod,
       ds_train,
       opt,
       n_epochs,
       energy_loss,
       w_e,
       w_f,
       batch_size,
       log_step
)


#opt = BFGS()
#n_epochs = 50
#w_e, w_f = 1.0, 0.0
#learn!(npod,
#       ds_train,
#       opt,
#       n_epochs,
#       energy_loss,
#       w_e,
#       w_f
#)

@save_var path Flux.params(npod.nn)
ps, re = Flux.destructure(npod.nn)
@save_var path ps


# Post-process output: calculate metrics, create plots, and save results #######

# Dimension reduction of energy descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
descr_path = "../../../POD/get_descriptors/test/"
e_descr_test = compute_local_descriptors(conf_test, pod, T = Float32, path = descr_path)
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



