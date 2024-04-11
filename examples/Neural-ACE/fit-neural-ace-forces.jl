# Run this script:
#   $ cd examples/Neural-ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-neural-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")
include("../PCA-ACE/pca.jl")


# Setup experiment #############################################################

# Experiment folder
path = "HfO2-NeuralACE/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)

# Define training and test configuration datasets ##############################

ds_path = "../data/HfO2/"

# Load complete configuration dataset
ds_train_path = "$(ds_path)/train/HfO2_mp352_ads_form_sorted.extxyz"
conf_train = load_data(ds_train_path, uparse("eV"), uparse("Å"))

ds_test_path = "$(ds_path)/test/Hf_mp103_ads_form_sorted.extxyz"
conf_test = load_data(ds_test_path, uparse("eV"), uparse("Å"))

n_train, n_test = length(conf_train), length(conf_test)

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in conf_train]...))

# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
dataset_selector = RandomSelector(length(conf_train); batch_size = 76)

# Subselector, option 2: DBSCANSelector. Pre-cond: const. no. of atoms
#ε, min_pts, sample_size = 0.05, 5, 3
#dataset_selector = DBSCANSelector(  conf_train,
#                                    ε,
#                                    min_pts,
#                                    sample_size)

# Subselector, option 3: kDPP + ACE (requires calculation of energy descriptors)
#basis = ACE(species           = [:Hf, :O],
#          body_order        = 2,
#          polynomial_degree = 3,
#          rcutoff           = 5.0,
#          wL                = 1.0,
#          csp               = 1.0,
#          r0                = 1.0)
#e_descr = compute_local_descriptors(conf_train, basis)
#conf_train_kDPP = DataSet(conf_train .+ e_descr)
#dataset_selector = kDPP(  conf_train_kDPP,
#                          GlobalMean(),
#                          DotProduct();
#                          batch_size = 75)

# Subsample trainig dataset
inds = PotentialLearning.get_random_subset(dataset_selector)
conf_train = conf_train[inds]
GC.gc()


# Define IAP model #############################################################

# Define ACE
basis = ACE(species           = [:Hf, :O],
            body_order        = 3,
            polynomial_degree = 3,
            rcutoff           = 5.0,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0)
@save_var path basis

# Update training dataset by adding energy descriptors
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train, basis)
f_descr_train = compute_force_descriptors(conf_train, basis)
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)


# Dimension reduction of energy and force descriptors of training dataset ######
reduce_descriptors = false
n_desc = length(basis)
if reduce_descriptors
    n_desc = n_desc ÷ 2
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end

# Define neural network model
Random.seed!(100)
nns = Dict()
n_desc_per_species = n_desc #÷ length(species)
for s in species
    nns[s] = Chain( Dense(n_desc_per_species,128,σ; init = Flux.glorot_uniform(gain=-10)),
                    Dense(128,128,σ; init = Flux.glorot_uniform(gain=-10)),
                    Dense(128,1; init = Flux.glorot_uniform(gain=-10), bias = false)) |> f64
end
nnbp = NNBasisPotential(nns, basis)

## Learn #######################################################################

opt = Adam(1e-2)
n_epochs = 50
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 1.0
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
n_epochs = 500
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 1.0
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

# Save current NN parameters
ps1, _ = Flux.destructure(nnbp.nns[:Hf])
ps2, _ = Flux.destructure(nnbp.nns[:O])
@save_var path ps1
@save_var path ps2


# Post-process output: calculate metrics, create plots, and save results #######

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, basis)
f_descr_test = compute_force_descriptors(conf_test, basis)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

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

f_train, f_train_pred = get_all_forces(ds_train),
                        get_all_forces(ds_train, nnbp)
@save_var path f_train
@save_var path f_train_pred

f_test, f_test_pred = get_all_forces(ds_test),
                      get_all_forces(ds_test, nnbp)
@save_var path f_test
@save_var path f_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train,
                              e_train_pred,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
@save_dict path e_train_metrics

f_train_metrics = get_metrics(f_train,
                              f_train_pred,
                              metrics = [mae, rmse, rsq],
                              label = "f_train")
@save_dict path f_train_metrics

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

f_train_plot = plot_forces(f_train, f_train_pred)
@save_fig path f_train_plot

f_test_plot = plot_forces(f_test, f_test_pred)
@save_fig path f_test_plot

f_plot = plot_forces(f_train, f_train_pred,
                     f_test, f_test_pred)
@save_fig path f_plot


################################################################################
#using JuLIP
#using Zygote
#using AtomsBase

## Returns species index of atom j in system s
#function species_index(j, s, species)
#    as = AtomsBase.atomic_symbol(s)[j]
#    return length(species) - findall(x -> x == as, species)[1] + 1
#end

## Returns species symbol index of atom j in system s
#AtomsBase.atomic_symbol(j, s) = atomic_symbol(s)[j]

## Returns descriptor range of atom j in system s
#function desc_range(j, s, basis)
#    i = species_index(j, s, basis.species)
#    nd = length(basis) ÷ length(species)
#    return (i-1)*nd+1:i*nd
#end

#k = 1
#c = ds_train[k]
#s = get_system(c)
#a = InteratomicPotentials.convert_system_to_atoms(s)
#ns = JuLIP.neighbourlist(a, 5.0)

## Removes species zero-blocks of local energy descriptors
#led = []
#for j = 1:length(a)
#    r = desc_range(j, s, basis)
#    ledj = JuLIP.site_energy(basis.rpib, a, j)[r]
#    push!(led, ledj)
#end

## Removes species zero-blocks of local energy descriptors
#aux = Dict() # aux represents dDidrj
#for j in 1:length(a)
#    lfdj = JuLIP.site_energy_d(basis.rpib, a, j)
#    nj, _ = neigs(ns, j)
#    dr = desc_range(j, s, basis)
#    for i in nj
#        if j < i
#            dDidrj_x = [lfdj[k][i][1] for k in dr]
#            dDidrj_y = [lfdj[k][i][2] for k in dr]
#            dDidrj_z = [lfdj[k][i][3] for k in dr]
#            aux[(j, i)] = [dDidrj_x, dDidrj_y, dDidrj_z]
#        end
#    end
#end

## Compute in GPU...

#dNNdDi = Dict()
#for i = 1:length(a)
#    as = atomic_symbol(s)[i]
#    dNNdDi[i] = Zygote.gradient(x->sum(nns[as](x)), led[i])[1]
#end


## aux represents f_ij
#for j in 1:length(a)
#    nj, _ = neigs(ns, j)
#    for i in nj
#        if j < i
#            aux[(j, i)] = aux[(j, i)] .⋅ [dNNdDi[j]]
#        end
#    end
#end

## Forces
#fs = []
#for j in 1:length(a)
#    nj, _ = neigs(ns, j)
#    s = 0.0
#    for i in nj
#        if i < j
#            s = aux[(i, j)]
#        else
#            s = -aux[(j, i)]
#        end
#    end
#    push!(fs, -s)
#end


##############



#################################################################################

#using JuLIP
#using Zygote
#using AtomsBase

## Returns species index of atom j in system s
#function species_index(j, s, species)
#    as = AtomsBase.atomic_symbol(s)[j]
#    return length(species) - findall(x -> x == as, species)[1] + 1
#end

## Returns species symbol index of atom j in system s
#AtomsBase.atomic_symbol(j, s) = atomic_symbol(s)[j]

## Returns descriptor range of atom j in system s
#function desc_range(j, s, basis)
#    i = species_index(j, s, basis.species)
#    nd = length(basis) ÷ length(species)
#    return (i-1)*nd+1:i*nd
#end

#k = 1
#c = ds_train[k]
#s = get_system(c)
#a = InteratomicPotentials.convert_system_to_atoms(s)
#ns = JuLIP.neighbourlist(a, 5.0)

## Removes species zero-blocks of local energy descriptors
#led = []
#for j = 1:length(a)
#    r = desc_range(j, s, basis)
#    ledj = JuLIP.site_energy(basis.rpib, a, j)[r]
#    push!(led, ledj)
#end

## Removes species zero-blocks of local energy descriptors
#aux = Dict() # aux represents dDidrj
#for j in 1:length(a)
#    lfdj = JuLIP.site_energy_d(basis.rpib, a, j)
#    nj, _ = neigs(ns, j)
#    dr = desc_range(j, s, basis)
#    for i in nj
#        if j < i
#            dDidrj_x = [lfdj[k][i][1] for k in dr]
#            dDidrj_y = [lfdj[k][i][2] for k in dr]
#            dDidrj_z = [lfdj[k][i][3] for k in dr]
#            aux[(j, i)] = [dDidrj_x, dDidrj_y, dDidrj_z]
#        end
#    end
#end

## Compute in GPU...

#dNNdDi = Dict()
#for i = 1:length(a)
#    as = atomic_symbol(s)[i]
#    dNNdDi[i] = Zygote.gradient(x->sum(nns[as](x)), led[i])[1]
#end


## aux represents f_ij
#for j in 1:length(a)
#    nj, _ = neigs(ns, j)
#    for i in nj
#        if j < i
#            aux[(j, i)] = aux[(j, i)] .⋅ [dNNdDi[j]]
#        end
#    end
#end

## Forces
#fs = []
#for j in 1:length(a)
#    nj, _ = neigs(ns, j)
#    s = 0.0
#    for i in nj
#        if i < j
#            s = aux[(i, j)]
#        else
#            s = -aux[(j, i)]
#        end
#    end
#    push!(fs, -s)
#end


################################################################################
#using JuLIP
#using Zygote
#k = 1
#c = ds_train[k]


#function force(
#    c::Configuration,
#    nnbp::NNBasisPotential
#)
#    s = get_system(c)
#    a = InteratomicPotentials.convert_system_to_atoms(s)
#    n_atoms = length(a)
#    ns = JuLIP.neighbourlist(a, nnbp.basis.rcutoff)
#    
#    dNNdDi = Dict()
#    for i = 1:n_atoms
#        as = atomic_symbol(s)[i]
#        #r = desc_range(i, s, basis)
#        ledi = JuLIP.site_energy(basis.rpib, a, i)#[r]
#        dNNdDi[i] = Zygote.gradient(x->sum(nnbp.nns[as](x)), ledi)[1]
#    end

#    fs = []
#    for j in 1:n_atoms
#        lfdj = JuLIP.site_energy_d(basis.rpib, a, j)
#        nj, _ = neigs(ns, j)
#        fs_j = []
#        for α in 1:3
#           dDidrj_α = [lfdj[k][i][α] for k in 1:26]
#           f_j_α = sum(dNNdDi[i] ⋅ dDidrj_α for i in nj)
#           push!(fs_j, f_j_α)
#        end
#        push!(fs, fs_j)
#    end
#    
#    return fs
#end



