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

ds_path = "../../../POD/get_descriptors_HfO2/"

# Load complete configuration dataset
ds_train_path = "$(ds_path)/train/HfO2_mp352_ads_form_sorted.extxyz"
#ds_train_path = "$(ds_path)/data/train/a-HfO2-300K-NVT-6000-train.extxyz"
conf_train = load_data(ds_train_path, uparse("eV"), uparse("Å"))

ds_test_path = "$(ds_path)/test/Hf_mp103_ads_form_sorted.extxyz"
#ds_test_path = "$(ds_path)/data/test/a-HfO2-300K-NVT-6000-test.extxyz"
conf_test = load_data(ds_test_path, uparse("eV"), uparse("Å"))

n_train, n_test = length(conf_train), length(conf_test)


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
            onebody = 1,
            twobody_number_radial_basis_functions = 2,
            threebody_number_radial_basis_functions = 3,
            threebody_angular_degree = 2,
            fourbody_number_radial_basis_functions = 2,
            fourbody_angular_degree = 1,
            true4BodyDesc = 1,
            fivebody_number_radial_basis_functions = 0,
            fivebody_angular_degree = 0,
            sixbody_number_radial_basis_functions = 0,
            sixbody_angular_degree = 0,
            sevenbody_number_radial_basis_functions = 0,
            sevenbody_angular_degree = 0)
@save_var path pod

# Update training dataset by adding energy descriptors
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train,
                                          pod,
                                          T = Float32,
                                          path = "$(ds_path)/train/")
ds_train = DataSet(conf_train .+ e_descr_train)
#ds_train = ds_train[rand(1:length(ds_train), 250)]

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
reduce_descriptors = true
n_desc = length(e_descr_train[1][1])
println("n_desc: $n_desc")
if reduce_descriptors
    n_desc = convert(Int, n_desc ÷ 2)
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end
println("n_desc: $n_desc")

# Define neural network model
nn = Chain( Dense(n_desc,16,Flux.sigmoid; init = Flux.glorot_normal),
            Dense(16,16,Flux.sigmoid; init = Flux.glorot_normal),
            Dense(16,1; init = Flux.glorot_normal))
#npod = NNIAP(nn, pod)

ps, re = Flux.destructure(nn)
ps = Float16[0.012821833, -2.026746, 1.513958, -3.2032232, -3.1854382, -0.34117565, -0.69744956, 0.08911327, -0.6082431, 1.3573208, -0.743514, 0.19580585, 2.660484, -1.957519, 0.6081305, 1.0417728, 0.85300106, -1.6361356, -0.5587512, 3.4363394, -1.0074027, -1.1062223, -1.6543163, 1.9474803, 0.26468122, -2.0474277, -1.1201034, 1.1462348, -0.97960544, 0.2851511, -0.67661977, -0.30634376, 0.84001565, -0.5494795, -0.20729393, -2.509133, 2.3518355, 0.3531426, 1.4568835, 0.41239542, -1.8627973, -2.216545, 1.5986003, 0.07722439, 3.8703518, -2.8089123, 2.3465838, 0.67482144, -1.835864, 0.37516367, -1.8814753, 0.9441299, 2.6920445, -1.2839327, -5.2061863, 5.2233825, 0.54169214, -4.7260838, -1.1843061, 1.4142175, 1.6658182, 3.2592916, -4.765916, -2.7490127, 8.320877, -3.5502045, 1.1439512, 0.32122034, -2.0037746, -0.017680917, -1.3740066, -0.9210883, 0.53191304, 1.0421548, -1.2685204, 0.8088931, -6.6541924, -1.3061852, 2.6557593, -0.83121157, 1.6658894, 1.1708544, 0.28467482, -1.4379514, -1.3777555, 1.8800396, 4.1259985, -5.2421017, -1.4596455, 7.209037, 1.0483669, -2.2839987, -1.1883192, -0.9778033, 9.20458, 0.7582441, 6.57638, -1.6012, 1.1350687, 1.5836867, -2.2186918, 0.6313459, 1.4013618, -1.1062969, 2.0183454, 2.7776456, -1.3730042, -0.22135665, -6.9705715, 0.042298276, 1.5788904, -0.38392487, -2.0338426, 1.805908, 0.7603775, -3.2963915, -0.22978868, 1.15507, 3.469672, -2.083059, -0.61771595, 2.0705554, 1.6724547, -1.0294985, -0.71590954, 0.09371979, 4.6065636, 0.6468181, -1.0934885, 1.9204252, -0.26519635, -1.8734804, -0.96011066, 0.961966, 3.358246, -4.9188395, -0.28403276, 3.6818874, 0.90615016, -0.8595186, -1.1383325, 0.7454356, 4.1052785, 0.8907188, -13.679621, 3.4815464, -0.618173, 0.5083444, 3.4891398, 0.60625786, -3.0989747, 9.932815, -1.7378374, -8.31489, 7.18416, -1.3686068, 8.061485, 0.052030984, -10.26681, -1.534751, -2.5508218, 1.808368, -1.0271419, 0.82523656, 0.5381614, -0.6736275, -3.7121265, 5.978425, -0.9367335, -2.3435698, 2.0373192, 0.4221225, 4.7941265, -0.10505227, -4.9257593, 0.18123235, 0.031291813, 0.74021894, -0.615306, 0.45457864, -1.1334113, 1.1186769, -3.1207955, 3.0278552, -1.029952, -1.3887917, 1.3212813, -0.4274291, 1.7167608, -1.1491034, -2.4629564, 0.61401826, -0.4976418, -0.392616, -0.64001477, 3.304594, 0.9012786, -1.4077861, -2.8857067, 6.3451457, 0.09058148, -3.4482298, -1.0611387, 0.9904257, 2.3658872, -0.6794427, -6.4984407, -0.67188597, -0.64322287, 1.1398473, 2.1872046, -2.1307948, -0.8693061, 0.4628904, 3.7490146, -4.723873, -0.14625874, 4.295035, 0.28551728, -1.1233038, -4.19772, 0.022967976, 3.5363996, 1.7868598, 1.4995424, 1.3414962, 1.9698124, -0.2592846, -1.9572886, 0.44335467, 3.043273, -4.810355, 0.57469547, 6.475274, 0.30833805, 0.46257567, -0.7833364, 0.5284485, 7.6085024, 0.6096216, -1.4869215, -0.034939684, 0.019387862, 1.9014561, 1.2593589, -1.5424308, -4.857535, 6.193185, 0.2212009, -5.2497187, 0.17995286, 1.0114682, 0.8991982, 0.6708605, -5.206238, -1.4437469, 2.2440042, -1.0891917, 0.31065634, -2.9005415, -3.181254, 0.8772307, 1.9433128, -8.418095, 0.3603454, 2.0820541, -0.019654049, -1.079702, -5.0222316, -0.28152612, 4.371757, 1.9024533, 1.2812194, -0.0749698, -0.83921164, -2.70693, -3.0108895, 0.45869982, 0.014260341, -2.5189025, -0.28580356, -0.14731267, 0.2429361, 0.037179705, -1.2013067, -2.430363, 0.7251078, 1.6819323, -2.7916224, 1.8582815, 0.07188939, 0.57565373, 0.4681368, 0.91875595, -1.3628762, -0.13835877, -0.37977996, -3.024302, 1.6600006, -0.48400977, 0.49885386, -0.36172107, -4.097155, 0.87728214, -2.5461824, -0.604217, 0.8734175, 1.9920319, 1.9607047, -1.3417524, -1.5313766, 3.260094, 0.53625804, -1.8307127, -0.39573354, 1.8433704, 0.5951235, 0.5517048, -2.8588417, -1.2849709, -0.603441, -1.2279365, -0.68799007, 0.78411585, 1.5104803, -2.495379, -3.5960777, -1.3611947, 0.6788644, -0.75188035, -2.8168914, 1.4815693, 3.06938, 1.9484555, 0.17205358, -0.6015902, 1.6885536, 1.0815189, -0.8357163, 0.6084924, 1.3611678, 3.0882964, -0.596495, -1.2705925, -1.5219232, -0.45040283, 2.6836681, -0.9311357, 0.9631617, -2.0358841, -4.7142534, 1.155331, -1.2833781, -1.4025037, 0.5348509, 1.5533158, -1.617966, -2.6006417, -4.6190257, 3.6993306, 1.4937356, -4.84186, -0.19135213, 1.3148876, 0.14907153, 2.454545, -3.4294198, -2.4824607, -6.305426, 2.3984783, 4.1665955, 1.1715624, 1.8261213, 1.7937425, -0.30114904, 3.2437994, 0.2099472, -7.2315974, 3.8285828, 0.49452186, 5.257592, -0.67587334, -7.3430185, -3.4402294, 2.4665868, 2.7259922, 1.2263232, -0.25225747, -0.6431076, 2.4087353, 2.4728017, -0.22039771, -1.9190998, 1.5223603, 1.0015873, -1.0794547, -0.39877647, -1.2533362, -0.64717346, 1.983752, -2.779723, 0.659908, 1.3474555, -0.23721066, 0.21612032, 0.059710994, 0.17215553, 1.1511174, 2.1180434, -0.694189, 1.4039326, -0.65516764, 2.8057263, 0.3305777, -1.826019, -0.58385634, 0.02673488, 1.1897825, 2.3042786, -4.0812936, 3.3041534, -0.10518163, 0.871141, 0.8487558, 3.2882118, -0.5697742, 0.45344654, 0.99963593, -0.61679745, -0.39946768, -0.0886026, 0.9809061, -0.8407626, -0.03329136, 0.98870105, 2.3857572, 0.54401386, -0.5009137, 1.2531322, -0.7048783, 0.37975731, -1.179705, 2.6908126, 1.5882015, -0.7775796, -0.9904066, -0.8333134, 1.3170981, -1.3290447, 0.9646865, -0.70193577, -0.3528673, -0.48407853, 1.1729017, 0.9489402, -1.145901, -0.25279006, -1.1828259, 0.15399331, 0.8270041, -1.3050741, -0.45364767, -1.6141901, 1.1196792, 0.4753681, 1.1135784, 5.1902757, -1.6359401, 5.622057, -0.4102556, -0.40290084, -0.23256983, 7.3352222, 0.11092559, 2.133344, 1.5611674, -0.2099885, -0.1678893, 0.02439233, 0.5062483, -0.11488358, 0.84817874, -0.4201721, 3.0504644, -0.25024077, -0.26511613, 2.3096042, 0.28552854, -1.0696894, -0.06506645, 0.70582265, 0.78859967, -0.19579858, 0.74907964, 0.2550585, 1.4052228, -1.1307024, 2.1350977, 3.0303502, 1.9654359, 2.92317, -0.0042194333, 2.7841973, -0.9951608, 1.7923521, -0.6293327, 2.5402677, 1.7422079, -0.13058223, 0.9077157, -0.9997142, 2.4954388, -1.3734664, -1.1478343, 1.6811147, -0.41545418, 1.6711183, 0.34151003, 0.16425869, -1.4265454, 0.9910383, -0.75369036, -1.2469614, -0.013930885, -0.24091457, -0.079914674, -1.4686948, 0.35659766, -0.077499285, 1.2743287, 0.19024462, 3.310306, -0.03751576, 0.13707489, 0.101305135, 0.379629, -0.057162404, -0.2969736, 0.80327487, 1.0741898, -0.7803668, -0.29829314, 0.13063262, 0.6430608, -1.1710373, 0.87502664, 1.9114321, 1.2703102, 2.3348398, -0.23976678, -0.40436998, -0.7591313, 3.9093778, -1.4943408, 0.5555517, 1.0314262, -0.8430239, -0.73443764, -1.3156899, 0.47805652, -0.31010085, -0.75014615, 0.9466749, -1.9102361, 0.3468231, 2.040204, 0.40150014, -0.65199184, 0.49517274, -1.520695, -0.5332975, 0.22481796, -2.0989525, -1.1824349, -1.5249155, 0.6387881, -0.61743575, 0.52469224, -3.448702, 2.6692872, -2.5258524, -0.4939086, 0.9260281, -1.3039551, -1.8355811, -0.7110611, -0.7743224, -0.65211105, -0.11401743, -2.318387, -0.7564569, 0.50540656, 0.25265086, 0.60409075, 0.58130866, 1.3621677, -0.019096768, 1.0674101, 0.7285558, 0.17787361, -0.23632225, 0.026615817, 1.4479191, 0.8922705, -0.8436415, -0.22738999, -0.40428734, 1.5391448, 0.11416137, 0.83082616, 0.61291, 3.5325224, 0.3117333, 0.6794503, 0.54797727, 0.3680466, 0.17965093, -0.091458544, 0.5595151, 1.2677572, 0.28000966, -0.1410963, 0.5731121, 1.7560319, -1.0455724, -0.97207886, 2.119234, 0.8124625, 1.6049191, -0.68228966, -1.6733521, -0.89333993, -0.35038322, -0.9599267, -0.37377387, 0.22036687, 0.30836636, -1.7664697, -0.70542556, 0.21737948, -2.0479777, 0.037292402, 1.6137276, -4.9672585, 1.9717671, 0.10058717, 0.44629344, -2.0183957, 1.8498461, -1.1687487, -0.15257303, 0.37891418, -0.13838828, -0.39230528, -0.86561453, 0.8814717, 0.019715078, -0.42280692, 0.48106995, -0.008700966, 0.16045184, 3.6325037, 0.13863064, -0.33158708, -0.032189183, -0.7812133, -0.21225798, 0.6425274, -1.2571434, 0.07203368, -0.7874132, 0.5238461, -0.37407058, 0.41507214, 0.29676434, 1.6331508, 0.14332128, 0.12750728, 0.23873712, -0.14785394, 0.12181247, -0.433329, 0.37851176, 0.9983572, -0.51361024, -0.30613336, -0.28960693, 1.061737, -1.2922537, -0.9108975, -0.67710453, -2.2548325, -0.62351495, 0.41576263, -0.6160799, -1.5699652, -0.70710003, -1.36342, -0.71570903, -1.1214364, -1.2407013, -1.149629, -1.4635677, -0.75487745, -0.6919828]
nn = re(ps)
npod = NNIAP(nn, pod)

# Learn
println("Learning energies...")


opt = Adam(1f-1)
n_epochs = 100
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-7
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

opt = Adam(1f-4)
n_epochs = 300
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-7
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


#opt = BFGS()
#n_epochs = 50
#w_e, w_f = 1.0, 0.0
#for i in 1:20
#    println(i)
#    learn!(npod,
#           ds_train,
#           opt,
#           n_epochs,
#           energy_loss,
#           w_e,
#           w_f
#    )
#end

@save_var path Flux.params(npod.nn)
ps, re = Flux.destructure(npod.nn)
@save_var path ps


# Post-process output: calculate metrics, create plots, and save results #######

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test,
                                         pod,
                                         T = Float32,
                                         path = "$(ds_path)/test/")
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test)

# Dimension reduction of energy descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

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

e_plot = plot_energy(e_train_pred, e_train, 
                     e_test_pred, e_test)
@save_fig path e_plot
