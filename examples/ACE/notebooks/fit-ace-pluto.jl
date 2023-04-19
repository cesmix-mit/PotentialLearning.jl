### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ b2349d7c-d59e-11ed-30d7-d5893a94a284
begin
	import Pkg
	Pkg.activate(".")
	using AtomsBase
	using Unitful, UnitfulAtomic
	using InteratomicPotentials 
	using InteratomicBasisPotentials
	using PotentialLearning
	using LinearAlgebra
	using Random
	include("utils/utils.jl")
	nothing
end

# ╔═╡ ae43b4b2-8745-4637-abb1-f55cfe9064fb
md"""# Fit ACE"""

# ╔═╡ 17503039-0e41-426a-a8f1-b24410399a46
md"""#### Load and split dataset"""

# ╔═╡ b92dd72b-e94f-40ce-a869-d72df58ae329
md"""Load dataset"""

# ╔═╡ 184d96cb-7850-4bf3-b571-c0ce7e0fd070
begin
	ds_path = "data/a-Hfo2-300K-NVT-6000.extxyz"
	energy_units, distance_units = u"eV", u"Å"
	confs = load_data(ds_path, energy_units, distance_units)
end

# ╔═╡ 55b0d727-7037-471c-9d47-3ef0f0d3d171
md"""Inspect dataset"""

# ╔═╡ b85ffd1e-bf7a-4789-8ddd-826ccb8ee77b
s = get_system(confs[1])

# ╔═╡ d026b0d5-9054-461a-87a5-a987900267c4
get_positions(confs[1])

# ╔═╡ 910864e4-15f2-4f2c-abba-c4cdb422e039
get_energy(confs[1])

# ╔═╡ 58ba9edd-b248-4b30-a650-401fe076eb73
md"""Split dataset"""

# ╔═╡ 53af2ca6-7e30-4e88-b114-be952b4e9815
begin
	n_train, n_test = 200, 200
	conf_train, conf_test = split(confs, n_train, n_test)
end

# ╔═╡ 7cb3a42b-7f33-446c-83ad-805cf3a5fc81
md"""#### Create ACE basis"""

# ╔═╡ ca22b6be-f5da-4eb5-868f-d2b488f6e642
ace = ACE(species = [:Hf, :O], body_order = 3, polynomial_degree = 4,
          wL = 1.0, csp = 1.0, r0 = 1.0, rcutoff = 5.0)

# ╔═╡ a12c09ff-8b1a-4d7b-9a0a-053d620ad928
md"""#### Calculate energy and force descriptors from the training data set and update the training data set"""

# ╔═╡ a3eabf8c-5732-4b85-8d73-a3dc1335733b
e_descr_train = compute_local_descriptors(conf_train, ace)

# ╔═╡ 7d21148b-58a7-41dc-9581-17b4a0c4700a
get_values(e_descr_train[1])

# ╔═╡ f9d9edc2-df7f-4771-af5f-5972b25002cc
f_descr_train = compute_force_descriptors(conf_train, ace)

# ╔═╡ 7421b5cc-f740-477d-8e7c-e409b2223616
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)

# ╔═╡ 4ff44330-edb4-4956-b619-bf8428a05b7e
md"""#### Learn energies and forces"""

# ╔═╡ 0f0eaeb8-12d8-4424-9302-e4685944f82a
begin
	lb = LBasisPotential(ace)
	learn!(lb, ds_train; w_e = 1.0, w_f = 1.0)
end

# ╔═╡ 4c310c3d-c091-4a5a-aed7-d72aa0e4cad2
md"""#### Calculate energy and force descriptors from the test data set and update the test data set"""

# ╔═╡ 9680ffc4-b919-4ab9-bcd3-7f3c037eac5e
e_descr_test = compute_local_descriptors(conf_test, ace)

# ╔═╡ dee5c16f-b717-4e82-8f1b-bd7acc697682
f_descr_test = compute_force_descriptors(conf_test, ace)

# ╔═╡ a4ad8c81-da5b-4eca-8778-a67245de2b54
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# ╔═╡ 5373543a-f3ed-4813-8de9-b51d47d5c4c5
md"""#### Get true and predicted values"""

# ╔═╡ 239c4650-2150-445a-9b4c-a11c0b6a01d2
begin
	e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
	e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
	e_train_pred, f_train_pred = get_all_energies(ds_train, lb), get_all_forces(ds_train, lb)
	e_test_pred, f_test_pred = get_all_energies(ds_test, lb), get_all_forces(ds_test, lb)
end

# ╔═╡ 07f13bef-bd51-4075-96a0-588b3b9e5ca7
md"""#### Compute metrics"""

# ╔═╡ bd0aa833-add9-4ee5-9e72-35f7a6f9220f
metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       0.0, 0.0, 0.0)

# ╔═╡ 83439593-056b-48be-a403-202e1192fd87
md"""#### Plot and save results"""

# ╔═╡ 8ccfc627-c687-45fd-8840-4ff158b1ad72
plot_energy(e_test_pred, e_test)

# ╔═╡ 59c6ebe6-8051-4fcd-b5de-1145e489119b
plot_forces(f_test_pred, f_test)

# ╔═╡ 714c8dfc-245d-45de-842d-7fb2f8fefa03
plot_cos(f_test_pred, f_test)

# ╔═╡ Cell order:
# ╟─ae43b4b2-8745-4637-abb1-f55cfe9064fb
# ╠═b2349d7c-d59e-11ed-30d7-d5893a94a284
# ╟─17503039-0e41-426a-a8f1-b24410399a46
# ╟─b92dd72b-e94f-40ce-a869-d72df58ae329
# ╠═184d96cb-7850-4bf3-b571-c0ce7e0fd070
# ╟─55b0d727-7037-471c-9d47-3ef0f0d3d171
# ╠═b85ffd1e-bf7a-4789-8ddd-826ccb8ee77b
# ╠═d026b0d5-9054-461a-87a5-a987900267c4
# ╠═910864e4-15f2-4f2c-abba-c4cdb422e039
# ╟─58ba9edd-b248-4b30-a650-401fe076eb73
# ╠═53af2ca6-7e30-4e88-b114-be952b4e9815
# ╟─7cb3a42b-7f33-446c-83ad-805cf3a5fc81
# ╠═ca22b6be-f5da-4eb5-868f-d2b488f6e642
# ╟─a12c09ff-8b1a-4d7b-9a0a-053d620ad928
# ╠═a3eabf8c-5732-4b85-8d73-a3dc1335733b
# ╠═7d21148b-58a7-41dc-9581-17b4a0c4700a
# ╠═f9d9edc2-df7f-4771-af5f-5972b25002cc
# ╠═7421b5cc-f740-477d-8e7c-e409b2223616
# ╟─4ff44330-edb4-4956-b619-bf8428a05b7e
# ╠═0f0eaeb8-12d8-4424-9302-e4685944f82a
# ╟─4c310c3d-c091-4a5a-aed7-d72aa0e4cad2
# ╠═9680ffc4-b919-4ab9-bcd3-7f3c037eac5e
# ╠═dee5c16f-b717-4e82-8f1b-bd7acc697682
# ╠═a4ad8c81-da5b-4eca-8778-a67245de2b54
# ╟─5373543a-f3ed-4813-8de9-b51d47d5c4c5
# ╠═239c4650-2150-445a-9b4c-a11c0b6a01d2
# ╟─07f13bef-bd51-4075-96a0-588b3b9e5ca7
# ╠═bd0aa833-add9-4ee5-9e72-35f7a6f9220f
# ╟─83439593-056b-48be-a403-202e1192fd87
# ╠═8ccfc627-c687-45fd-8840-4ff158b1ad72
# ╠═59c6ebe6-8051-4fcd-b5de-1145e489119b
# ╠═714c8dfc-245d-45de-842d-7fb2f8fefa03
