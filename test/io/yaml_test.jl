using AtomsBase
using Unitful, UnitfulAtomic

energy_units = u"eV"
distance_units = u"Å"
ds, t = load_data("../examples/Sodium/data/empirical_sodium_2d.yaml", YAML(energy_units, distance_units));

@test typeof(ds) == DataSet
@test all(typeof.(get_energy.(ds)) .<: Energy )
@test all(typeof.(get_forces.(ds)) .<: Forces )