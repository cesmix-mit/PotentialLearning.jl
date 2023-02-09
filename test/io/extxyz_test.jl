using AtomsBase
using Unitful, UnitfulAtomic

energy_units = u"eV"
distance_units = u"Å"
ds = load_data("../examples/Si-3Body-LAMMPS/data.xyz", ExtXYZ(energy_units, distance_units));

@test length(ds) == 201
@test typeof(ds) == DataSet
