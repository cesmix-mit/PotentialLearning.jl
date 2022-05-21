using AtomsBase


system = load_data("../examples/Na-LAMMPS/starting_configuration.na", LAMMPS([:Na], [Periodic(), Periodic(), Periodic()]));

@test length(system) == 91
@test isa(system, AtomsBase.AbstractSystem)
