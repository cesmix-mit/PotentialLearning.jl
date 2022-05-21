using AtomsBase


systems, energies, forces, stresses = load_data("../examples/Si-3Body-LAMMPS/data.xyz", ExtXYZ());

@test length(systems) == 201
@test isa(systems[1], AtomsBase.AbstractSystem)
@test isa(energies, Vector) 