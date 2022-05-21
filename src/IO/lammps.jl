"""
    struct LAMMPS <: IO
        elements :: Vector{Symbol}
        boundary_conditions :: Vector
    end
"""
struct LAMMPS <: IO
    elements :: Vector{Symbol}
    boundary_conditions :: Vector
end

"""
    load_data(file::string, lammps::LAMMPS)

    Load configuration from a lammps data file into a Flexible System.
    Need LAMMPS struct to know the symbols of the elements, and the boundary_conditions.
"""

function load_data(file, lammps :: LAMMPS )
    num_atoms = 0
    num_atom_types = length(lammps.elements)
    bias = []
    box = [[1.0, 0.0, 0.0]*1u"Å", 
            [0.0, 1.0, 0.0]*1u"Å",
            [0.0, 0.0, 1.0]*1u"Å"]
    system = []
    elements = []
    positions = SVector{3}[]
    velocities = SVector{3}[]
    open(file, "r") do io
        while !eof(io)
            line = split(readline(io))
            if "atoms" in line
                num_atoms = parse(Int, line[1])
            end
            if ("atom" in line) && ("types" in line)
                num_atom_types = parse(Int, line[1])
                if num_atom_types != length(lammps.elements)
                    error("num_atom_types in LAMMPS file does not match number of elements in LAMMPS struct!")
                end
            end
            if "xlo" in line
                x = parse.(Float64, line[1:2])
                x_bias = x[1]
                x = x .- x_bias
                
                line = split(readline(io))
                y = parse.(Float64, line[1:2])
                y_bias = y[1]
                y = y .- y_bias

                line = split(readline(io))
                z = parse.(Float64, line[1:2])
                z_bias = z[1]
                z = z .- z_bias

                bias = [x_bias, y_bias, z_bias]
                
                bbox = [x[2], 0.0, 0.0, 0.0, y[2], 0.0, 0.0, 0.0, z[2]] * 1u"Å"
                box = [bbox[1:3], bbox[4:6], bbox[7:9]]
            end
            if "Atoms" in line
                readline(io)
                
                for l = 1:num_atoms
                    line = split(readline(io))

                    push!(elements, parse.(Int, line[2]))
                    pos = parse.(Float64, line[3:5]) - bias
                    push!(positions, SVector{3}(pos))
                end
            end
            if "Velocities" in line
                readline(io)
            
                for l = 1:num_atoms
                    line = split(readline(io))
                    vel = parse.(Float64, line[2:4])
                    push!(velocities, SVector{3}(vel))
                end

            end
        end
        atoms = Vector{Atom}(undef, num_atoms)
        for i = 1:num_atoms 
            if isempty(velocities)
                atoms[i] = Atom( lammps.elements[elements[i]], positions[i] * 1u"Å") 
            else
                atoms[i] = Atom( lammps.elements[elements[i]], positions[i] * 1u"Å", velocities[i] * 1u"Å/ps") 
            end
        end
        system = FlexibleSystem(atoms, box, lammps.boundary_conditions)
    end
    return system
end


