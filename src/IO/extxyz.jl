"""
    ExtXYZ <: IO
"""
struct ExtXYZ <: IO 
    energy_units :: Unitful.FreeUnits
    distance_units :: Unitful.FreeUnits
end
"""
    load_data(file::string, extxyz::ExtXYZ)
    Load configuration from an extxyz file into a DataSet
"""
function load_data(file, extxyz::ExtXYZ; T = Float64)
    configs = Configuration[]
    open(file, "r") do io
        count = 1
        while !eof(io) 
            # Read info line
            line = readline(io)
            num_atoms = parse(Int, line)
            
            line = readline(io)
            lattice_line = match(r"Lattice=\"(.*?)\" ", line).captures[1]
            lattice = parse.(Float64, split(lattice_line)) * extxyz.distance_units
            box = [lattice[1:3],
                        lattice[4:6], 
                        lattice[7:9]]
            energy = try 
                energy_line = match(r"energy=(.*?) ", line).captures[1]
                energy = parse(Float64, energy_line)
                Energy(energy, extxyz.energy_units)
            catch
                Energy(NaN, extxyz.energy_units)
            end
            
            # try
            #     stress_line = match(r"stress=\"(.*?)\" ", line).captures[1]
            #     stress = parse.(Float64, split(stress_line))
            #     push!(stresses, SVector{6}(stress))
            # catch
            #     push!(stresses, SVector{6}( fill(NaN, (6,))))
            # end

            bc = []
            try
                bc_line = match(r"pbc=\"(.*?)\"", line).captures[1]
                bc = [ t=="T" ? Periodic() : DirichletZero() for t in split(bc_line) ]
            catch
                bc = [DirichletZero(), DirichletZero(), DirichletZero()]
            end

            properties = match(r"Properties=(.*?) ", line).captures[1]
            properties = split(properties, ":")
            properties = [properties[i:i+2] for i = 1:3:(length(properties)-1)]
            atoms = Vector{AtomsBase.Atom}(undef, num_atoms)
            forces = Force[]
            for i = 1:num_atoms
                line = split(readline(io))
                line_count = 1
                position = 0.0
                element = 0.0
                data = Dict( () )
                for prop in properties
                    if prop[1] == "species"
                        element = Symbol(line[line_count])
                        line_count += 1 
                    elseif prop[1] == "pos"
                        position = SVector{3}(parse.(T, line[line_count:line_count+2])) 
                        line_count += 3
                    elseif prop[1] == "move_mask"
                        ft = Symbol(line[line_count])
                        line_count += 1 
                    elseif prop[1] == "tags"
                        ft = Symbol(line[line_count])
                        line_count += 1 
                    elseif prop[1] == "forces"
                        push!(forces, Force(parse.(T, line[line_count:line_count+2]), extxyz.energy_units / extxyz.distance_units))
                        line_count += 3
                    else
                        length = parse(Int, prop[3])
                        if length == 1
                            data = merge(data, Dict( (Symbol(prop[1]) => line[line_count]) ) )
                        else
                            data = merge(data, Dict( (Symbol(prop[1]) => line[line_count:line_count+length-1]) ) )
                        end
                    end
                end
                if isempty(data)
                    atoms[i] = AtomsBase.Atom(element, position .* extxyz.distance_units )
                else
                    atoms[i] = AtomsBase.Atom(element, position .* extxyz.distance_units, data... )
                end

            end

            system = FlexibleSystem(atoms, box, bc)
            count += 1
            push!(configs, Configuration(system, energy, Forces(forces)))
        end
    end

    return DataSet(configs)
end


