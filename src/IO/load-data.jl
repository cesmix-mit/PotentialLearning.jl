# This code will be added to PotentialLearning.jl

function load_data(file; max_entries = 2000, T = Float64)
    systems  = AbstractSystem[]
    energies = T[]
    forces    = Vector{SVector{3, T}}[]
    stresses = SVector{6, T}[]
    open(file, "r") do io
        count = 1
        while !eof(io) && (count <= max_entries)
            # Read info line
            line = readline(io)
            num_atoms = parse(Int, line)

            line = readline(io)
            lattice_line = match(r"Lattice=\"(.*?)\" ", line).captures[1]
            lattice = parse.(T, split(lattice_line)) * 1u"Å"
            box = [lattice[1:3],
                   lattice[4:6], 
                   lattice[7:9]]
            bias = -5.0 
            try 
                energy_line = match(r"energy=(.*?) ", line).captures[1]
                energy = parse(T, energy_line)
                push!(energies, energy)
            catch
                push!(energies, NaN)
            end

            try
                stress_line = match(r"stress=\"(.*?)\" ", line).captures[1]
                stress = parse.(T, split(stress_line))
                push!(stresses, SVector{6}(stress))
            catch
                push!(stresses, SVector{6}( fill(NaN, (6,))))
            end

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
            force = Vector{SVector{3, T}}(undef, num_atoms) 
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
                        position = SVector{3}(parse.(T, line[line_count:line_count+2])) .- bias
                        line_count += 3
                    elseif prop[1] == "move_mask"
                        ft = Symbol(line[line_count])
                        line_count += 1 
                    elseif prop[1] == "tags"
                        ft = Symbol(line[line_count])
                        line_count += 1 
                    elseif prop[1] == "forces"
                        force[i] = SVector{3}(parse.(T, line[line_count:line_count+2]))
                        line_count += 3
                    else
                        length = parse(Int, prop[3])
                        if length == 1
                            data = merge(data, Dict( (prop[1] => line[line_count]) ) )
                        else
                            data = merge(data, Dict( (prop[1] => line[line_count:line_count+length-1]) ) )
                        end
                    end
                end
                atoms[i] = AtomsBase.Atom(element, position .* 1u"Å", data = data)
            end

            push!(forces, force)
            system = FlexibleSystem(atoms, box, bc)
            push!(systems, system)
            count += 1
        end
    end
    return systems, energies, forces, stresses
end
