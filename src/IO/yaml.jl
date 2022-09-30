import YAML as YML
"""
    struct YAML <: IO
        energy_units :: Unitful.FreeUnits
        distance_units :: Unitful.FreeUnits
    end
"""
struct YAML <: IO 
    energy_units :: Unitful.FreeUnits
    distance_units :: Unitful.FreeUnits
end
function YAML(; energy_units = u"eV", distance_units = u"Å")
    YAML(energy_units, distance_units)
end

function load_yaml(yaml_dict::Dict, yaml::YAML)
    box = yaml_dict["box"]
    data = yaml_dict["data"]
    bcs = yaml_dict["boundary"][1:3]

    thermo = yaml_dict["thermo"]
    e = thermo[2]["data"][findall(thermo[1]["keywords"] .== "PotEng")[1]] 
    if e == "-nan"
        e = Energies(NaN, yaml.energy_units) 
    else
        e = Energies(e, yaml.energy_units)
    end

    pos = map(x->x[1:3], data)
    forces = map(x->x[4:end], data)
    p = SVector{3}.(pos).*yaml.distance_units
    f = Forces(Force.(forces, (yaml.energy_units/yaml.distance_units,)))
    xlo, xhi = box[1]
    ylo, yhi = box[2]
    zlo, zhi = box[3]
    box = SVector{3}.([[xhi-xlo 0.0 0.0].*yaml.distance_units, 
                    [0.0 yhi-ylo 0.0].*yaml.distance_units,
                    [0.0 0.0 zhi-zlo].*yaml.distance_units])

    system = FlexibleSystem([Atom(:Na, p_i) for p_i in p], box, [bc == "p" ? Periodic() : Dirichlet() for bc in bcs])
    Configuration(system, e, f), thermo
end
    
function load_yaml(file::String)
    load_yaml(YML.load_file(file))
end

"""
    load_data(file::string, yaml::YAML)

    Load configurations from a yaml file into a Vector of Flexible Systems, with Energies and Force.
    Returns 
        ds - DataSet
        t = Vector{Dict} (any miscellaneous info from yaml file)
    
"""
function load_data(file::String, yaml::YAML)
    c = Configuration[]
    t = Vector{Dict}[]
    for fs in YML.load_all(open(fs->read(fs, String), file))
        ci, ti = load_yaml(fs)
        push!(t, ti)
        push!(c, ci)
    end
    DataSet(c), t
end
