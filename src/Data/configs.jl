abstract type ConfigurationDataSet end
struct Configuration <: ConfigurationDataSet
    data :: Dict{DataType, CFG_TYPE}
end
"""
    Configuration(data::Union{AtomsBase.FlexibleSystem, ConfigurationData} )

A Configuration is a data struct that contains information unique to a particular configuration of atoms (Energy, LocalDescriptors, ForceDescriptors, and a FlexibleSystem) in a dictionary. 
    Example:
    '''julia
        e = Energy(-0.57, u"eV")
        ld = LocalDescriptors(...)
        c = Configuration(e, ld)
    '''

Configurations can be added together, which merges the data dictionaries 
    '''julia 
    c1 = Configuration(e) # Contains energy 
    c2 = Configuration(f) # contains forces 
    c = c1 + c2 # c <: Configuration, contains energy and forces
    '''

"""
function Configuration(data::CFG_TYPE...) 
    Configuration(Dict{DataType, CFG_TYPE}(zip(typeof.(data), data)))
end
function Base.show(io::IO, c::Configuration) 
    types = string(collect(keys(c.data)))[10:end-1]
    print(io, "Configuration{S, $types}")
end
function Base.:+(c1::Configuration, c2::Configuration)
    Configuration(merge(c1.data, c2.data))
end
function Base.:+(c::Configuration, d::CFG_TYPE)
    c + Configuration(d)
end
"""
    get_system(c::Configuration) <: AtomsBase.AbstractSystem

Retrieves the AtomsBase system (if available) in the Configuration c. 
"""
function get_system(c::Configuration)
    i = findall( keys(c.data) .<: AtomsBase.AbstractSystem )[1]
    collect(values(c.data))[i]
end
"""
    get_positions(c::Configuration) <: Vector{SVector}

Retrieves the AtomsBase system positions (if available) in the Configuration c. 
"""
get_positions(c::Configuration) = position(get_system(c))
"""
    get_energy(c::Configuration) <: Energy

Retrieves the energy (if available) in the Configuration c. 
"""
get_energy(c::Configuration) = c.data[Energy]
"""
    get_local_descriptors(c::Configuration) <: LocalDescriptors

Retrieves the local descriptors (if available) in the Configuration c. 
"""
get_local_descriptors(c::Configuration) = c.data[LocalDescriptors]
"""
    get_forces(c::Configuration) <: Forces

Retrieves the forces (if available) in the Configuration c. 
"""
get_forces(c::Configuration) = c.data[Forces]
"""
    get_force_descriptors(c::Configuration) <: ForceDescriptors

Retrieves the force descriptors (if available) in the Configuration c. 
"""
get_force_descriptors(c::Configuration) = c.data[ForceDescriptors]
"""
    DataBase 
Abstract type for DataSets. 
"""
abstract type DataBase end 
"""
    DataSet 
Struct that holds vector of configuration. Most operations in PotentialLearning are built around the DataSet structure.
"""
struct DataSet <: DataBase 
    Configurations :: Vector{Configuration}
end

Base.length(ds::DataSet) = length(ds.Configurations)
Base.getindex(ds::DataSet, i::Int) = ds.Configurations[i]
Base.getindex(ds::DataSet, i::Vector{<:Int}) = DataSet(ds.Configurations[i])
Base.getindex(ds::DataSet, i::Union{UnitRange{<:Int}, StepRange{<:Int, <:Int}}) = DataSet(ds.Configurations[i])
Base.firstindex(ds::DataSet) = ds[1]
Base.lastindex(ds::DataSet) = length(ds)
Base.iterate(ds::DataSet, state=1) = state > length(ds) ? nothing : (ds[state], state+1)


function Base.show(io::IO, ds::DataSet) 
    print(io, "DataSet{num_configs = $(length(ds.Configurations))} \n")
    print(io, "\t $(ds.Configurations[1])")
    if length(ds) > 1
        print(io, "\n\t $(ds.Configurations[2])")
    end
    if length(ds) > 2
        print(io, "\n\t ⋮\n\t $(ds.Configurations[end])")
    end
end
