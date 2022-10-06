abstract type ConfigurationDataSet end
struct Configuration <: ConfigurationDataSet
    data :: Dict{DataType, CFG_TYPE}
end

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

function get_system(c::Configuration)
    i = findall( keys(c.data) .<: AtomsBase.AbstractSystem )[1]
    collect(values(c.data))[i]
end

get_positions(c::Configuration) = position(get_system(c))
get_energy(c::Configuration) = c.data[Energy]
get_descriptors(c::Configuration) = c.data[LocalDescriptors]
get_forces(c::Configuration) = c.data[Forces]
get_force_descriptors(c::Configuration) = c.data[ForceDescriptors]

abstract type DataBase end 

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

# Base.show(io::IO, c::Config) = print(io, "Configuration{$(string([ti for ti in c.types])[10:end-1])")

# struct FullConfiguration{T} <: AtomicData 
#     e :: Energy{T}
#     B :: Vector{LocalDescriptor{T}}
#     f :: Vector{Force{T}}
#     dB :: Vector{ForceDescriptor{T}}
# end
# Base.show(io::IO, fc::FullConfiguration{T}) = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.e), $(fc.B[1]), Force{3, $T}, $(fc.dB[1])")

# struct EnergyConfiguration{T} <: AtomicData 
#     e :: Energy{T}
#     B :: Vector{LocalDescriptor{T}}
# end
# Base.show(io::IO, fc::EnergyConfiguration{T}) = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.e), $(fc.B[1])")

# struct ForceConfiguration{T} <: AtomicData 
#     f :: Vector{Force{T}}
#     dB :: Vector{ForceDescriptor{T}}
# end
# Base.show(io::IO, fc::ForceConfiguration{T}) = print(io, "Configuration{$T, num_atoms = $(length(fc.f))}(Force{3, $T}, $(fc.dB[1])")

# struct LocalDescriptorSet{T} <: AtomicData 
#     B :: Vector{LocalDescriptor{T}}
# end
# Base.show(io::IO, fc::LocalDescriptorSet{T}) = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.B[1])")

# function LocalDescriptorSet(B::Vector{Vector{T}}) where T<:Real
#     LocalDescriptorSet(LocalDescriptor.(B))
# end

# struct ForceDescriptorSet{T} <: AtomicData 
#     B :: Vector{ForceDescriptor{T}}
# end
# Base.show(io::IO, fc::ForceDescriptorSet{T}) = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.B[1])")


# function ForceDescriptorSet(B::Vector{Vector{T}}) where T<:Vector{<:Real}
#     LocalDescriptorSet([LocalDescriptor(bi for bi in B)])
# end

# function Configuration(e :: Energy{T}, B :: Vector{LocalDescriptor{T}}) 
#     EnergyConfiguration(e, B)
# end

# function Configuration(e :: Energy{T}, B :: Vector{LocalDescriptor{T}}, f::Vector{Force{T}}, dB::Vector{ForceDescriptor{T}}) 
#     FullConfiguration(e, B, f, dB)
# end

# function Configuration(f::Vector{Force{T}}, dB::Vector{ForceDescriptor{T}}) 
#    ForceConfiguration(f, dB)
# end

# function Configuration(B :: Vector{LocalDescriptor{T}}) 
#     LocalDescriptorSet(B)
# end

# function Configuration(dB :: Vector{ForceDescriptor{T}}) 
#     ForceDescriptorSet(dB)
# end
