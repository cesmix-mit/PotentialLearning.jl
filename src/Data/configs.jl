abstract type ConfigurationDataSet{T} end
struct Configuration{S<:Real, T<:CFG_TYPE{<:S}} <: ConfigurationDataSet{S} 
    data :: Dict{DataType, T}
end

function Configuration(data::CFG_TYPE{T}...) where T <: Real
    Configuration{T, CFG_TYPE{T}}(Dict{DataType, CFG_TYPE{T}}(zip(typeof.(data), data)))
end
function Base.show(io::IO, c::Configuration{S, T}) where {S, T} 
    types = string(collect(keys(c.data)))[10:end-1]
    print(io, "Configuration{S, $types}")
end
function get_data(c::Configuration{S, CFG_TYPE{S}}, dt) where S
    c.data[dt{S}]
end

get_system(c::Configuration) = get_data(c, FlexibleSystem)
get_positions(c::Configuration) = position(get_data(c, FlexibleSystem))
get_energy(c::Configuration) = get_data(c, Energy)
get_descriptors(c::Configuration) = get_data(c, LocalDescriptors)
get_forces(c::Configuration) = get_data(c, Forces)
get_force_descriptors(c::Configuration) = get_data(c, ForceDescriptors)

abstract type DataBase{T} end 

struct DataSet{T} <: DataBase{T} 
    Configurations :: Vector{Configuration{T, CFG_TYPE{T}}}
end

Base.length(ds::DataSet{T}) where T = length(ds.Configurations)
Base.getindex(ds::DataSet{T}, i::Int) where T = ds.Configurations[i]
Base.getindex(ds::DataSet{T}, i::Vector{<:Int}) where T = DataSet(ds.Configurations[i])
Base.getindex(ds::DataSet{T}, i::Union{UnitRange{<:Int}, StepRange{<:Int, <:Int}}) where T = DataSet(ds.Configurations[i])
Base.firstindex(ds::DataSet{T}) where T = ds[1]
Base.lastindex(ds::DataSet{T}) where T = length(ds)
Base.iterate(ds::DataSet{T}, state=1) where T = state > length(ds) ? nothing : (ds[state], state+1)


function Base.show(io::IO, ds::DataSet{T}) where T 
    print(io, "DataSet{$T, num_configs = $(length(ds.Configurations))} \n")
    print(io, "\t $(ds.Configurations[1])")
    if length(ds) > 1
        print(io, "\n\t $(ds.Configurations[2])")
    end
    if length(ds) > 2
        print(io, "\n\t ⋮\n\t $(ds.Configurations[end])")
    end
end

# Base.show(io::IO, c::Config) = print(io, "Configuration{$(string([ti for ti in c.types])[10:end-1])")

# struct FullConfiguration{T} <: AtomicData where T <: Real
#     e :: Energy{T}
#     B :: Vector{LocalDescriptor{T}}
#     f :: Vector{Force{T}}
#     dB :: Vector{ForceDescriptor{T}}
# end
# Base.show(io::IO, fc::FullConfiguration{T}) where T = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.e), $(fc.B[1]), Force{3, $T}, $(fc.dB[1])")

# struct EnergyConfiguration{T} <: AtomicData where T <: Real
#     e :: Energy{T}
#     B :: Vector{LocalDescriptor{T}}
# end
# Base.show(io::IO, fc::EnergyConfiguration{T}) where T = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.e), $(fc.B[1])")

# struct ForceConfiguration{T} <: AtomicData where T <: Real
#     f :: Vector{Force{T}}
#     dB :: Vector{ForceDescriptor{T}}
# end
# Base.show(io::IO, fc::ForceConfiguration{T}) where T = print(io, "Configuration{$T, num_atoms = $(length(fc.f))}(Force{3, $T}, $(fc.dB[1])")

# struct LocalDescriptorSet{T} <: AtomicData where T <: Real
#     B :: Vector{LocalDescriptor{T}}
# end
# Base.show(io::IO, fc::LocalDescriptorSet{T}) where T = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.B[1])")

# function LocalDescriptorSet(B::Vector{Vector{T}}) where T<:Real
#     LocalDescriptorSet(LocalDescriptor.(B))
# end

# struct ForceDescriptorSet{T} <: AtomicData where T 
#     B :: Vector{ForceDescriptor{T}}
# end
# Base.show(io::IO, fc::ForceDescriptorSet{T}) where T = print(io, "Configuration{$T, num_atoms = $(length(fc.B))}($(fc.B[1])")


# function ForceDescriptorSet(B::Vector{Vector{T}}) where T<:Vector{<:Real}
#     LocalDescriptorSet([LocalDescriptor(bi for bi in B)])
# end

# function Configuration(e :: Energy{T}, B :: Vector{LocalDescriptor{T}}) where T <: Real
#     EnergyConfiguration(e, B)
# end

# function Configuration(e :: Energy{T}, B :: Vector{LocalDescriptor{T}}, f::Vector{Force{T}}, dB::Vector{ForceDescriptor{T}}) where T <: Real
#     FullConfiguration(e, B, f, dB)
# end

# function Configuration(f::Vector{Force{T}}, dB::Vector{ForceDescriptor{T}}) where T <: Real
#    ForceConfiguration(f, dB)
# end

# function Configuration(B :: Vector{LocalDescriptor{T}}) where T <: Real
#     LocalDescriptorSet(B)
# end

# function Configuration(dB :: Vector{ForceDescriptor{T}}) where T <: Real
#     ForceDescriptorSet(dB)
# end
