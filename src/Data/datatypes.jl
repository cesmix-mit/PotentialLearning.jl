###########################################################
###########################################################
### Data 
"""
    Data 

Abstract supertype of ConfigurationData.
"""
abstract type Data end
"""
    ConfigurationData <: Data 

Abstract type declaring the type of data that is unique to a particular configuration (instead of just an atom).
"""
abstract type ConfigurationData <: Data end
"""
    AtomicData <: Data 
Abstract type declaring the type of information that is unique to a particular atom (instead of a whole configuration).
"""
abstract type AtomicData <: Data end

CFG_TYPE = Union{AtomsBase.FlexibleSystem,ConfigurationData}

"""
    get_values(v::SVector)

Removes units from a position.
"""
get_values(v::SVector) = ustrip(v)

"""
    Energy <: ConfigurationData
        d :: Real
        u :: Unitful.FreeUnits

Convenience struct that holds energy information (and corresponding units). Default unit is eV
"""
struct Energy <: ConfigurationData
    d::Real
    u::Unitful.FreeUnits
end
Base.show(io::IO, e::Energy) = print(io, e.d, " u", "\"$(e.u)\"")
"""
    get_values(e::Energy) <: Real 
Get the underlying real value (= e.d)
"""
get_values(e::Energy) = e.d

function Energy(e)
    Energy(e, u"eV")
end

"""
    LocalDescriptor <: AtomicData 

A vector corresponding to the descriptor for a particular atom's neighborhood.
"""
struct LocalDescriptor <: AtomicData
    b::Vector{<:Real}
end
Base.length(ld::LocalDescriptor) = length(ld.b)
Base.show(io::IO, l::LocalDescriptor) = print(io, "LocalDescriptor{$(length(l))}")
get_values(ld::LocalDescriptor) = ld.b
Base.:*(W::AbstractMatrix{T}, ld::LocalDescriptor) where {T<:Real} =
    LocalDescriptor(W * ld.b)

"""
    LocalDescriptors <: ConfigurationData 

A vector of LocalDescriptor, which now should represent all local descriptors for atoms in a configuration.
"""
struct LocalDescriptors <: ConfigurationData
    b::Vector{LocalDescriptor}
end

function LocalDescriptors(b::Vector{Vector{T}}) where {T<:Real}
    LocalDescriptors(LocalDescriptor.(b))
end
function LocalDescriptors(b::Matrix)
    LocalDescriptors([b[i, :] for i = 1:size(b, 1)])
end
Base.length(ld::LocalDescriptors) = length(ld.b)
Base.getindex(ld::LocalDescriptors, i::Int) = ld.b[i]
Base.getindex(ld::LocalDescriptors, i::Vector{<:Int}) = ld.b[i]
Base.getindex(ld::LocalDescriptors, i::StepRange{<:Int,<:Int}) = ld.b[i]
Base.firstindex(ld::LocalDescriptors) = 1
Base.lastindex(ld::LocalDescriptors) = length(ld)
Base.iterate(ld::LocalDescriptors, state = 1) =
    state > length(ld) ? nothing : (ld[state], state + 1)
Base.show(io::IO, l::LocalDescriptors) =
    print(io, "LocalDescriptors{n = $(length(l)), d = $(length(l.b[1]))}")
get_values(ld::LocalDescriptors) = [ldi.b for ldi in ld.b]
Base.:*(W::AbstractMatrix{T}, ld::LocalDescriptors) where {T} =
    LocalDescriptors((W,) .* ld.b)

"""
    Force <: AtomicData 
        f :: Vector{<:Real}
        u :: Unitful.FreeUnits

Contains the force with (x,y,z)-components in f with units u. Default unit is "eV/Å". 
"""
struct Force <: AtomicData  # Per atom force (Vector of with 3 components)
    f::Vector{<:Real}
    u::Unitful.FreeUnits
end
Base.show(io::IO, f::Force) = print(io, f.f, " u", "\"$(f.u)\"")
get_values(f::Force) = f.f

function Force(f::Vector)
    Force(f, u"eV/Å")
end
"""
    Forces <: ConfigurationData
        f :: Vector{force}

Forces is a struct that contains all force information in a configuration.
"""
struct Forces <: ConfigurationData
    f::Vector{Force}
end
function Forces(f::Vector{<:Vector{T}}, u::Unitful.FreeUnits) where {T<:Real}
    Forces(Force.(f, (u,)))
end

Base.show(io::IO, f::Forces) = print(io, "Forces{n = $(length(f.f)), $(f.f[1].u)}")
get_values(f::Forces) = [fi.f for fi in f.f]

"""
    ForceDescriptor <: AtomicData
        b :: Vector{<:Vector{<:Real}}

Contains the x,y,z components (out vector) of the force descriptor (inner vector).
"""
struct ForceDescriptor <: AtomicData # Per atom descriptors (Vector with 3 vector components)
    b::Vector{<:Vector{<:Real}}
end
get_values(fd::ForceDescriptor) = fd.b
Base.length(fd::ForceDescriptor) = length(fd.b)
Base.show(io::IO, l::ForceDescriptor) = print(io, "ForceDescriptor{$(length(l.b))}")
Base.:*(W::AbstractMatrix{T}, fd::ForceDescriptor) where {T<:Real} =
    ForceDescriptor((W,) .* fd.b)

function ForceDescriptor(fd::Matrix{T}) where {T<:Real}
    ForceDescriptor([fd[i, :] for i = 1:3])
end
"""
    ForceDescriptors <: ConfigurationData
        b :: Vector{ForceDescriptor}

A container holding all of the ForceDescriptors for all atoms in a configuration.
"""
struct ForceDescriptors <: ConfigurationData
    b::Vector{ForceDescriptor}
end
function ForceDescriptors(fd::Vector{<:Vector{<:Vector{T}}}) where {T<:Real}
    ForceDescriptors(ForceDescriptor.(fd))
end
function ForceDescriptors(fd::Vector{<:Matrix{T}}) where {T<:Real}
    ForceDescriptors(ForceDescriptor.(fd))
end

Base.length(fd::ForceDescriptors) = length(fd.b)
Base.getindex(fd::ForceDescriptors, i::Int) = fd.b[i]
Base.getindex(fd::ForceDescriptors, i::Vector{<:Int}) = fd.b[i]
Base.getindex(fd::ForceDescriptors, i::StepRange{<:Int,<:Int}) = fd.b[i]
Base.firstindex(fd::ForceDescriptors) = 1
Base.lastindex(fd::ForceDescriptors) = length(fd)
Base.iterate(fd::ForceDescriptors, state = 1) =
    state > length(fd) ? nothing : (fd[state], state + 1)
Base.show(io::IO, l::ForceDescriptors) =
    print(io, "ForceDescriptors{n = $(length(l)), d = $(length(l[1]))}")
get_values(fd::ForceDescriptors) = [bi.b for bi in fd]
Base.:*(W::AbstractMatrix{T}, fd::ForceDescriptors) where {T<:Real} =
    ForceDescriptors((W,) .* fd.b)



##########
##########
