###########################################################
###########################################################
### Data 
abstract type Data{T} end 
abstract type ConfigurationData{T} <: Data{T} end 
abstract type AtomicData{T} <: Data{T} end

CFG_TYPE{T} = Union{AtomsBase.AbstractSystem{T}, ConfigurationData{T}} 

struct Energy{T<:Real} <: ConfigurationData{T} 
    d :: T
    u :: Unitful.FreeUnits
end
Base.show(io::IO, e::Energy) = print(io, e.d, " u","\"$(e.u)\"")
get_values(e::Energy) = e.d

function Energy(e::T) where T <: Real
    Energy(e, u"eV")
end

struct LocalDescriptor{T<:Real} <: AtomicData{T} 
    b :: Vector{T}
end
Base.length(ld::LocalDescriptor) = length(ld.b)
Base.show(io::IO, l::LocalDescriptor{T}) where T = print(io, "LocalDescriptor{$(length(l)), $T}")
get_values(ld::LocalDescriptor) = ld.b

struct LocalDescriptors{T<:Real} <: ConfigurationData{T} 
    b :: Vector{LocalDescriptor{T}}
end
function LocalDescriptors(b::Vector{Vector{T}}) where T<:Real 
    LocalDescriptors(LocalDescriptor.(b))
end
function LocalDescriptors(b::Matrix{T}) where T<:Real
    LocalDescriptors([b[i, :] for i = 1:size(b, 1)])
end
Base.length(ld::LocalDescriptors{T}) where T = length(ld.b)
Base.getindex(ld::LocalDescriptors{T}, i::Int) where T = ld.b[i]
Base.getindex(ld::LocalDescriptors{T}, i::Vector{<:Int}) where T = ld.b[i]
Base.getindex(ld::LocalDescriptors{T}, i::StepRange{<:Int, <:Int}) where T = ld.b[i]
Base.firstindex(ld::LocalDescriptors{T}) where T = 1
Base.lastindex(ld::LocalDescriptors{T}) where T = length(ld)
Base.iterate(ld::LocalDescriptors{T}, state=1) where T = state > length(ld) ? nothing : (ld[state], state+1)
Base.show(io::IO, l::LocalDescriptors{T}) where T = print(io, "LocalDescriptors{n = $(length(l)), d = $(length(l.b[1])), $T}")
get_values(ld::LocalDescriptors) = [ldi.b for ldi in ld.b]

struct Force{T<:Real} <: AtomicData{T}  # Per atom force (Vector of with 3 components)
    f :: Vector{T}
    u :: Unitful.FreeUnits
end
Base.show(io::IO, f::Force) = print(io, f.f, " u","\"$(f.u)\"")
get_values(f::Force) = f.f

function Force(f::Vector{T}) where T<:Real
    Force(f, u"eV/Å")
end

struct Forces{T<:Real} <: ConfigurationData{T} 
    f :: Vector{Force{T}}
end
function Forces(f::Vector{Vector{T}}, u::Unitful.FreeUnits) where T<:Real 
    Forces(Force.(f, (u,)))
end

Base.show(io::IO, f::Forces) = print(io, "Forces{n = $(length(f.f)), $(f.f[1].u)}")
get_values(f::Force) = [fi.f for fi in f.f]

struct ForceDescriptor{T<:Real} <: AtomicData{T} # Per atom descriptors (Vector with 3 vector components)
    b :: Vector{<:Vector{T}}
end
get_values(fd::ForceDescriptor) = fd.b
Base.show(io::IO, l::ForceDescriptor{T}) where T = print(io, "ForceDescriptor{$(length(l.b)), $T}")

struct ForceDescriptors{T<:Real} <: ConfigurationData{T}
    b :: Vector{ForceDescriptor{T}}
end
function ForceDescriptors(fd::Vector{Vector{<:Vector{T}}}) where T<:Real 
    ForceDescriptors(ForceDescriptor.(fd))
end

Base.length(fd::ForceDescriptors{T}) where T = length(fd.b)
Base.getindex(fd::ForceDescriptors{T}, i::Int) where T = fd.b[i]
Base.getindex(fd::ForceDescriptors{T}, i::Vector{<:Int}) where T = fd.b[i]
Base.getindex(fd::ForceDescriptors{T}, i::StepRange{<:Int, <:Int}) where T = fd.b[i]
Base.firstindex(fd::ForceDescriptors{T}) where T = 1
Base.lastindex(fd::ForceDescriptors{T}) where T = length(fd)
Base.iterate(fd::ForceDescriptors{T}, state=1) where T = state > length(fd) ? nothing : (fd[state], state+1)
Base.show(io::IO, l::ForceDescriptors{T}) where T = print(io, "ForceDescriptors{n = $(length(l)), d = $(length(l[1])), $T}")
values(fd::ForceDescriptors) = [bi.b for bi in fd]



##########
##########