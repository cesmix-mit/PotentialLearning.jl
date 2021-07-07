using Base: Float64
using LinearAlgebra

# Position
mutable struct Position
    x::Float64
    y::Float64
    z::Float64
end
LinearAlgebra.:norm(r::Position) = norm([r.x, r.y, r.z])
Base.:+(r1::Position, r2::Position) = Position(r1.x + r2.x, r1.y + r2.y, r1.z + r2.z)
Base.:-(r1::Position, r2::Position) = Position(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z)
Base.:*(k::Number, r::Position) = Position(k * r.x, k * r.y, k * r.z)
Base.:*(r::Position, k::Number) = k * r

