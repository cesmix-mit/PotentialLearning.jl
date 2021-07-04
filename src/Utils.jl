using Base: Float64
using LinearAlgebra

# Point
mutable struct Point
    x::Float64
    y::Float64
    z::Float64
end
LinearAlgebra.:norm(r::Point) = norm([r.x, r.y, r.z])
Base.:+(r1::Point, r2::Point) = Point(r1.x + r2.x, r1.y + r2.y, r1.z + r2.z)
Base.:-(r1::Point, r2::Point) = Point(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z)
Base.:*(k::Number, r::Point) = Point(k * r.x, k * r.y, k * r.z)
Base.:*(r::Point, k::Number) = k * r

