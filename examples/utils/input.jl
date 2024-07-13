# New function to read XYZ files ###############################################
include("$base_path/examples/utils/xyz.jl")

# New functions to get input parameters from OrderedDict #######################

"""
    to_num(str)
    
`str`: string with a number: integer or float

Returns an integer or float.

"""
function to_num(str)
    val = nothing
    if occursin(".", str)
        val = parse(Float64, str)
    else
        val = parse(Int64, str)
    end
    return val
end


"""
    get_input(args)
    
`args`: vector of arguments (strings)

Returns an OrderedDict with the arguments.
See https://github.com/cesmix-mit/AtomisticComposableWorkflows documentation
for information about how to define the input arguments.

"""
function get_input(args)
    input = OrderedDict()
    for (key, val) in partition(args,2,2)
        val = replace(val, " " => "")
        # if val is a boolean
        if val == "true" || val == "false"
            val = val == "true"
        # if val is a vector, e.g. "[1.5,1.5]"
        elseif val[1] == '['
            val = to_num.(split(val[2:end-1], ","))
        # if val is a number, e.g. 1.5 or 1
        elseif tryparse(Float64, val) != nothing
            val = to_num(val)
        end
        input[key] = val
    end
    return input
end

# New function to Split datasets ###############################################

function Base.split(ds, n, m; f = 0.8)
    k = floor(Int64, f * length(ds))
    i_tr = randperm(k)
    i_ts = randperm(length(ds)-k) .+ k
    return @views ds[first(i_tr, n)], ds[first(i_ts, m)]
end



