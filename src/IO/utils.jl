# This code will be added to PotentialLearning.jl
using CSV

macro savevar(path, var)
    quote
        write("$(path)" * $(string(var)) * ".dat", string($(esc(var))))
    end
end

macro savecsv(path, var)
    return :( CSV.write("$(path)" * $(string(var)) * ".csv", $(var), header = false) )
end

macro savefig(path, var)
    return :( savefig($(var), "$(path)" * $(string(var)) * ".png") )
end

