
export savevar, savecsv, savefig

"""
    savevar(path, var)
    
Simplifies the saving of a variable.

"""
macro savevar(path, var)
    quote
        write("$(path)" * $(string(var)) * ".dat", string($(esc(var))))
    end
end


"""
    savevar(path, dict)
    
Simplifies the saving of a dictionary to a CSV file.

"""
macro savecsv(path, dict)
    return :( CSV.write("$(path)" * $(string(dict)) * ".csv", $(dict), header = false) )
end

"""
    savefig(path, var)
    
Simplifies the saving of a figure.

"""
macro savefig(path, var)
    return :( savefig($(var), "$(path)" * $(string(var)) * ".png") )
end

