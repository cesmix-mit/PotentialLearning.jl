
export @savevar, @savecsv, @savefig

path = ""
var = ""
dict = ""


"""
    savevar(path, var)
    
Simplifies the saving of a variable.

"""
macro savevar(path, var)
     quote
        write($(esc(path)) * $(string(var)) * ".dat", string($(esc(var))))
     end
end


"""
    savecsv(path, dict)
    
Simplifies the saving of a dictionary to a CSV file.

"""
macro savecsv(path, dict)
    quote
        CSV.write($(esc(path)) * $(string(dict)) * ".csv", $(esc(dict)), header = false)
    end
end

"""
    savefig(path, var)
    
Simplifies the saving of a figure.

"""
macro savefig(path, var)
    quote
        savefig($(esc(var)), $(esc(path)) * $(string(var)) * ".png")
    end
end

