export @savevar, @savecsv, @savefig


"""
    savevar(path, var)

`path`: path where the variable will be saved.
`var`: variable to be saved.

Simplifies the saving of a variable.

"""
macro savevar(path, var)
    quote
        write($(esc(path)) * $(string(var)) * ".dat", string($(esc(var))))
    end
end


"""
    savecsv(path, dict)

`path`: path where the dictionary will be saved.
`dict`: dictionary to be saved.

Simplifies the saving of a dictionary to a CSV file.

"""
macro savecsv(path, dict)
    quote
        CSV.write($(esc(path)) * $(string(dict)) * ".csv", $(esc(dict)), header = false)
    end
end


"""
    savefig(path, fig)

`path`: path where the figure will be saved.
`fig`: figure to be saved.

Simplifies the saving of a figure.

"""
macro savefig(path, fig)
    quote
        savefig($(esc(fig)), $(esc(path)) * $(string(fig)) * ".png")
    end
end
