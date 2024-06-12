"""
    save_var(path, var)

`path`: path where the variable will be saved.
`var`: variable to be saved.

Simplifies the saving of a variable.

"""
macro save_var(path, var)
     quote
        write($(esc(path)) * $(string(var)) * ".dat", string($(esc(var))))
     end
end


"""
    save_dict(path, dict)

`path`: path where the dictionary will be saved.
`dict`: dictionary to be saved.

Simplifies the saving of a dictionary to a CSV file.

"""
macro save_dict(path, dict)
    quote
        CSV.write($(esc(path)) * $(string(dict)) * ".csv", $(esc(dict)), header = false)
    end
end

"""
    save_dataframe(path, df)

`path`: path where the dictionary will be saved.
`df`: dataframe to be saved.

Simplifies the saving of a dataframe to a CSV file.

"""
macro save_dataframe(path, df)
    quote
        CSV.write($(esc(path)) * $(string(df)) * ".csv", $(esc(df)))
    end
end


"""
    save_fig(path, fig)

`path`: path where the figure will be saved.
`fig`: figure to be saved.

Simplifies the saving of a figure.

"""
macro save_fig(path, fig)
    quote
        savefig($(esc(fig)), $(esc(path)) * $(string(fig)) * ".png")
    end
end

