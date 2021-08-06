# Convert the DFT data from https://github.com/FitSNAP/FitSNAP/tree/master/examples/Ta_Linear_JCP2014/JSON
# to the current format required by PotentialLearning.jl. 
# The current format will probably change in the coming months.

using JSON

path = "./JSON/"
for d in readdir(path)
    sub_path = joinpath(path, d, "DATA")
    rm(sub_path, recursive=true)
    dirs = readdir(joinpath(path, d))
    mkdir(sub_path)
    for (i, f) in enumerate(dirs)
        global dft_data = Dict()
        open(joinpath(path, d, f) , "r") do file
            @show joinpath(path, d, f)
            readline(file)
            dft_data = JSON.parse(readline(file))
        end
        ssub_path = joinpath(sub_path, "$i")
        mkdir(ssub_path)
        
        open(joinpath(ssub_path, "ENERGY"), "w") do file
            write(file, "$(dft_data["Dataset"]["Data"][1]["Energy"])\n")
        end
    
        open(joinpath(ssub_path, "FORCES"), "w") do file
            for force in dft_data["Dataset"]["Data"][1]["Forces"]
                write(file, "$(force[1]) $(force[2]) $(force[3])\n")
            end
        end
        
        open(joinpath(ssub_path, "DATA") , "w") do file
            write(file, "LAMMPS DATA file\n")
            write(file, "\n")
            write(file, "$(dft_data["Dataset"]["Data"][1]["NumAtoms"]) atoms\n")
            write(file, "0 bonds\n")
            write(file, "0 angles\n")
            write(file, "0 dihedrals\n")
            write(file, "0 impropers\n")
            write(file, "\n")
            write(file, "1 atom types\n")
            write(file, "0 bond types\n")
            write(file, "0 angle types\n")
            write(file, "\n")
            write(file, "0.0 12.990503 xlo xhi\n")
            write(file, "0.0 11.250148 ylo yhi\n")
            write(file, "0.0 15.845291 zlo zhi\n")
            write(file, "\n")
            write(file, "Masses\n")
            write(file, "\n")
            write(file, "1 73\n")
            write(file, "\n")
            write(file, "Atoms\n")
            write(file, "\n")
            positions = dft_data["Dataset"]["Data"][1]["Positions"]
            for (j, pos) in enumerate(positions)
                write(file, "$j 1 $(pos[1]) $(pos[2]) $(pos[3]) 0 0 0\n")
            end
        end
    end
end

