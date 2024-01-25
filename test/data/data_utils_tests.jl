using Unitful 
using InteratomicPotentials

# TODO: Many tests needed to be added for structs/fns in Data section

#= 
This data is a small subset of the TiAl_tutorial.xyz artifact that was used for
testing and documentation in Ace1pack https://github.com/ACEsuit/ACE1pack.jl
=# 
TiAl_ds = load_data("./data/TiAl_examples.xyz", ExtXYZ(u"eV", u"Å"))

ace = ACE(species           = [:Ti, :Al],
          body_order        = 3,
          polynomial_degree = 6,
          wL                = 1.5,
          csp               = 1.0,
          r0                = 2.9,
          rcutoff           = 5.5 )

e_descr = compute_local_descriptors(TiAl_ds, ace; pbar=false)
@test length(e_descr) == 5
@test typeof(e_descr) <: Vector{LocalDescriptors}
@test length(e_descr[1]) == 2  # num atoms in 1st config
@test length(e_descr[end]) == 54  # num atoms in last config
@test length(e_descr[1][1]) == 104  # number local descrs

#TODO need to document reference better, preferably w/ repro Manifest.toml etc.
#long-run, should do more than spot-checking
@testset "data_utils: local descriptors ref check" begin 
    #= 
    spot-checking descriptor values (and order), based off of the above basis 
    IP.jl v.0.2.7
    ACE1 v0.10.7
    JuLIP v0.11.5
    Julia 1.9.2+0.aarch64.apple.darwin14
    =#

    ed_vals11 = get_values(e_descr[1][1])
    ed_vals32 = get_values(e_descr[3][2]) 
    ed_vals51 = get_values(e_descr[5][1])

    @test ed_vals11[10]     ≈ 0.0 
    @test ed_vals11[end-15] ≈ -5.541800263121354  #TODO should actual specify tols

    @test ed_vals32[1]  ≈ 0.9105447479710141 
    @test ed_vals32[52] ≈ 7.927103583234019

    @test ed_vals51[100] ≈ -0.3889692376173769
    @test ed_vals51[end] ≈ 4.344079434030667
end
 
f_descr = compute_force_descriptors(TiAl_ds,ace; pbar=false)
@test length(f_descr) ==5 
@test typeof(f_descr) <: Vector{ForceDescriptors} 
@test length(f_descr[1]) == 2 
@test length(f_descr[end]) == 54 
@test length(f_descr[1][1]) == 3 # 3 cartesian directions
@test length(get_values(f_descr[1][1])[1]) == 104 

@testset "data_utils: force descriptors ref check" begin 
    #= 
    spot-checking descriptor values (and order), based off of the above basis 
    IP.jl v.0.2.7
    ACE1 v0.10.7
    JuLIP v0.11.5
    Julia 1.9.2+0.aarch64.apple.darwin14
    =#

    fd_vals11_1 = get_values(f_descr[1][1])[1]
    fd_vals32_2 = get_values(f_descr[3][2])[2]
    fd_vals51_3 = get_values(f_descr[5][1])[3]

    @test fd_vals11_1[14]  ≈ 0.0
    @test fd_vals11_1[66]  ≈ 0.21012541753661917
    @test fd_vals11_1[90]  ≈ 0.07680220708487306 
    @test fd_vals11_1[end] ≈ 0.0

    @test fd_vals32_2[7]  ≈ 0.09927939932561863
    @test fd_vals32_2[31] ≈ -1.1185724606158156 
    @test fd_vals32_2[80] ≈ 1.780519258001138

    @test fd_vals51_3[23]  ≈ 0.0
    @test fd_vals51_3[50]  ≈ 8.247266509962259
    @test fd_vals51_3[end] ≈ 8.194593298142163
end
