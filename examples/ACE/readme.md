## Fitting ACE example

This folder contains an example on how to fit a DFT dataset using ACE.


### Chose a DFT dataset

Choose a DFT dataset. Currently, this code accepts either two `xyz` files, one for training and one for testing, or a single `xyz` file, which is automatically split into training and testing. Example datasets can be downloaded from the following urls.
- a-HfO2 dataset: "Machine-learned interatomic potentials by active learning: amorphous and liquid hafnium dioxide". Ganesh Sivaraman, Anand Narayanan Krishnamoorthy, Matthias Baur, Christian Holm, Marius Stan, Gábor Csányi, Chris Benmore & Álvaro Vázquez-Mayagoitia. DOI: 10.1038/s41524-020-00367-7. [Dataset url](https://github.com/argonne-lcf/active-learning-md/tree/master/data)
- FitSNAP: A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package. [Datasets url](https://github.com/FitSNAP/FitSNAP/tree/master/examples)
- CESMIX training data repository. [Datasets url](https://github.com/cesmix-mit/TrainingData)


### Fit ACE

The input parameters are listed below:

| Input parameter      | Description                                               | E.g.                      |
|----------------------|-----------------------------------------------------------|---------------------------|
| experiment_path      | Experiment path                                           | a-Hfo2-300K-NVT-6000/     |
| dataset_path         | Dataset path                                              | data/                     |
| dataset_filename     | Dataset filename                                          | a-Hfo2-300K-NVT-6000.exyz |
| random_seed          | Random seed of the current experiment                     | 100                       |
| n_train_sys          | No. of atomic systems in training dataset                 | 800                       |
| n_test_sys           | No. of atomic systems in test dataset                     | 200                       |
| n_body               | Body order                                                | 3                         |
| max_deg              | Maximum polynomial degree                                 | 3                         |
| r0                   | An estimate on the nearest-neighbour distance for scaling | 1.0                       |
| rcutoff              | Outer cutoff radius                                       | 5.0                       |
| wL                   | See run-experiments.jl                                    | 1.0                       |
| csp                  | See run-experiments.jl                                    | 1.0                       |
| w_e                  | Energy weight                                             | 1.0                       |
| w_f                  | Force weight                                              | 1.0                       |


## Installation

### Install Julia on Ubuntu

Open terminal and download Julia from https://julialang.org/downloads/
```shell
    $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.0-linux-x86_64.tar.gz
```
Extract file
```shell
    $ tar xvzf julia-1.8.0-linux-x86_64.tar.gz
```
Copy to `/opt` and create link
```shell
    $ sudo mv  ./julia-1.8.0 /opt/
    $ sudo ln -s /opt/julia-1.8.0/bin/julia /usr/local/bin/julia
```
Alternative: add line to `.shellrc`
```shell
    $ nano .shellrc
    PATH=$PATH:/home/youruser/julia-1.8.0 /bin/
```
Restart the terminal


### Add registries

Open a Julia REPL and add registries: General, CESMIX, and MolSim.
```shell
    $ julia
```
Type `]`, then:
```julia
    pkg> registry add https://github.com/JuliaRegistries/General
    pkg> registry add https://github.com/cesmix-mit/CESMIX.git 
    pkg> registry add https://github.com/JuliaMolSim/MolSim.git
```

### Clone repository, change branch, and install dependencies

Clone repository in your work directory and change branch
```shell
    $ git clone git@github.com:cesmix-mit/PotentialLearning.jl.git
    $ git checkout ace-examples
```
Access to ACE example folder and open a Julia REPL
```shell
    $ cd PotentialLearning.jl/examples/ACE
    $ julia
```
Activate and instantiate
```julia
    pkg> activate .
    pkg> instantiate
```

### Run fitting experiment

```shell
    $ julia fit-ace.jl  experiment_path       a-Hfo2-300K-NVT-6000/ \
                        dataset_path          data/ \
                        dataset_filename      a-Hfo2-300K-NVT-6000.extxyz \
                        random_seed           100 \
                        n_train_sys           800 \
                        n_test_sys            200 \
                        n_body                3 \
                        max_deg               3 \
                        r0                    1.0 \
                        rcutoff               5.0 \
                        wL                    1.0 \
                        csp                   1.0 \
                        w_e                   1.0 \
                        w_f                   1.0
```

In addition, you can run the experiments with default parameters (parameters shown above).
```shell
    $ julia fit-ace.jl
```

Alternatively, you can open the Julia REPL first, and then include `fit-ace.jl`.
```shell
    $ julia
```
```julia
    $ include("fit-ace.jl")
```
