# Introduction

This repository contains a fork of the JETSCAPE framework for a study first
documented in the paper ``Fast prediction of hydrodynamnical evolution in
ultra-relativistic heavy-ion collisions using Fourier Neural Operators''
located at https://doi.org/10.48550/arXiv.2507.23598

This repository contains the code generated for that study, and is intended
to both archive what was done, and also to provide resources for additional
studies. To do so, it contains the following parts:

## `../`:  A fork of JETSCAPE

   Most of the code is fork of the JETSCAPE project (see one directory up). 
   In this fork, additional code has been added for the following purposes:

   - Code to write out the hydrodynamic evolution of QGP using the MUSIC module
     into ROOT files. The data is written into `TTree` objects, one entry
     per event, in the form of:

     `vector<vector<vector<vector<float>,float>,float>,float>`

     Where the indices when called in code are ordered as
     `data[x][y][parameter][tau]`, where x and y are the indices into the x-y
     grid at mid-rapidity, parameter is currently indexed to: {0: energy
     density, 1: x-velocity, 2: y-velocity} at the grid-point, and tau is
     the time evolution step.

     These are the files that are taken to train FNO models. See `FNO training
     ipynb` below.

   - A new module for JETSCAPE that uses C++ compiled machine learning model (a
     trained FNO model from the data above) in place of MUSIC to do the
     QGP flow evolution.

##  `./flatten`: A convenience C++ code

   - Summary: 

     Small C++ code to read in ROOT files with TTree with entries formated as
     `vector<vector<vector<vector<float>,float>,float>,float>` and writes a new
     ROOT file with a TTREE with entries of type `array<float>`

   - Longer (see `./flatten/README.md` for more detail)

     The ROOT files with the QGP flow data are read into `ipynb` notebooks
     for training with the Neural Operator library. They ultimately must be
     read into fixed-size numpy arrays.

     The python `uproot` and `awkward` libraries can read the
     `vector<vector<...` data type, but it is a bit slow when there are many
     events. This executable, generates new TFiles with TTrees with the single
     arrays (i.e. n_parmams x nx x ny x ntau = 3x60x60x60 = 648000 in the
     reference paper). This output, when read into 1D numpy arrays, can be converted
     into a tensor with a line like (see `./train_model/train_model.ipyn` for 
     a full example):

     `np.reshape(array, (nparams, xy_size, xy_size, last_time_step))`

   - Usage: see `./flatten/README.md`

## `./train_model/`

    The `./train_model/train_model.ipynb` ipython notebook takes a ROOT input file 

     This directory contains `ipynb` notebooks to take inputs of ROOT files of
     data flow from MUSIC and, using the `neuraloperator` python library trains
     Fourier Neural Operator machine learning models.

     It also contains notebooks to apply the models to fresh data and visualize
     the goodness of the predictions against the truth data, generating images
     as seen in https://doi.org/10.48550/arXiv.2507.23598

## `./inspect_model`

    Contains the `./app_model.ipynb` notebook. This notebook can read the model 
    weights, load them into a model, read an input `.root` file, make
    predictions from the first tau step to the remaining tau steps, and draw 2D
    plots of energy density distributions with the boundary and values of
    envelopes for percentiles of the total energy for both the truth and the
    model predicted values. These are the 2D plots listed in
    https://doi.org/10.48550/arXiv.2507.23598

    Note that this functionality is also contained at the end of the
    `./train_model/train_model.ipynb`, although there the model is 
    already present in memory from the training.


## `./macros`

    This directory contains C macros to use ROOT to plot jet and bulk parameters
    from input `*.root` files into 1D histograms, similar to those in the
    publication listed on arXiv.
