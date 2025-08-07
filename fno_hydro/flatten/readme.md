Short C++ code that read through an input ROOT file with a TTree named `t` with
a branch named `user_res` of data type
`vector<vector<vector<vector<float>,float>,float>,float>` and writes a new
TFile with a TTree and TBranch with the same names as the input TFile, but with
the data in the TBranch flattened into a single fixed-sized array of floats.

Input parameters:
   1. `<name_stemp>` : the name of input ROOT file (without the `.root` extension,
      which the program automatically adds internally)

   2. last event. Defaults to -1, in which case it will take all events in the
      input file.

   3. `<nsteps_tau>` : number of tau steps to put into output. Defaults to 60.

   4. `freeze_only` : a boolean to see if only the tau lenghts of the events
      should be kept.

Output:

   Generates an output file name of name `<name_stem>_flat_xy<grid_size>_t<nsteps_tau>.root`
   The grid size is read from the size of the input TTree vector size, `<nsteps_tau>`is
   an input parameter.

   The flat output vector is such that it will reshape appropriately with a numpy command:
   `np.reshape( <array variable>, (nevents, n_parameters, n_x, n_y, n_tau))`

Use:
    - Compile with `make` requires that environment tool `root-config` be
      locally set

    - Use the executable `bin/flat` with the name-stem of the input ROOT file.

