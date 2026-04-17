# Process FNO training for d+Au 3D Collision Study using XSCAPE

## Summary:

  - use the `writer` branch of XSCAPE to run 3D collision simulations
    - use the RootBulkWriter class (see `config/jetscape_main.xml` for all
      options, and `config/jetscape_user_dAu3d.xml` for an example
      configuration)
  - use the flat/bin/flat3d program with a `[flat_file_name].ini` configuration
    file to generate a `[flat_file_name].root` file (the name in the `.ini` file 
    points to the xscape output `.root` file).
  - use the library from https://github.com/neuraloperator/neuraloperator.git
    for the 3+1D FNO model, update the following three files (found under the
    `writer` branch of the XSCAPE respsotory under the `./fno4d` directory.
    More details are found in `./fno4d/readme`:
        ./neuralop/losses/data_losses.py
        ./neuralop/losses/finite_diff.py
        ./neuralop/models/fno.py
    Add this modified library to the python environment
  - use a python script, such as `train/fno_dAu3d/train_model.py` to train
    the FNO model.
     

    output file includes:
    - best_model_state_dict.pt : contains the trained model weights
    - training.log : the training loss information, printed out every other epoch
    
  - use a script in a notebook, such as `train/dAu3d/apply.py` (which is 
    a python file but can be run as a marimo notebook) to test the model against
    new data

## More details:

### In XSCAPE, RootBulkWriter generates a `.root` file. In that file:
    - `TTree t` contains a `vector<float>` for each event. That vector
       contain {rho, vx, vy, xz} values in a 3D grid of {x, y, eta} points,
       with an entire grid for each tau step until freeze-out.
       (note that freezeout is estimated before MUSIC runs, and there are typically
        several time steps with all zero entries at the end of each event)
    -  `TParameter<int>` for `nx`, `ny`, `neta` (grid sizes)
    -  `TParameter<float>` for `dx`, `dy`, `deta`, `dtau` (grid spacing)
                           `x_min`, `y_min`, `eta_min` (grid minimum values)

### `flat/bin/flat3d` program uses the `[name].ini` file and reads
    the `TTree t` (using the TParameter values to determine the grid structure).

    - The `[name].init` file will write a `TTree t` output tree of a fixed-sized
      array of floats. It takes input paremeters for the following. Comments follow
      c-style `//` symbols. Here is an example:

        NT            11 // The number of time steps written to each array entry
        MIN_NT        11 // The minimum number of time steps

        // Note: if MIN_NT < NT, then events will just pad with zeros at the end
        // to reach NT time steps.
        // If a single input event can provide multiple NT arrays, then it will be
        // broken up into multiple input entires. For example, if the input event
        // has 30 time steps, and NT==11, and MIN_NT=8, then 3 events will be generated:
        //  event 1: time steps 0-10
        //  event 2: time steps 11-21
        //  event 3: time steps 22-29 (padded with zeros to out to 33)
        // This assumes that TOFFSET0 and TOFFSET1 (below) are both set to 0
    
        // A random number generator will be used to select the tau offset each
        // output event starts at. For example, with the setting below, then if the 
        // initial time step was selected to be 5, then the above event (with 30 time steps)
        // would have been broken into only two events:
        // event 1: time steps 5-15
        // event 2: time steps 16-26 (padded with zeros out to 37)
        // note that there are not enough time steps left to make a third event

        TOFFSET0      0 // The starting time step for the first array 
                        // will be offset by at least this amount
        TOFFSET1      11 // The starting time step for subsequent arrays
                         // will be offset by at most this amount
        IFILE_NAME   /home/davidstewart/xscape-docker/X-SCAPE/build/dAu_seed5_5k_10cen.root
        IS_TAUEP75    1 // select whether or not to pre-scale each rho value by
                        // rho_scaled = tau * tau^(0.75)
        EVENT_FIRST   0  // first event to process from input file
        EVENT_LAST    -1 // last event to process from input file (-1 means go to final event)

### train_model.py

    An example input script is provided in `./train/dAu3d/train_model.py`

    The example input script is a pure python script, but is is also writen so
    that it can use a marimo notebook feature. In current workflow, that isn't
    necessary, and it is ok to stript out the meta-information that would allow
    it to be run in a notebook. Those features are much more useful when running
    a notebook with plots interactively to see how good the output model it.

### apply.py

    Use a script to o load the trained model and apply it to some new data to see
    how well the model performs. An example is proved in `train/dAu3d/apply.py`.
    The example script is written as a marimo notebook file, and it is most convenient
    if it is run as sucn. To do this, do something like (from inside the directory holding
    the apply.py file):

    `uv run marimo edit --headless --watch apply.py`
    The output should tell which port to open in a web browser to run the notebook.

    Note that by default, the apply.py script will locally look for the `log.txt` file
    which contains some data about the model selection. It also contains a cell to
    read the local `training.log` file and plot the training losses throughout the epochs.

### `.loc_libs/`

    These contains some local python functions that are useful in the plotting.
    In the scripts above, three are used:

    1. read_3d_root.py : how to read the output of the [name.root] output from
       flat3d program. This is used by both model_training.py and apply.py

    2. `skinnycontour.py` used in apply.py, is a cleaned up version of previous
       code) which generates the 2D histogram plots with contours around data
       percentiles

    3. `parse_training_log.py` which plots the training loss from the
       training.log file

### Some notes are marimo notebooks:

  Advantages:
   - marimon notebooks are really just `*.py` files, and can therefore be 
   grepped. They can also be run directly as python scripts, which may be
   helpful when splitting the GPU memory over both processors.
   - uv is quite fast, and the toml file makes archiving and (with marimo)
   re-running the notebooks much easier.
   - grepping the files is much easier, as it avoids the in-place json/jpeg/etc...
   data in `*.ipynb` files.

   Disadvantages:
   - images in marimon notebooks have to be saved locally, instead
   of in the marimo files themselves


- Some instructions on using marimo:

`
   - To start a notebook locally, use:

      uv run marimo edit --headless --watch [file].py

    n.b.: --headless avoid trying to open the browser on the lambda machine
          --watch: optional, will have the browser marimo instance auto-update
                   data from directly editing the text file


    This will return a url like:
    URL: http://localhost:2721?access_token=1-xYTe5Mtr8TQBI00ag9Ww

    On local computer, do (matching the port number 2721 from above):

      ssh -L 9999:localhost:2721  [ssh-sign-in]

    On local computer browser, go to:
     
       http://localhost:9999?access_token=1-xYTe5Mtr8TQBI00ag9Ww

   - To instead run the notebook as a python script locally do:

        uv run python your_script.py

    uv will automatically use the local virtual environment (setup 
    with a `uv init` command somewhere upstream in the file tree).

`


