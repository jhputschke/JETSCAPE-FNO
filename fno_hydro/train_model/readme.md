# Summary

This directory constains the `.ipynb` notebooks which do the following:

 - main purpose:

   - input: ROOT files with TTree's names `t` with branches named `user_res`
     of data type `float[n-inputs]`

   - output: trained Fourier Nueral Operator (FNO) models to predict QGP hydrodynamic
     flow from the inital fluid configuration. These are in `.pt` files.

 - additional purpose:

   - has `ipynb` which can take an input ROOT file (same formate as above) and 
     an plot the goodness of the energy distribution (see figures in arxived paper)


 - Required libraries:
    - NeuralOperator library. It is installed from `pip install neuraloperator`
    - awkward
    - uproot
    - numpy
    - matplotlib


 - Example: 

   In the local file `./train_model.ipynb`

   Four small input files (500 events each) are read in. These can be
   downloaded from Zenodo at: [file location]

   It reads in the input, scales the energy by tau in each time step. This is
   the physics normalization that approximately cancels the effect that the
   energy in each tau step is cooled by axial expansion at the speed of light,
   which is the same scaling as tau.

   It trains the first time step to the following 59 time steps.

   It saves the output model under `etau/example_fno/best_model_state_dict.pt`,
   in additional to a log of the training parameters and checkpoints saving the
   state dictionary periodically (every fiften epoques by default). After doing
   the training and looking at the log files, it is ok to delete all
   checkpoints which are not desired.

   The last cells of the `ipynb` also plots some of the results on the verify
   data. Given that the verify data was used to train the model, these are not
   entirely independent, so that new files should be used for testing the
   goodness of the model.

