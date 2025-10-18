# Motivation:
  
  See `../README.md`

# Compilation:

Ensure that ROOT is configured on the system so that root-config works. Then
type make.

# Usage:

./bin/flat [INPUT_FILE] [LAST_EVENT] [OUTPUT_FILE] [TIME_STEPS]

DESCRIPTION:
    Flattens 4D hydrodynamic data from ROOT files into a more convenient format.
    Converts `vector<vector<vector<vector<float>>>>` data structure to flat arrays
    (i.e. float[nFeatures*nX*nY*nT]), organized as [nFeatures, nX, nY, nT], so
    that in numpy they can be converted back into a tensor like:
    `reshaped_tensor = np.reshape(array, (nFeatures, nX, nX, nT))`

ARGUMENTS:
    INPUT_FILE     Input ROOT file stem (without .root extension)
                   Default: "jetscape_main"
                   Program will read from INPUT_FILE.root

    LAST_EVENT     Maximum event number to process (-1 for all events)
                   Default: -1 (process all events)
                   Use positive integer to limit processing

    OUTPUT_FILE    Output ROOT file name
                   Default: auto-generated as INPUT_FILE_flat_xyNXY_tNT.root
                   where NXY is grid size and NT is number of time steps

    TIME_STEPS     Number of time steps to include in output
                   Default: 60