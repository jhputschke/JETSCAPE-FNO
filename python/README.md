# JETSCAPE-FNO Python Interface

A `pybind11`-based Python extension (`pyjetscape_core`) that exposes the JETSCAPE
C++ framework to Python.  It provides:

- Full simulation control (`JetScape`, `Init`, `Exec`, `Finish`) from Python
- Module instantiation via the C++ factory (`create_module("TrentoInitial")`)
- Zero-copy numpy views of initial-state and pre-equilibrium data
- A `FluidDynamics` trampoline base class so a PyTorch FNO model written in
  Python can operate as a first-class JETSCAPE hydro module and feed results
  back into `EvolutionHistory`

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | ≥ 3.8 | 3.12 used in the `js_fno` conda env |
| PyTorch | ≥ 2.0 | Required for `PyFNOHydro`; `pyjetscape_core` itself is pure C++ |
| numpy | ≥ 1.21 | |
| pybind11 | ≥ 2.11 | Fetched automatically by CMake |
| CMake | ≥ 3.18 | FetchContent support |

The pre-built `.so` in `build_conda/` was compiled against the `js_fno` conda
environment.  If you use a different Python installation, rebuild as shown below.

---

## Build

```bash
# Activate the conda environment (or any Python ≥ 3.8 environment)
conda activate js_fno

cd JETSCAPE-FNO
mkdir -p build_conda && cd build_conda

cmake .. \
  -DUSE_PYTHON=ON \
  -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

make -j$(nproc) pyjetscape_core
```

The shared library is placed in `build_conda/src/pyjetscape/` and a copy is
installed to `python/jetscape/pyjetscape_core.<platform>.so`.

### Making the package importable

```bash
# From the repo root — add python/ to PYTHONPATH
export PYTHONPATH="$PWD/python:$PYTHONPATH"
# or, permanently, add it to your conda env
conda develop python/   # requires conda-build
```

Verify the build:

```bash
python -c "from jetscape import JetScape; js = JetScape(); print('pyjetscape OK')"
```

---

## Package layout

```
python/jetscape/
├── __init__.py            # Re-exports from pyjetscape_core
├── pyjetscape_core.*.so   # Compiled pybind11 extension
├── fno_hydro.py           # PyFNOHydro — Python FluidDynamics trampoline
├── utils.py               # numpy / torch conversion utilities
└── run_jetscape.py        # High-level driver helpers (run_automatic, run_manual)
```

---

## Usage

### Mode A — All-C++ pipeline driven from Python

All modules are read from XML and instantiated by the C++ factory.  Python is
the driver only; no Python modules can be injected.  Suitable for running
standard JETSCAPE configurations and post-processing their output in Python.

```python
from jetscape.run_jetscape import run_automatic

js = run_automatic(
    main_xml="config/jetscape_main.xml",
    user_xml="config/jetscape_user_MUSIC.xml",
    # XML must contain:
    # <enableAutomaticTaskListDetermination>true</enableAutomaticTaskListDetermination>
)
```

### Mode B — Manual pipeline with a Python FNO hydro module

Modules are supplied explicitly in pipeline order.  The only mode that supports
`PyFNOHydro` or any other Python trampoline module.

All C++ modules are obtained via `create_module(name)` — no individual Python
class bindings are needed for them.  The `xml` file must still carry all
`<Module>` parameter blocks so `InitializeHydro()` can read XML values, but
`<enableAutomaticTaskListDetermination>` must be `false`.

```python
from jetscape import create_module
from jetscape.fno_hydro import PyFNOHydro
from jetscape.run_jetscape import run_manual

# ── Initial state & pre-equilibrium (added flat to JetScape) ──────────────────
ini   = create_module("TrentoInitial")
preeq = create_module("FreestreamMilne")   # or "NullPreDynamics"

# ── Hydro: Python FNO trampoline ──────────────────────────────────────────────
config = dict(
    nx=60, ny=60, ntau=50, neta=1, n_features=4,
    x_min=-7.5, y_min=-7.5,
    dtau=0.05, deta=0.0,
    T_freeze=0.150,      # GeV
    device="cpu",
)
fno = PyFNOHydro("build_conda/fno_hydro/models/traced_model.pt", config)

# ── Jet energy loss: two-level hierarchy ─────────────────────────────────────
jloss_mgr = create_module("JetEnergyLossManager")
jloss     = create_module("JetEnergyLoss")
matter    = create_module("Matter")
# Note: if using Matter+Martini, Matter MUST come first
jloss.Add(matter)
jloss_mgr.Add(jloss)

# ── Hadronization: same two-level hierarchy ───────────────────────────────────
hadro_mgr = create_module("HadronizationManager")
hadro     = create_module("Hadronization")
colorless = create_module("ColorlessHadronization")
hadro.Add(colorless)
hadro_mgr.Add(hadro)

# ── Run — run_manual validates pipeline order before calling js.Add() ─────────
js = run_manual(
    main_xml="config/jetscape_main.xml",
    user_xml="config/jetscape_user_fno_python.xml",
    modules=[ini, preeq, fno, jloss_mgr, hadro_mgr],
)
```

---

## Model loading — three approaches

`PyFNOHydro` selects the loading strategy from the type of the `model` argument:

```python
# 1. JIT-traced .pt  (compatible with C++ FnoHydro traced models)
fno = PyFNOHydro("models/traced_JS3.7.fno_model_cpu.pt", config)

# 2. Checkpoint + Python class definition
from my_fno_package import FNOModel
net = FNOModel(modes=12, width=32)
fno = PyFNOHydro((net, "checkpoints/epoch_100.pt"), config)

# 3. Live Python model — no file I/O, ideal for research iteration
net = FNOModel(modes=12, width=32)
net.load_state_dict(torch.load("checkpoints/epoch_100.pt"))
fno = PyFNOHydro(net, config)
```

All three call `self._model(input_tensor)` identically during `EvolveHydro()`.
A live model (approach 3) can also be JIT-traced at runtime before the event
loop with `torch.jit.trace(net, example_input)` for improved inference speed.

---

## Accessing physics data as numpy / torch tensors

### Initial state entropy density

```python
ini  = create_module("TrentoInitial")
# after js.Init() has been called:
ed_2d = ini.get_entropy_density_numpy()   # shape (nx, ny), dtype float64
```

### Pre-equilibrium stress-energy fields

All fields are zero-copy numpy views into the C++ `std::vector<double>` memory:

```python
preeq = create_module("FreestreamMilne")
# after js.Exec() step:
e   = preeq.get_e_numpy()      # energy density,  shape (nx*ny,)
ux  = preeq.get_ux_numpy()     # flow velocity x, shape (nx*ny,)
uy  = preeq.get_uy_numpy()     # flow velocity y, shape (nx*ny,)
# also: get_P_numpy, get_ueta_numpy, get_pi00_numpy … get_pi33_numpy, get_bulk_Pi_numpy
```

### Hydro evolution history

```python
from jetscape.utils import bulk_info_to_numpy, bulk_info_to_tensor

bulk  = fno.get_bulk_info()

# numpy array — shape (ntau, nx, ny, n_features)
arr   = bulk_info_to_numpy(bulk, n_features=4)

# torch tensor — shape (1, n_features, nx, ny, ntau)  ← FNO convention
t     = bulk_info_to_tensor(bulk, n_features=4, device="cpu")
```

Feature index mapping (n_features = 4):

| Index | Quantity |
|-------|----------|
| 0 | energy density [GeV/fm³] |
| 1 | temperature [GeV] |
| 2 | v_x |
| 3 | v_y |

### Writing a numpy / tensor result back into C++

```python
from jetscape.utils import numpy_to_bulk_info, tensor_to_bulk_info

# from numpy  (n_features, nx, ny, ntau)
numpy_to_bulk_info(arr, fno, n_features=4)

# from torch tensor  (1, n_features, nx, ny, ntau)
tensor_to_bulk_info(t, fno, n_features=4)
```

---

## Writing a custom Python hydro module

Subclass `FluidDynamics` from `pyjetscape_core` and override
`InitializeHydro` and `EvolveHydro`:

```python
from jetscape import FluidDynamics
from jetscape.utils import rebin_preeq_to_fno_grid, tensor_to_bulk_info
import torch

class MyHydro(FluidDynamics):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.SetId("MyHydro")

    def InitializeHydro(self, params):
        # Called once during js.Init()
        # self.config holds grid parameters; XML values accessible via
        #   self.GetXMLElementDouble(["Hydro", "MyHydro", "dtau"])
        self.nx   = self.config["nx"]
        self.ntau = self.config["ntau"]

    def EvolveHydro(self):
        # Called once per event during js.Exec()
        preeq = self.get_preeq_pointer()
        ini   = self.get_ini_pointer()

        # Access pre-eq grid
        e_flat = preeq.get_e_numpy()        # (nx_preq * ny_preq,)
        ux_flat = preeq.get_ux_numpy()

        # Build, run, write back ...
        # self.SetHydroGridInfo_from_python(...)
        # tensor_to_bulk_info(output, self)
        # self.FindAConstantTemperatureSurface(T_freeze, ...)
```

Add the module to the pipeline in Mode B the same way as `PyFNOHydro`.

---

## Examples

| Script | Description |
|--------|-------------|
| `python examples/python_fno_test.py` | Mode B pipeline test: `TrentoInitial → FreestreamMilne → PyFNOHydro → Matter → Hadronization`. Loads the real JIT-traced model, runs one event, and prints `bulk_info` shape and freeze-out surface statistics. Pass `--help` for all flags. |
| `python examples/inspect_bulk_info.py` | Run a simulation (or load a saved `.npy`) and produce matplotlib plots of energy density slices, temperature evolution, velocity field quiver, and tensor feature statistics. Outputs are saved to `inspect_bulk_info_out/`. Pass `--help` for all flags. |
| `python examples/convert_binary_to_ASCII_output.py` | Convert JETSCAPE binary hadron output to ASCII (Mode A) |

---

## Troubleshooting

**`ImportError: cannot import name 'JetScape' from 'jetscape'`**  
The `.so` is not on `PYTHONPATH`.  Run `export PYTHONPATH=/path/to/JETSCAPE-FNO/python:$PYTHONPATH`.

**`RuntimeError: error loading the model`**  
Check that the `.pt` file was traced with the same PyTorch version.  JIT-traced
models are not always forward-compatible.  Use approach 2 or 3 instead.

**`ValueError: ... is out of pipeline order`**  
`run_manual()` detected that modules were supplied in the wrong order.  The
required order is `InitialState → PreequilibriumDynamics → FluidDynamics →
JetEnergyLossManager → HadronizationManager`.

**Segfault in `EvolveHydro()`**  
The most common cause is accessing `get_preeq_pointer()` or `get_ini_pointer()`
before `js.Init()` has been called, or returning from `EvolveHydro()` without
calling `self.SetHydroGridInfo_from_python(...)` and populating `bulk_info`.
