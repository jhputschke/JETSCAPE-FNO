# Plan: JETSCAPE-FNO Python Interface with FNO Feedback Loop

**TL;DR** — Add a `pybind11`-based Python extension module (`pyjetscape`) to JETSCAPE-FNO. The module exposes all JETSCAPE C++ infrastructure to Python, provides zero-copy numpy/torch views of physics data, and enables a Python `FluidDynamics` **trampoline class** so a PyTorch FNO model running in Python can act as a native JETSCAPE hydro module — receiving initial conditions directly from C++ and writing results back into the framework's C++ `EvolutionHistory`.

---

## Phase 1 — Build System Integration

1. Add `pybind11` via CMake `FetchContent` in `external_packages/CMakeLists.txt`
2. Add `-DUSE_PYTHON=ON` flag to root `CMakeLists.txt`
3. Under that flag: define `pybind11_add_module(pyjetscape ...)` linking against `libJetScape.so` and libtorch
4. Install output to `python/jetscape/pyjetscape_core.so`

**Files:** `CMakeLists.txt`, `external_packages/CMakeLists.txt`

---

## Phase 2 — pybind11 C++ Binding Sources

Create `src/pyjetscape/` with four binding files, all compiled into the single `pyjetscape` extension:

| File | Binds |
|------|-------|
| `bind_framework.cc` | `JetScape` (SetXML, Init, Exec, Finish, Add) |
| `bind_evolution.cc` | `FluidCellInfo`, `EvolutionHistory` (numpy buffer protocol for `data_vector`, grid metadata), `SurfaceCellInfo` |
| `bind_initial_state.cc` | `InitialState` — `get_entropy_density_numpy()` as 2D numpy array; `GetXSize/YSize/XStep/XMax/YMax` |
| `bind_fluid_dynamics.cc` | `PreequilibriumDynamics` — zero-copy numpy views for all public fields (`e_`, `P_`, `ux_`, `uy_`, `ueta_`, `pi00_`–`pi33_`, `bulk_Pi_`); `FluidDynamics` + **Python trampoline** `PyFluidDynamics` using `PYBIND11_OVERRIDE_PURE` for `InitializeHydro()` and `EvolveHydro()`; exposes `get_bulk_info()`, `get_ini_pointer()`, `get_preeq_pointer()`, `set_bulk_info_from_numpy()`, `SetHydroGridInfo_from_python()`, `FindAConstantTemperatureSurface()` |

The trampoline is the critical piece: it lets a Python class inherit from `pyjetscape.FluidDynamics` and be registered with `JetScape.Add()`, making the Python FNO model a first-class C++ module.

**New files:**
- `src/pyjetscape/pyjetscape_core.cc`
- `src/pyjetscape/bind_framework.cc`
- `src/pyjetscape/bind_evolution.cc`
- `src/pyjetscape/bind_initial_state.cc`
- `src/pyjetscape/bind_fluid_dynamics.cc`
- `src/pyjetscape/CMakeLists.txt`

---

## Phase 3 — Data Conversion Utilities

New `python/jetscape/utils.py`:

```python
bulk_info_to_numpy(bulk_info)        # → np.ndarray shape (ntau, nx, ny, n_fields)
bulk_info_to_tensor(bulk_info)       # → torch.Tensor shape (1, n_fields, nx, ny, ntau)  ← FNO convention
numpy_to_bulk_info(arr, bulk_info)   # writes (ntau, nx, ny, n_fields) back into data_vector
tensor_to_bulk_info(tensor, bulk_info)
```

Zero-copy where possible: `EvolutionHistory::data_vector` (a `vector<float>`) is exposed via pybind11's buffer protocol so numpy arrays point directly into C++ memory for reading. Writes go back via the setter.

---

## Phase 4 — `PyFNOHydro` Python Class

New `python/jetscape/fno_hydro.py`:

```python
class PyFNOHydro(pyjetscape.FluidDynamics):
    def __init__(self, model, config):
        """
        model: one of
          - str path          → torch.jit.load()  (JIT-traced, compatible with C++ FnoHydro models)
          - (nn.Module, path) → torch.load() + load_state_dict()  (checkpoint + Python class)
          - nn.Module         → used as-is (live Python model, no serialization needed)
        """
        super().__init__()
        if isinstance(model, str):
            self.model = torch.jit.load(model); self.model.eval()
        elif isinstance(model, tuple):
            net, ckpt = model
            state = torch.load(ckpt, map_location="cpu")
            net.load_state_dict(state.get("model_state_dict", state)); net.eval()
            self.model = net
        elif isinstance(model, torch.nn.Module):
            self.model = model; self.model.eval()
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
        self.config = config
        self.SetId("PyFNOHydro")

    def InitializeHydro(self, params):
        # read grid config from self.config dict: nx, ny, ntau, n_features, dx, dtau, x_min, T_freeze
        ...

    def EvolveHydro(self):
        preeq = self.get_preeq_pointer()
        # 1. Access full pre-eq stress-energy from public fields — all available as numpy:
        #      preeq.get_e_numpy()   → energy density  (nx_preq * ny_preq,)
        #      preeq.get_ux_numpy(), preeq.get_uy_numpy()  → flow velocities
        #      preeq.get_pi00_numpy() ... → viscous tensor components
        # 2. Rebin to FNO grid, apply edensity*tau normalization (matches FnoHydro.cc)
        #    → input tensor shape (1, n_feat, nx, ny, ntau)
        # 3. with torch.no_grad(): output = self.model(input_tensor)
        # 4. self.SetHydroGridInfo_from_python(...)
        # 5. tensor_to_bulk_info(output, self.get_bulk_info())
        # 6. self.FindAConstantTemperatureSurface(T_freeze, ...)
        ...
```

The rebinning step replicates the normalization logic from `FnoHydro::EvolveHydro()` (energy density × tau normalization, currently hardcoded in `FnoHydro.cc`). All three model-loading approaches call `self.model(input_tensor)` identically in step 3.

---

## Phase 5 — Python Simulation Driver

JETSCAPE supports two task-list modes controlled by `<enableAutomaticTaskListDetermination>` in the user XML. The Python driver must handle both.

### Mode A — Automatic task list (`enableAutomaticTaskListDetermination = true`)

JetScape reads the XML and instantiates all modules via the C++ factory/registry (`RegisterJetScapeModule<T>`). A Python-defined `PyFNOHydro` cannot be registered into that C++ factory, so **this mode is incompatible with Python trampoline modules**. It remains fully usable from Python for running all-C++ pipelines (e.g. inspecting output from a standard MUSIC run):

```python
# All-C++ pipeline — Python is only the driver, no Python modules injected
js = pyjetscape.JetScape()
js.SetXMLMainFileName("config/jetscape_main.xml")
js.SetXMLUserFileName("config/jetscape_user_MUSIC.xml")
# XML must contain <enableAutomaticTaskListDetermination>true</enableAutomaticTaskListDetermination>
# and list all module names (Trento, FreestreamMilne, FnoHydro, etc.)

js.Init()
js.Exec()
js.Finish()
```

### Mode B — Manual task list (`enableAutomaticTaskListDetermination = false`)

Modules are added explicitly in the **correct pipeline order** — the same order JetScape would resolve from the XML in automatic mode. This is the only mode that supports injecting Python trampoline modules such as `PyFNOHydro`.

**Mode B does NOT require individual Python class bindings for every C++ module.** The framework already provides `JetScapeModuleFactory::createInstance(string name)` which returns a `shared_ptr<JetScapeModuleBase>` for any C++ module already registered via `RegisterJetScapeModule<T>`. Binding that single factory function in `bind_framework.cc` is sufficient to instantiate all existing C++ modules from Python by name:

```python
# Mixed pipeline — C++ modules via factory, Python trampoline for FNO only
js = pyjetscape.JetScape()
js.SetXMLMainFileName("config/jetscape_main.xml")
js.SetXMLUserFileName("config/jetscape_user_fno_python.xml")
# XML must contain <enableAutomaticTaskListDetermination>false</enableAutomaticTaskListDetermination>
# and still carry all <Module> parameter blocks so InitializeHydro() can read XML values

# --- Initial state + pre-equilibrium (flat, added directly to js) ---
ini   = pyjetscape.create_module("TrentoInitial")
preeq = pyjetscape.create_module("FreestreamMilne")  # or "NullPreDynamics"

# --- Hydro: Python trampoline — not in the C++ factory ---
fno = PyFNOHydro("models/traced.pt", config)

# --- Jet energy loss: two-level hierarchy ---
# Individual algorithms are added to a JetEnergyLoss container,
# which is then added to a JetEnergyLossManager, which is added to js.
jloss_mgr  = pyjetscape.create_module("JetEnergyLossManager")
jloss      = pyjetscape.create_module("JetEnergyLoss")
matter     = pyjetscape.create_module("Matter")
# martini  = pyjetscape.create_module("Martini")  # Note: Matter MUST come first
jloss.Add(matter)
jloss_mgr.Add(jloss)

# --- Hadronization: same two-level hierarchy ---
hadro_mgr  = pyjetscape.create_module("HadronizationManager")
hadro      = pyjetscape.create_module("Hadronization")
colorless  = pyjetscape.create_module("ColorlessHadronization")
hadro.Add(colorless)
hadro_mgr.Add(hadro)

# --- Assemble pipeline in stage order ---
js.Add(ini)
js.Add(preeq)
js.Add(fno)
js.Add(jloss_mgr)
js.Add(hadro_mgr)

js.Init()
js.Exec()
js.Finish()
```

Individual Python class bindings (Phase 2 `bind_initial_state.cc`, `bind_fluid_dynamics.cc`, etc.) are needed **only** when Python code must call module-specific methods to access or write data — not for mere instantiation and execution. The factory covers the rest.

### `run_jetscape.py` helpers

New `python/jetscape/run_jetscape.py` provides two purposeful helpers — one per mode.

**Mode A** needs no module list; the helper is just a clean one-liner entry point:

```python
def run_automatic(main_xml, user_xml):
    """Run a fully-XML-driven pipeline (enableAutomaticTaskListDetermination=true)."""
    js = pyjetscape.JetScape()
    js.SetXMLMainFileName(main_xml)
    js.SetXMLUserFileName(user_xml)
    js.Init()
    js.Exec()
    js.Finish()
    return js
```

**Mode B** adds real value only if `run_manual` validates that the supplied modules are in a legal pipeline order before handing them to `js.Add()`, catching mistakes early rather than letting `JetScapeSignalManager` fail silently at runtime:

```python
# Expected stage order uses the *manager* types for the two-level subsystems.
# JetEnergyLoss and Hadronization algorithms are nested inside their managers
# and are never added directly to JetScape — so the validation only sees managers.
_STAGE_ORDER = [
    pyjetscape.InitialState,
    pyjetscape.PreequilibriumDynamics,
    pyjetscape.FluidDynamics,
    pyjetscape.JetEnergyLossManager,   # optional; contains JetEnergyLoss → [Matter, Martini, ...]
    pyjetscape.HadronizationManager,   # optional; contains Hadronization → [ColorlessHadronization, ...]
    pyjetscape.SoftParticlization,     # optional
]

def run_manual(main_xml, user_xml, modules):
    """
    Run with explicit module list (enableAutomaticTaskListDetermination=false).
    `modules` must be the top-level objects added directly to JetScape — i.e.
    JetEnergyLossManager and HadronizationManager, not the nested algorithms.
    Validates stage order before adding to the task list.
    """
    _validate_pipeline_order(modules, _STAGE_ORDER)
    js = pyjetscape.JetScape()
    js.SetXMLMainFileName(main_xml)
    js.SetXMLUserFileName(user_xml)
    for mod in modules:
        js.Add(mod)
    js.Init()
    js.Exec()
    js.Finish()
    return js

def _validate_pipeline_order(modules, stage_order):
    last_stage = -1
    for mod in modules:
        for i, stage_type in enumerate(stage_order):
            if isinstance(mod, stage_type):
                if i < last_stage:
                    raise ValueError(
                        f"{type(mod).__name__} is out of pipeline order — "
                        f"must come after stage index {last_stage}"
                    )
                last_stage = i
                break
```

---

## Relevant Files

### To modify
- [JETSCAPE-FNO/CMakeLists.txt](JETSCAPE-FNO/CMakeLists.txt) — add `USE_PYTHON` flag + pybind11 target
- [JETSCAPE-FNO/external_packages/CMakeLists.txt](JETSCAPE-FNO/external_packages/CMakeLists.txt) — pybind11 FetchContent

### Reference (read, do not modify)
- [JETSCAPE-FNO/src/framework/FluidDynamics.h](JETSCAPE-FNO/src/framework/FluidDynamics.h) — trampoline base; `bulk_info`, `surfaceCellVector_`
- [JETSCAPE-FNO/src/framework/FluidEvolutionHistory.h](JETSCAPE-FNO/src/framework/FluidEvolutionHistory.h) — `data_vector`, grid metadata
- [JETSCAPE-FNO/src/framework/InitialState.h](JETSCAPE-FNO/src/framework/InitialState.h) — `entropy_density_distribution_`
- [JETSCAPE-FNO/src/framework/PreequilibriumDynamics.h](JETSCAPE-FNO/src/framework/PreequilibriumDynamics.h) — `e_[]` array (confirm field name + visibility)
- [JETSCAPE-FNO/fno_hydro/fno_module/FnoHydro.cc](JETSCAPE-FNO/fno_hydro/fno_module/FnoHydro.cc) — normalization / rebinning reference

### To create
- `JETSCAPE-FNO/src/pyjetscape/pyjetscape_core.cc`
- `JETSCAPE-FNO/src/pyjetscape/bind_framework.cc`
- `JETSCAPE-FNO/src/pyjetscape/bind_evolution.cc`
- `JETSCAPE-FNO/src/pyjetscape/bind_initial_state.cc`
- `JETSCAPE-FNO/src/pyjetscape/bind_fluid_dynamics.cc`
- `JETSCAPE-FNO/src/pyjetscape/CMakeLists.txt`
- `JETSCAPE-FNO/python/jetscape/__init__.py`
- `JETSCAPE-FNO/python/jetscape/fno_hydro.py`
- `JETSCAPE-FNO/python/jetscape/utils.py`
- `JETSCAPE-FNO/python/jetscape/run_jetscape.py`

---

## Verification Steps

1. `cmake -DUSE_PYTHON=ON .. && make pyjetscape` — builds without error
2. `python -c "import pyjetscape; js = pyjetscape.JetScape(); print(js)"` — works
3. `python examples/python_fno_test.py` — brick test replacing C++ `FnoHydro` with `PyFNOHydro`
4. Compare `bulk_info.numpy()` shape `(ntau+1, nx, ny, n_features)` from Python path vs C++ `FnoHydro` for same event — values must match within floating-point tolerance
5. Verify freeze-out surface is populated: `len(surface_cells) > 0`
6. Profile GIL release: `EvolveHydro()` call overhead must not significantly exceed C++ path

---

## Decisions & Scope

- **In scope**: JetScape task control, `JetScapeModuleFactory` binding (`create_module()`), `InitialState`, `PreequilibriumDynamics` (with full numpy views of all public stress-energy fields), `FluidDynamics` trampoline + `PyFNOHydro`; numpy/torch conversion utils; CMake integration
- **Out of scope (later phase)**: Individual class bindings for `JetEnergyLoss`, `Matter`, `LBT`, `Hadronization`, `Afterburner` — these are covered by `create_module(name)` for Mode B and automatic XML-driven instantiation for Mode A
- **pybind11 source**: FetchContent recommended over conda for build reproducibility
- **GIL**: Released during `EvolveHydro()` for model inference via `py::gil_scoped_release`
- **Model format**: Three approaches supported via constructor duck-typing — JIT-traced `.pt` (`str`), checkpoint + Python class (`(nn.Module, path)` tuple), live `nn.Module` directly; all three use identical `self.model(input)` inference call

---

## Open Questions

1. ~~**Pre-equilibrium `e_[]` visibility**~~ — **Resolved.** `e_` and the full stress-tensor set (`P_`, `ux_`, `uy_`, `ueta_`, `pi00_`–`pi33_`, `bulk_Pi_`) are already **public** members of `PreequilibriumDynamics` (confirmed in `PreequilibriumDynamics.h` lines 83–99; used directly by `MusicWrapper.cc` and `CLViscWrapper.cc`). No accessor or `friend` declaration is needed. The pybind11 binding in `bind_fluid_dynamics.cc` can expose all of them as numpy buffer views directly:
   ```cpp
   py::class_<PreequilibriumDynamics, ...>(m, "PreequilibriumDynamics")
     .def("get_e_numpy",    [](PreequilibriumDynamics& p) {
         return py::array_t<double>({(py::ssize_t)p.e_.size()}, p.e_.data()); })
     .def("get_ux_numpy",   [](PreequilibriumDynamics& p) {
         return py::array_t<double>({(py::ssize_t)p.ux_.size()}, p.ux_.data()); })
     .def("get_uy_numpy",   [](PreequilibriumDynamics& p) {
         return py::array_t<double>({(py::ssize_t)p.uy_.size()}, p.uy_.data()); })
     .def("get_pi00_numpy", [](PreequilibriumDynamics& p) {
         return py::array_t<double>({(py::ssize_t)p.pi00_.size()}, p.pi00_.data()); })
     // ... same pattern for pi01_–pi33_, bulk_Pi_
   ```
2. ~~**Model format preference**~~ — **Resolved.** `PyFNOHydro` will support all three approaches, selected by what is passed as `model` to the constructor. No config flag needed — duck-typing on the argument is sufficient:

   | Approach | How to use | When to use |
   |----------|-----------|-------------|
   | **JIT-traced `.pt`** (`torch.jit.load`) | Pass a `str` path | Compatible with existing C++ `FnoHydro` traced models; fastest inference; no Python model class required |
   | **Checkpoint `.pt`** (`torch.load` + model instance) | Pass a `(nn.Module, path)` tuple | Full model weights + Python class; requires the class definition to be importable at load time |
   | **Live Python model** (any `nn.Module`) | Pass the instantiated model directly | Model defined and configured entirely in Python; ideal for research/iteration; no serialization step needed |

   `PyFNOHydro.__init__` resolves the approach at construction time:

   ```python
   class PyFNOHydro(pyjetscape.FluidDynamics):
       def __init__(self, model, config):
           super().__init__()
           if isinstance(model, str):
               # Approach 1: JIT-traced .pt — compatible with C++ FnoHydro models
               self.model = torch.jit.load(model)
               self.model.eval()
           elif isinstance(model, tuple):
               # Approach 2: (nn.Module instance, checkpoint path)
               net, ckpt_path = model
               state = torch.load(ckpt_path, map_location="cpu")
               net.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
               net.eval()
               self.model = net
           elif isinstance(model, torch.nn.Module):
               # Approach 3: live Python model — used as-is, no loading
               self.model = model
               self.model.eval()
           else:
               raise TypeError(f"Unsupported model argument type: {type(model)}")
           self.config = config
           self.SetId("PyFNOHydro")
   ```

   `EvolveHydro()` calls `self.model(input_tensor)` identically for all three — the interface is uniform once the model is loaded. Approach 3 also allows tracing at runtime (`torch.jit.trace(self.model, example_input)`) before the inference loop if needed for performance.
3. ~~**Module binding scope**~~ — **Resolved.** Phase 1 is scoped to the **initial state → pre-equilibrium → hydro** path only. `JetEnergyLoss`, `Matter`, `LBT`, `Hadronization`, and downstream modules are instantiated via `pyjetscape.create_module(name)` (the factory binding) and added to their manager hierarchy as shown in the Mode B example — no individual class bindings required for them. Individual bindings for those stages are deferred to a later phase once the hydro trampoline is validated end-to-end.

