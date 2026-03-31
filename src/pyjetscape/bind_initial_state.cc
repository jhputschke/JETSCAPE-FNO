/*******************************************************************************
 * bind_initial_state.cc
 *
 * Binds InitialState:
 *   - Grid accessors (GetXMax/Step/Size, GetYMax/Step/Size, GetZMax/Step/Size)
 *   - GetEntropyDensityDistribution() → flat std::vector<double>
 *   - get_entropy_density_numpy() → 2D numpy array (nx, ny) for boost-invariant
 *   - GetNumOfBinaryCollisions() → flat std::vector<double>
 *   - Event-level quantities (GetNpart, GetNcoll, GetTotalEntropy, …)
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "InitialState.h"
#include "JetScapeModuleBase.h"

namespace py = pybind11;
using namespace Jetscape;

void bind_initial_state(py::module_ &m) {

  // InitialState is a C++ abstract-ish base — concrete implementations are
  // registered via RegisterJetScapeModule<T> and created through create_module().
  // Python code only *calls* methods on an InitialState pointer returned by
  // PyFluidDynamics::get_ini_pointer().  No Python trampoline is needed.
  py::class_<InitialState, JetScapeModuleBase,
             std::shared_ptr<InitialState>>(m, "InitialState",
      R"pbdoc(
        Interface for initial-state physics modules (e.g. TrentoInitial).

        Provides the entropy/binary-collision density grids that seed
        pre-equilibrium and hydro evolution.
      )pbdoc")
      // ── Grid range accessors ───────────────────────────────────────────────
      .def("GetXMax",  &InitialState::GetXMax,
           "Maximum |x| of the nuclear profile grid [fm].")
      .def("GetXStep", &InitialState::GetXStep,
           "Step size dx [fm].")
      .def("GetYMax",  &InitialState::GetYMax,
           "Maximum |y| of the nuclear profile grid [fm].")
      .def("GetYStep", &InitialState::GetYStep,
           "Step size dy [fm].")
      .def("GetZMax",  &InitialState::GetZMax,
           "Maximum |z or eta| of the nuclear profile grid.")
      .def("GetZStep", &InitialState::GetZStep,
           "Step size dz (or d-eta) in z direction.")
      .def("GetXSize", &InitialState::GetXSize,
           "Number of grid points in x: ceil(2*xmax/dx).")
      .def("GetYSize", &InitialState::GetYSize,
           "Number of grid points in y: ceil(2*ymax/dy).")
      .def("GetZSize", &InitialState::GetZSize,
           "Number of grid points in z (1 for boost-invariant).")
      // ── Density distributions ──────────────────────────────────────────────
      .def("GetEntropyDensityDistribution",
           &InitialState::GetEntropyDensityDistribution,
           R"pbdoc(
             Return the entropy density distribution as a flat list.
             Storage order: for z { for y { for x } }.
             Length = GetXSize() * GetYSize() * GetZSize().
           )pbdoc")
      .def("GetNumOfBinaryCollisions",
           &InitialState::GetNumOfBinaryCollisions,
           R"pbdoc(
             Return the un-normalised T_A*T_B binary-collision density as a flat list.
             Same storage order as GetEntropyDensityDistribution().
           )pbdoc")
      // ── Convenience: 2D numpy view (boost-invariant case) ─────────────────
      .def("get_entropy_density_numpy",
           [](InitialState &ini) -> py::array_t<double> {
             const auto &dist = ini.GetEntropyDensityDistribution();
             int nx = ini.GetXSize();
             int ny = ini.GetYSize();
             if ((int)dist.size() < nx * ny)
               throw std::runtime_error(
                   "Entropy density distribution has fewer entries than nx*ny. "
                   "Make sure Init()+Exec() have been called on the module.");
             // Storage order is for y { for x }, i.e. row = y-index.
             // Return shape (ny, nx) so arr[iy, ix] == dist[iy*nx + ix].
             py::array_t<double> arr({ny, nx});
             auto r = arr.mutable_unchecked<2>();
             for (int iy = 0; iy < ny; ++iy)
               for (int ix = 0; ix < nx; ++ix)
                 r(iy, ix) = dist[iy * nx + ix];
             return arr;
           },
           R"pbdoc(
             Return the entropy density as a 2D numpy array of shape (ny, nx).
             Only valid for the boost-invariant (2D) case (GetZSize() == 1).
             Equivalent to:
               dist = GetEntropyDensityDistribution()
               arr  = np.array(dist).reshape(ny, nx)
           )pbdoc")
      // ── Event-level scalars ────────────────────────────────────────────────
      .def("GetNpart",           &InitialState::GetNpart,
           "Number of participant nucleons (-1 if not available).")
      .def("GetNcoll",           &InitialState::GetNcoll,
           "Number of binary collisions (-1 if not available).")
      .def("GetTotalEntropy",    &InitialState::GetTotalEntropy,
           "Total entropy of the event (-1 if not available).")
      .def("GetEventCentrality", &InitialState::GetEventCentrality,
           "Event centrality class (-1 if not available).")
      .def("GetEventId",         &InitialState::GetEventId,
           "Current event ID.")
      .def("SetEventId",         &InitialState::SetEventId,
           py::arg("event_id"))
      // ── Geometry setters (useful for testing without XML) ─────────────────
      .def("SetRanges", &InitialState::SetRanges,
           "Set grid extents (xmax, ymax, zmax).",
           py::arg("xmax"), py::arg("ymax"), py::arg("zmax"))
      .def("SetSteps", &InitialState::SetSteps,
           "Set grid step sizes (dx, dy, dz).",
           py::arg("dx"), py::arg("dy"), py::arg("dz"));
}
