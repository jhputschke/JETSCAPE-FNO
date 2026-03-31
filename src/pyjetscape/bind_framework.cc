/*******************************************************************************
 * bind_framework.cc
 *
 * Binds:
 *   - JetScapeTask  (base of all modules: Add, SetId, GetId)
 *   - JetScapeModuleBase (SetXMLMainFileName, SetXMLUserFileName, Init, Exec)
 *   - JetScape (Init, Exec, Finish, SetNumberOfEvents, SetReuseHydro, …)
 *   - create_module(name) — wraps JetScapeModuleFactory::createInstance
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "JetScape.h"
#include "JetScapeModuleBase.h"
#include "JetScapeTask.h"

namespace py = pybind11;
using namespace Jetscape;

void bind_framework(py::module_ &m) {

  // ── JetScapeTask ────────────────────────────────────────────────────────────
  // Base of every JETSCAPE module; provides Add(), SetId(), GetId().
  py::class_<JetScapeTask, std::shared_ptr<JetScapeTask>>(m, "JetScapeTask")
      .def("Add", &JetScapeTask::Add,
           "Add a sub-module/task to this task's list.",
           py::arg("task"))
      .def("SetId", &JetScapeTask::SetId,
           "Set the string identifier for this task.",
           py::arg("id"))
      .def("GetId", &JetScapeTask::GetId,
           "Return the string identifier of this task.")
      .def("GetNumberOfTasks", &JetScapeTask::GetNumberOfTasks,
           "Return the number of sub-tasks currently registered.")
      .def("ClearTaskList", &JetScapeTask::ClearTaskList,
           "Remove all sub-tasks.")
      .def("GetActive", &JetScapeTask::GetActive,
           "Return whether this task is active.")
      .def("SetActive", &JetScapeTask::SetActive,
           "Enable or disable this task.",
           py::arg("active"));

  // ── JetScapeModuleBase ──────────────────────────────────────────────────────
  // Adds XML configuration accessors on top of JetScapeTask.
  py::class_<JetScapeModuleBase, JetScapeTask,
             std::shared_ptr<JetScapeModuleBase>>(m, "JetScapeModuleBase")
      .def("SetXMLMainFileName", &JetScapeModuleBase::SetXMLMainFileName,
           "Set the main (default) XML configuration file path.",
           py::arg("filename"))
      .def("GetXMLMainFileName", &JetScapeModuleBase::GetXMLMainFileName,
           "Return the main XML configuration file path.")
      .def("SetXMLUserFileName", &JetScapeModuleBase::SetXMLUserFileName,
           "Set the user (override) XML configuration file path.",
           py::arg("filename"))
      .def("GetXMLUserFileName", &JetScapeModuleBase::GetXMLUserFileName,
           "Return the user XML configuration file path.")
      .def("Init", &JetScapeModuleBase::Init,
           "Initialise this module (reads XML, sets up sub-tasks).")
      .def("Exec", &JetScapeModuleBase::Exec,
           "Execute this module.");

  // ── JetScape ────────────────────────────────────────────────────────────────
  // Top-level framework controller.  Drive a simulation with:
  //   js = JetScape()
  //   js.SetXMLMainFileName("config/jetscape_main.xml")
  //   js.SetXMLUserFileName("config/jetscape_user.xml")
  //   js.Init(); js.Exec(); js.Finish()
  py::class_<JetScape, JetScapeModuleBase, std::shared_ptr<JetScape>>(
      m, "JetScape")
      .def(py::init<>())
      .def("SetXMLMainFileName", &JetScape::SetXMLMainFileName,
           py::arg("filename"))
      .def("SetXMLUserFileName", &JetScape::SetXMLUserFileName,
           py::arg("filename"))
      .def("Init", &JetScape::Init,
           "Initialise the full task list (reads XML, wires signals).")
      .def("Exec", &JetScape::Exec,
           "Run all events.")
      .def("Finish", &JetScape::Finish,
           "Finalise output and clean up.")
      .def("SetNumberOfEvents", &JetScape::SetNumberOfEvents,
           "Override the number of events from Python.",
           py::arg("n_events"))
      .def("GetNumberOfEvents", &JetScape::GetNumberOfEvents,
           "Return the configured number of events.")
      .def("SetReuseHydro", &JetScape::SetReuseHydro,
           "Enable or disable hydro reuse across jet events.",
           py::arg("reuse"))
      .def("GetReuseHydro", &JetScape::GetReuseHydro,
           "Return whether hydro reuse is enabled.")
      .def("SetNReuseHydro", &JetScape::SetNReuseHydro,
           "Set how many jet events reuse each hydro event.",
           py::arg("n"))
      .def("GetNReuseHydro", &JetScape::GetNReuseHydro,
           "Return the hydro-reuse count.");

  // ── Module factory ──────────────────────────────────────────────────────────
  // Instantiate any C++ module that is registered via RegisterJetScapeModule<T>.
  // Returns shared_ptr<JetScapeModuleBase> (nullptr if name is not registered).
  //
  // Usage (Mode B manual pipeline):
  //   ini   = pyjetscape.create_module("TrentoInitial")
  //   preeq = pyjetscape.create_module("FreestreamMilne")
  //   jmgr  = pyjetscape.create_module("JetEnergyLossManager")
  m.def("create_module",
        [](const std::string &name) -> std::shared_ptr<JetScapeModuleBase> {
          auto mod = JetScapeModuleFactory::createInstance(name);
          if (!mod)
            throw py::value_error(
                "Module '" + name +
                "' is not registered in JetScapeModuleFactory. "
                "Make sure the corresponding library (e.g. libFnoModule) is "
                "loaded before calling create_module().");
          return mod;
        },
        R"pbdoc(
          Create a registered JETSCAPE C++ module by name.

          Parameters
          ----------
          name : str
              Registered module name, e.g. "TrentoInitial", "FreestreamMilne",
              "FnoHydro", "JetEnergyLossManager", "Matter", …

          Returns
          -------
          JetScapeModuleBase
              A shared_ptr-managed instance of the requested module.

          Raises
          ------
          ValueError
              If the name is not found in the module registry.
        )pbdoc",
        py::arg("name"));
}
