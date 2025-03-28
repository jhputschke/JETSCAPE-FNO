/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Modular, task-based framework for simulating all aspects of heavy-ion collisions
 *
 * For the list of contributors see AUTHORS.
 *
 * Report issues at https://github.com/JETSCAPE/JETSCAPE/issues
 *
 * or via email to bugs.jetscape@gmail.com
 *
 * Distributed under the GNU General Public License 3.0 (GPLv3 or later).
 * See COPYING for details.
 ******************************************************************************/

#include <stdio.h>
#include <sys/stat.h>
#include <MakeUniqueHelper.h>

#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include "JetScapeLogger.h"
//#include "surfaceCell.h"
#include "FnoHydro.h"

using namespace Jetscape;

// Register the module with the base class
RegisterJetScapeModule<FnoHydro> FnoHydro::reg("FnoHydro");

FnoHydro::FnoHydro() {
  hydro_status = NOT_START;
  freezeout_temperature = 0.0;
  //doCooperFrye = 0;

  //has_source_terms = false;
  SetId("FnoHydro");
  //hydro_source_terms_ptr =
  //    std::shared_ptr<HydroSourceJETSCAPE>(new HydroSourceJETSCAPE());
}

FnoHydro::~FnoHydro() {}

void FnoHydro::InitializeHydro(Parameter parameter_list) {
  JSINFO << "Initialize FnoHydro ...";
  VERBOSE(8);

  /// ------------------------------------------------------------------
  // Dummy test here ...
  //
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    //c10::Device device(c10::DeviceType::CPU);
    //REMARK: Libtorch has its on OMP library so currently we cannot use it with MUSCIC at the same time !!!
    //setenv OMP_NUM_THREADS 1 for testing, single core ...

    JSINFO<<"Loading the traced Pytorch model ../fno_hydro/models/traced_JS3.7.fno_model_cpu.pt";//<<endl;
    module = torch::jit::load("../fno_hydro/models/traced_JS3.7.fno_model_cpu.pt"); //, device);

    /// ------------------------------------------------------------------
  }
  catch (const c10::Error& e) {
    JSWARN << "error loading the model\n";
    exit(-1);
  }

  JSINFO << "--> traced Pytorch model loaded";
  /// ------------------------------------------------------------------

  /*
  freezeout_temperature =
      GetXMLElementDouble({"Hydro", "MUSIC", "freezeout_temperature"});
  if (freezeout_temperature > 0.05) {
    music_hydro_ptr->set_parameter("T_freeze", freezeout_temperature);
  } else {
    JSWARN << "The input freeze-out temperature is too low! T_frez = "
           << freezeout_temperature << " GeV!";
    exit(1);
  }

  music_hydro_ptr->add_hydro_source_terms(hydro_source_terms_ptr);
  */
}

void FnoHydro::EvolveHydro() {
  VERBOSE(8);
  JSINFO << "Initialize density profiles in FnoHydro ...";

  if (pre_eq_ptr == nullptr) {
    JSWARN << "Missing the pre-equilibrium module ...";
    exit(1);
  }

  double dx = ini->GetXStep();
  double dz = ini->GetZStep();
  double z_max = ini->GetZMax();
  int nz = ini->GetZSize();
  double tau0 = pre_eq_ptr->GetPreequilibriumEndTime();
  JSINFO << "hydro initial time set by PreEq module tau0 = " << tau0 << " fm/c";
  JSINFO << "initial density profile dx = " << dx << " fm";

  //SetPreEqGridInfo();

  /*
  has_source_terms = false;
  if (hydro_source_terms_ptr->get_number_of_sources() > 0) {
    has_source_terms = true;
  }
  JSINFO << "number of source terms: "
         << hydro_source_terms_ptr->get_number_of_sources()
         << ", total E = " << hydro_source_terms_ptr->get_total_E_of_sources()
         << " GeV.";
   */

   // *************************************************************************
   // REMARK: How to get form the pre-eq the 4 features e-density, temp, ux, uy
   // see music.cpp and init.cpp .... but maybe a short cut!???
   // *************************************************************************
   //
   // https://github.com/JETSCAPE/JETSCAPE/pull/254/files
   // Rotation etc fix ... make sure not to repeat here !!!

  //DEBUG
  //for(int i=0;i<10000;i++) cout<<pre_eq_ptr->e_[i]<<" "; // this is the energy density according to Chun ...
  //cout<<endl;

  // Get Temperature for now via ideal gas EOS ... more sophistacted later use the EOS part Musics, or includie musics and some timesteps TBD !!!
  // double EOS_idealgas::get_temperature(double eps, double rhob) const {
  //     return pow(
  //         90.0 / M_PI / M_PI * (eps / 3.0)
  //             / (2 * (Nc * Nc - 1) + 7. / 2 * Nc * Nf),
  //         .25);
  // }
  // Nc = 3
  // Nf = 3
  //
  // *************************************************************************

  clear_up_evolution_data();
  PassPreEqEvolutionHistoryToFramework();

  hydro_status = INITIALIZED;

  if (hydro_status == INITIALIZED) {
    JSINFO << "running FnoHydro ...";
    //music_hydro_ptr->run_hydro();

    //********************************************
    //definitely a memeory leak here ... !!!!????
    //********************************************

    /// ------------------------------------------------------------------
    // Dummy test here ...
    // Create a vector of inputs.

    // *************************************************************************
    // REMARK: How to duplicate the first entry, like in nump!????
    // *************************************************************************

    // tensor.unsqueeze(0): Adds a new dimension at the beginning (position 0), changing the shape from [4] to [1, 4].
    // // Repeat the tensor twice along dimension 1 and once along dimension 0
    // auto repeated_tensor_2 = tensor.repeat({1, 2});
    // std::cout << "Repeated tensor (1x2):\n" << repeated_tensor_2 << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1,4, 60, 60, 50})); // .to(at::kMPS));

    // Execute the model and turn its output into a tensor.
    ///*
    torch::Tensor output = module.forward(inputs).toTensor();

    c10::IntArrayRef shape = output.sizes();

     JSINFO << "Tensor shape: ";
      for (int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
      }
     std::cout << std::endl;

    output.detach().resize_({0});
    inputs.clear();
    //cout<<inputs.size()<<endl;
    /// ------------------------------------------------------------------

    hydro_status = FINISHED;
  }

  //PassHydroEvolutionHistoryToFramework();
  JSINFO << "number of fluid cells received by the JETSCAPE: "
             << bulk_info.data.size();

  /*
  if (flag_surface_in_memory == 1) {
    clearSurfaceCellVector();
    PassHydroSurfaceToFramework();
  } else {
    collect_freeze_out_surface();
  }

  if (hydro_status == FINISHED && doCooperFrye == 1) {
    music_hydro_ptr->run_Cooper_Frye();
  }
  */
}

void FnoHydro::SetPreEqGridInfo() {
  bulk_info.tau_min = pre_eq_ptr->GetPreequilibriumStartTime();
  bulk_info.dtau = pre_eq_ptr->GetPreequilibriumEvodtau();
  JSINFO << "preEq evo: tau_0 = " << bulk_info.tau_min
         << " fm/c, dtau = " << bulk_info.dtau << " fm/c.";
}


/*
void MpiMusic::SetHydroGridInfo() {
  bulk_info.neta = music_hydro_ptr->get_neta();
  bulk_info.nx = music_hydro_ptr->get_nx();
  bulk_info.ny = music_hydro_ptr->get_nx();
  bulk_info.x_min = -music_hydro_ptr->get_hydro_x_max();
  bulk_info.dx = music_hydro_ptr->get_hydro_dx();
  bulk_info.y_min = -music_hydro_ptr->get_hydro_x_max();
  bulk_info.dy = music_hydro_ptr->get_hydro_dx();
  bulk_info.eta_min = -music_hydro_ptr->get_hydro_eta_max();
  bulk_info.deta = music_hydro_ptr->get_hydro_deta();

  bulk_info.boost_invariant = music_hydro_ptr->is_boost_invariant();

  if (flag_preEq_output_evo_to_memory == 0) {
    bulk_info.tau_min = music_hydro_ptr->get_hydro_tau0();
    bulk_info.dtau = music_hydro_ptr->get_hydro_dtau();
    bulk_info.ntau = music_hydro_ptr->get_ntau();
  } else {
    bulk_info.ntau = music_hydro_ptr->get_ntau() + pre_eq_ptr->get_ntau();
  }
}
*/

void FnoHydro::PassPreEqEvolutionHistoryToFramework() {
  JSINFO << "Passing preEq evolution information to JETSCAPE ... ";
  auto number_of_cells = pre_eq_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of preEq fluid cells: " << number_of_cells;

  SetPreEqGridInfo();

  for (int i = 0; i < number_of_cells; i++) {
    std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
    pre_eq_ptr->get_fluid_cell_with_index(i, fluid_cell_info_ptr);
    StoreHydroEvolutionHistory(fluid_cell_info_ptr);
  }
  pre_eq_ptr->clear_evolution_data();
}

/*
void FnoHydro::PassHydroEvolutionHistoryToFramework() {
  JSINFO << "Passing hydro evolution information to JETSCAPE ... ";
  auto number_of_cells = music_hydro_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of MUSIC fluid cells: " << number_of_cells;

  SetHydroGridInfo();

  fluidCell *fluidCell_ptr = new fluidCell;
  for (int i = 0; i < number_of_cells; i++) {
    std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
    music_hydro_ptr->get_fluid_cell_with_index(i, fluidCell_ptr);

    fluid_cell_info_ptr->energy_density = fluidCell_ptr->ed;
    fluid_cell_info_ptr->entropy_density = fluidCell_ptr->sd;
    fluid_cell_info_ptr->temperature = fluidCell_ptr->temperature;
    fluid_cell_info_ptr->pressure = fluidCell_ptr->pressure;
    fluid_cell_info_ptr->vx = fluidCell_ptr->vx;
    fluid_cell_info_ptr->vy = fluidCell_ptr->vy;
    fluid_cell_info_ptr->vz = fluidCell_ptr->vz;
    fluid_cell_info_ptr->mu_B = 0.0;
    fluid_cell_info_ptr->mu_C = 0.0;
    fluid_cell_info_ptr->mu_S = 0.0;
    fluid_cell_info_ptr->qgp_fraction = 0.0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        fluid_cell_info_ptr->pi[i][j] = fluidCell_ptr->pi[i][j];
      }
    }
    fluid_cell_info_ptr->bulk_Pi = fluidCell_ptr->bulkPi;
    StoreHydroEvolutionHistory(fluid_cell_info_ptr);
  }
  delete fluidCell_ptr;

  music_hydro_ptr->clear_hydro_info_from_memory();
}


void FnoHydro::PassHydroSurfaceToFramework() {
    JSINFO << "Passing hydro surface cells to JETSCAPE ... ";
    auto number_of_cells = music_hydro_ptr->get_number_of_surface_cells();
    JSINFO << "total number of fluid cells: " << number_of_cells;
    SurfaceCell surfaceCell_i;
    for (int i = 0; i < number_of_cells; i++) {
        SurfaceCellInfo surface_cell_info;
        music_hydro_ptr->get_surface_cell_with_index(i, surfaceCell_i);
        surface_cell_info.tau = surfaceCell_i.xmu[0];
        surface_cell_info.x = surfaceCell_i.xmu[1];
        surface_cell_info.y = surfaceCell_i.xmu[2];
        surface_cell_info.eta = surfaceCell_i.xmu[3];
        double u[4];
        for (int j = 0; j < 4; j++) {
            surface_cell_info.d3sigma_mu[j] = surfaceCell_i.d3sigma_mu[j];
            surface_cell_info.umu[j] = surfaceCell_i.umu[j];
        }
        surface_cell_info.energy_density = surfaceCell_i.energy_density;
        surface_cell_info.temperature = surfaceCell_i.temperature;
        surface_cell_info.pressure = surfaceCell_i.pressure;
        surface_cell_info.baryon_density = surfaceCell_i.rho_b;
        surface_cell_info.mu_B = surfaceCell_i.mu_B;
        surface_cell_info.mu_Q = surfaceCell_i.mu_Q;
        surface_cell_info.mu_S = surfaceCell_i.mu_S;
        for (int j = 0; j < 10; j++) {
            surface_cell_info.pi[j] = surfaceCell_i.shear_pi[j];
        }
        surface_cell_info.bulk_Pi = surfaceCell_i.bulk_Pi;
        StoreSurfaceCell(surface_cell_info);
    }
}
*/

void FnoHydro::GetHydroInfo(
    Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
  //GetHydroInfo_JETSCAPE(t, x, y, z, fluid_cell_info_ptr);
  //GetHydroInfo_MUSIC(t, x, y, z, fluid_cell_info_ptr);
}
