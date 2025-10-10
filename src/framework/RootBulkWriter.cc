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
// -----------------------------------------
// JETSCAPE module for soft particlization
// This module will generate Monte-Carlo samples for soft hadrons
// -----------------------------------------
#ifdef USE_ROOT

#include "RootBulkWriter.h"
#include <iostream>
#include <time.h>
#include <string>

// JetScape Framework includes ...
#include "JetScape.h"
#include "JetEnergyLoss.h"
#include "JetEnergyLossManager.h"
#include "JetScapeWriterStream.h"
#include "JetScapeSignalManager.h"
#ifdef USE_HEPMC
#include "JetScapeWriterHepMC.h"
//#include "JetScapeWriterRootHepMC.h"
#endif


// User modules derived from jetscape framework clasess
#include "TrentoInitial.h"
#include "AdSCFT.h"
#include "Matter.h"
#include "LBT.h"
#include "Martini.h"
#include "Brick.h"
#include "GubserHydro.h"
#include "MusicWrapper.h"
#include "PythiaGun.h"
#include "iSpectraSamplerWrapper.h"
#include "TrentoInitial.h"
#include "NullPreDynamics.h"
#include "PGun.h"
#include "HadronizationManager.h"
#include "Hadronization.h"
#include "ColoredHadronization.h"
#include "ColorlessHadronization.h"
//#include "HydroFromFile.h"

#include <chrono>
#include <thread>

#include "TParameter.h"
#include <Riostream.h>
#include "TRandom.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TF1.h"
#include "TMath.h"
#include "TFile.h"
#include "TString.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

using namespace std;
using namespace Jetscape;

RegisterJetScapeModule<RootBulkWriter> RootBulkWriter::reg("RootBulkWriter");

// Forward declarations
// / -------------------------------------

void Show();

// -------------------------------------
void RootBulkWriter::Init() {
    out_file_name = GetXMLElementText({"RootBulkWriter","ofilename"});
    nT = GetXMLElementInt({"RootBulkWriter","n_tau_steps"});
    min_dtau = GetXMLElementDouble({"RootBulkWriter","min_dtau"});
    nAvg = GetXMLElementInt({"RootBulkWriter","navg_xy"});
    nAvgsq = nAvg*nAvg;

    if (nT < 0) use_vec = true;
}

RootBulkWriter::RootBulkWriter() : 
    /* JetScapeModuleBase(), */ 
    f{nullptr}, 
    t{nullptr},
    isinit{false}, 
  // The information to init (the size of music, etc...) isn't present until
  // after the first Exec() call, so init is really done there.
    nX{0}, nY{0}, nT{0},
    xMin{0}, xMax{0}, yMin{0}, yMax{0}, dX{0}, dY{0}, dTau{0}, tau0{0},
    out_file_name{"bulk_root_writer.root"},
    in_music_name{"HYDRO"}
{
    JSINFO << " Adding RootBulkWriter ";
    SetId("RootBulkWriter");
} 

void RootBulkWriter::Show()
{
  INFO_NICE<<"------------------------------------------";
  INFO_NICE<<"| Bulk ROOT Writer JetScape Framework ... |";
  INFO_NICE<<"------------------------------------------";
  INFO_NICE;
}

void RootBulkWriter::init_tree(const EvolutionHistory& bInfo) {
  isinit=true;

  f=new TFile(out_file_name.c_str(),"RECREATE");
  t=new TTree("t","Tree");

  nX_in = bInfo.ny;
  nY_in = bInfo.nx;

  nX = nX_in / nAvg;
  nY = nY_in / nAvg;

  if (use_vec) {
     nT = -1;
  }
  tau0 = bInfo.Tau0();

  xMin = bInfo.XMin();
  xMax = bInfo.XMax(); //same for y axis ...

  dX = bInfo.dx;
  dTau = bInfo.dtau;

  // put the parameters in the output root file
  TParameter<int> p_nxyAvg("grid_nXYavg", nAvg);
  p_nxyAvg.Write();

  TParameter<int> p_nX("nX", nX);
  p_nX.Write();

  TParameter<int> p_nY("nY", nY);
  p_nY.Write();

  TParameter<int> p_nT("nT", nT);
  p_nT.Write();

  TParameter<int> p_nFeatures("nFeatures", nFeatures);
  p_nFeatures.Write();

  TParameter<float> p_tau0("tau0", tau0);
  p_tau0.Write();

  TParameter<float> p_xMin("xMin", xMin);
  p_xMin.Write();

  TParameter<float> p_xMax("xMax", xMax);
  p_xMax.Write();


  TParameter<float> p_dX("dX", dX);
  p_dX.Write();


  mult_T = nX*nY*nFeatures;
  mult_X = nY*nFeatures;
  mult_Y = nFeatures;

  float out_dtau = dTau;

  if (use_vec) {
    ntotal = nX*nY*nFeatures;
    JSINFO << " RootBulkWriter writing variable sized vector for tau steps ";
    t->Branch("user_res", &v_data);
  } else {

    tau_skip = int(min_dtau/dTau);
    if (tau_skip < 1) tau_skip = 1;
    if (tau_skip > 1) {
      // if (min_dtau > tau_skip*dTau) tau_skip++;
      out_dtau = dTau * tau_skip;
      JSINFO << "RootBulkWriter::init_tree :: MUSIC's dTau(" 
             << dTau << ") < min_dtau(" << min_dtau <<") so only writing"
             << " out every " << tau_skip << "th step, for new dtau of " << out_dtau << ".";
    }
    JSINFO << " RootBulkWriter writing fixed sized array for " << nT << " tau steps out to " << nT*out_dtau+tau0 << " fm/c";

    ntotal = mult_T * nT;
    data = std::make_unique<float[]>(ntotal); //new float[ntotal];
    JSINFO << " RootBulkInfo Filling size from " << ntotal << " nX " << nX << " nY " << nY << " nT " << nT << " nFeatures " << nFeatures << " and xMax: " << xMax << " dX " << dX << " xMin " << xMin;
    t->Branch("user_res", data.get(), Form("user_res[%d]/F", ntotal));
    JSINFO << " What is the size? " << bInfo.get_data_size();
  }

  TParameter<float> p_dTau("dTau", out_dtau);
  p_dTau.Write();
}

void RootBulkWriter::Exec() {
  auto hydro = JetScapeSignalManager::Instance()->GetHydroPointer();

  if (!hydro.lock()) {
    JSWARN << " No hydro pointer found for RootBulkWriter. "
           << " Skipping RootBulkWriter::Exec() logic.";
    return;
  }

  auto bInfo = hydro.lock()->get_bulk_info();
  if (!isinit) {
    init_tree(bInfo);
  }

  int this_ntau = bInfo.ntau;
  if (use_vec) { // fill in vector of tau times
    v_data.clear();
    v_data.reserve(nX * nY * nFeatures * this_ntau);
    if (nAvg == 1) {
      for (int k = 0; k < this_ntau; k += tau_skip) {
        double tau_In = tau0 + k * dTau;
        for (int i = 0; i < nX_in; i++) {
          double x_In = xMin + i * dX;
          for (int j = 0; j < nY_in; j++) {
            double y_In = xMin + j * dX;
            auto mCell = bInfo.get(tau_In, x_In, y_In, 0);
            v_data.push_back((float)(mCell.energy_density));
            v_data.push_back((float)(mCell.vx));
            v_data.push_back((float)(mCell.vy));
          }
        }
      }
    } else { // averaging over (nAvg)x(nAvg) cells
      for (int k = 0; k < this_ntau; k += tau_skip) {
        double tau_In = tau0 + k * dTau;
        for (int i = 0; i < nX_in-nAvg+1; i += nAvg) {
          for (int j = 0; j < nY_in-nAvg+1; j += nAvg) {
            float mean_rho{0.};
            float mean_vx{0.};
            float mean_vy{0.};
            for (int ii = i; ii < i + nAvg; ii++) {
              double x_In = xMin + ii * dX;
              for (int jj = j; jj < j + nAvg; jj++) {
                double y_In = xMin + jj * dX;
                auto mCell = bInfo.get(tau_In, x_In, y_In, 0);
                mean_rho += mCell.energy_density;
                mean_vx += mCell.vx;
                mean_vy += mCell.vy;
              }
            }
            v_data.push_back((float)(mean_rho / nAvgsq));
            v_data.push_back((float)(mean_vx / nAvgsq));
            v_data.push_back((float)(mean_vy / nAvgsq));
          }
        }
      }
    }
  } else { // use fixed-size for number of tau times
    // fill in the data
    int max_itau = nT * tau_skip;
    if (max_itau > this_ntau) {
      max_itau = this_ntau;
    }
    size_t index = 0;

    if (nAvg == 1) {
      for (int k = 0; k < (max_itau); k += tau_skip) {
        double check_rho = 0;
        double tau_In = tau0 + k * dTau;
        for (int i = 0; i < nX_in; i++) {
          double x_In = xMin + i * dX;
          for (int j = 0; j < nY_in; j++) {
            double y_In = xMin + j * dX;
            auto mCell = bInfo.get(tau_In, x_In, y_In, 0);

            check_rho += (float)(mCell.energy_density);
            data[index++] = (float)(mCell.energy_density);
            data[index++] = (float)(mCell.vx);
            data[index++] = (float)(mCell.vy);
          }
        }
        VERBOSE(3) << " RootBulkWriter writing step " << k << " of " << nT
                   << " rho_sum: " << check_rho << " tau: " << tau_In;
      }
    } else { // averaging over (nAvg)x(nAvg) cells}
      for (int k = 0; k < (max_itau); k += tau_skip) {
        int check_0tau = index;
        double check_rho = 0;
        double tau_In = tau0 + k * dTau;
        for (int i = 0; i < nX_in-nAvg+1; i += nAvg) {
          for (int j = 0; j < nY_in-nAvg+1; j += nAvg) {
            float mean_rho{0.};
            float mean_vx{0.};
            float mean_vy{0.};

            for (int ii = i; ii < i + nAvg; ii++) {
              double x_In = xMin + ii * dX;
              for (int jj = j; jj < j + nAvg; jj++) {
                double y_In = xMin + jj * dX;
                auto mCell = bInfo.get(tau_In, x_In, y_In, 0);
                mean_rho += mCell.energy_density;
                mean_vx += mCell.vx;
                mean_vy += mCell.vy;
              }
            }

            data[index++] = (float)(mean_rho / nAvgsq);
            data[index++] = (float)(mean_vx / nAvgsq);
            data[index++] = (float)(mean_vy / nAvgsq);
          }
        }
        VERBOSE(5) << " RootBulkWriter: Index passed: " << (index - check_0tau) << " comp to 3*(nX*nY) " << 3*(nX*nY) 
            << " with nX_in:nY_in:nX:nX " << nX_in << ":" << nY_in << ":" << nX << ":" << nY;
        VERBOSE(5) << " ROOTBULKWRITEER writing step " << k << " of " << nT
                   << " rho_sum: " << check_rho << " tau: " << tau_In;
      } // end loop over tau

    } // end fixed array with avg cells
     
    // fill in zeros for the rest taus (if any)
    for (int i = index; i < ntotal; i++) {
      data[index++] = 0.;
    }
  } // end fixed arrays
  t->Fill();
}

RootBulkWriter::~RootBulkWriter() {
  f->cd();
  f->ls();
  t->Print();
  f->Write();
  f->Close();
}

#endif // USE_ROOT
