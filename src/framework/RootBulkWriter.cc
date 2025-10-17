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
    out_file_name = GetXMLElementText({"RootBulkWriter","out_file_name"});
    dTau = GetXMLElementDouble({"RootBulkWriter","dTau"});
    maxTau = GetXMLElementDouble({"RootBulkWriter","maxTau"});
    dX = GetXMLElementDouble  ({"RootBulkWriter","dX"});
    maxabsX = GetXMLElementDouble({"RootBulkWriter","maxabsX"});
    if (maxTau <= 0) use_vec = true;

  bool ensure_MUSIC = (int)(GetXMLElementInt({"RootBulkWriter","ensure_MusicWrapper_output"}, false));
  if (!ensure_MUSIC) {
    JSWARN << " WARNING: RootBulkWriter: ensure_MusicWrapper_output not set to true. This will likely lead to erroneous output. Please fix your XML file.";
    JSINFO << " WARNING: RootBulkWriter: ensure_MusicWrapper_output not set to true. This will likely lead to erroneous output. Please fix your XML file.";
  }
}

RootBulkWriter::RootBulkWriter() 
  // The information to init (the size of music, etc...) isn't present until
  // after the first Exec() call to MUSIC, so init is done in first Exec() instead.
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

  // save MUSIC parameters
  int nX_MUSIC = bInfo.nx;
  float dX_MUSIC = bInfo.dx;

  dTau_MUSIC = bInfo.dtau;
  tau0 = bInfo.Tau0();

  TParameter<int> p_nX_MUSIC("nX_MUSIC", nX_MUSIC);
  p_nX_MUSIC.Write();

  TParameter<float> p_dX_MUSIC("dX_MUSIC", dX_MUSIC);
  p_dX_MUSIC.Write();

  TParameter<int> p_dTau_MUSIC("dTau_MUSIC", dTau_MUSIC);
  p_dTau_MUSIC.Write();

  TParameter<float> p_tau0_MUSIC("tau0", tau0);
  p_tau0_MUSIC.Write();

  if (dX == 0) dX = dX_MUSIC;
  if (maxabsX == 0) maxabsX = fabs(bInfo.XMin());
  if (dTau == 0) dTau = dTau_MUSIC;

  // derive values of nX (nY=nX), and possible nTau
  nX = int(2*maxabsX/dX);
  if (!use_vec) nTau = int((maxTau - tau0)/dTau+1);

  TParameter<float> p_maxabsX("maxabsX", maxabsX);
  p_maxabsX.Write();

  TParameter<float> p_dX("dX", dX);
  p_dX.Write();

  TParameter<float> p_maxTau("maxTau", maxTau);
  p_maxTau.Write();

  TParameter<float> p_dTau("dTau", dTau);
  p_dTau.Write();

  TParameter<int> p_nX("nX", nX);
  p_nX.Write();

  TParameter<int> p_nTau("nTau", nTau);
  p_nTau.Write();

  TParameter<int> p_nFeatures("nFeatures", nFeatures);
  p_nFeatures.Write();

  int mult_T = nX*nX*nFeatures;

  if (use_vec) {
    ntotal = nX*nX*nFeatures;
    JSINFO << " RootBulkWriter writing variable sized vector for tau steps ";
    t->Branch("user_res", &v_data);
  } else {
    ntotal = nX*nX*nFeatures*nTau;
    JSINFO << " RootBulkWriter writing fixed sized array for " << nTau << " tau steps out to " << (tau0+nTau*dTau) << " fm/c";

    data = std::make_unique<float[]>(ntotal); //new float[ntotal];
    t->Branch("user_res", data.get(), Form("user_res[%d]/F", ntotal));
    JSINFO << " What is the size? " << bInfo.get_data_size();
  }
  t->Branch("tau_freezeout", &tau_freezeout, "tau_freezeout/F");
  t->Branch("ntau_freezeout", &ntau_freezeout, "ntau_freezeout/I");
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

  tau_freezeout = tau0 + bInfo.ntau * dTau_MUSIC;
  ntau_freezeout = int((tau_freezeout - tau0) / dTau) + 1;
  if (use_vec) { // fill in vector of tau times
    v_data.clear();
    v_data.reserve(nX * nX * nFeatures * ntau_freezeout);
    for (int k = 0; k < ntau_freezeout; k++) {
      double tau_In = tau0 + k * dTau;
      for (int i = 0; i < nX; i++) {
        double x_In = -maxabsX + i * dX;
        for (int j = 0; j < nX; j++) { // y-direction
          double y_In = -maxabsX + j * dX;
          auto mCell = bInfo.get(tau_In, x_In, y_In, 0);
          v_data.push_back((float)(mCell.energy_density));
          v_data.push_back((float)(mCell.vx));
          v_data.push_back((float)(mCell.vy));
        }
      }
    }
  } else { // use fixed-size for number of tau times
    int _nTau = std::min(ntau_freezeout, nTau);
    size_t index = 0;

    for (int k = 0; k < _nTau; k++) {
      double tau_In = tau0 + k * dTau;
      for (int i = 0; i < nX; i++) {
        double x_In = -maxabsX + i * dX;
        for (int j = 0; j < nX; j++) {
          double y_In = -maxabsX + j * dX;
          auto mCell = bInfo.get(tau_In, x_In, y_In, 0);
          data[index++] = (float)(mCell.energy_density);
          data[index++] = (float)(mCell.vx);
          data[index++] = (float)(mCell.vy);
        }
      }
      VERBOSE(3) << " RootBulkWriter writing step " << k << " of " << nTau
                 << " rho_sum: " << " tau: " << tau_In;
    }
    // pad with zeros for time past freezeout, if needed
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
