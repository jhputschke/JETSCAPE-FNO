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
// Reader test (focus on graph)

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <thread>

#include "gzstream.h"
#include "PartonShower.h"
#include "JetScapeLogger.h"
#include "JetScapeReader.h"
#include "JetScapeReaderFinalStateHadrons.h"
#include "JetScapeBanner.h"
#include "fjcore.hh"

#include <GTL/dfs.h>

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
//using namespace fjcore;

using namespace Jetscape;

// -------------------------------------

// Forward declaration
ostream & operator<<(ostream & ostr, const fjcore::PseudoJet & jet);

// ----------------------

int main(int argc, char** argv)
{
  JetScapeLogger::Instance()->SetDebug(false);
  JetScapeLogger::Instance()->SetRemark(false);
  //SetVerboseLevel (9 a lot of additional debug output ...)
  //If you want to suppress it: use SetVerboseLevle(0) or max  SetVerboseLevle(9) or 10
  JetScapeLogger::Instance()->SetVerboseLevel(0);

  TString fNameOut = "test_bulk_ana.root";
  string fNameIn = "test_out.dat";

  if (argc > 1)
    fNameIn = argv[1];
    if (argc > 2)
      fNameOut = argv[2];

  cout<<endl;

  TFile* file = new TFile(fNameOut, "RECREATE");
  TH1D* hPt = new TH1D("hPt", "Pt", 50, 0, 5); hPt->Sumw2();
  TH1D *hPhi = new TH1D("hPhi","Phi",180,0,2*TMath::Pi());
  TH1D *hEta = new TH1D("hEta","Eta",22,-1.1,1.1);

  auto reader=make_shared<JetScapeReaderAscii>(fNameIn);

  while (!reader->Finished())
    {
      reader->Next();

      cout<<"Analyze current event = "<<reader->GetCurrentEvent()<<endl;
      auto hadrons = reader->GetHadrons();
      cout<<"Number of hadrons is: " << hadrons.size() << endl;
      for (auto h : hadrons)
      {
        //cout<<h<<endl;
        if (TMath::Abs(h->eta())<1) {
            hEta->Fill(h->eta());
            hPt->Fill(h->pt());
            hPhi->Fill(h->phi());
        }
      }
    }

    reader->Close();
    file->Write();
    file->Close();
}

//----------------------------------------------------------------------
/// overloaded jet info output

ostream & operator<<(ostream & ostr, const fjcore::PseudoJet & jet) {
  if (jet == 0) {
    ostr << " 0 ";
  } else {
    ostr << " pt = " << jet.pt()
         << " m = " << jet.m()
         << " y = " << jet.rap()
         << " phi = " << jet.phi();
  }
  return ostr;
}


//----------------------------------------------------------------------
