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

  TString fNameOut = "test_ana.root";
  string fNameIn = "test_out.dat";

  if (argc > 1)
    fNameIn = argv[1];
    if (argc > 2)
      fNameOut = argv[2];


  cout<<endl;

  TFile* file = new TFile(fNameOut, "RECREATE");
  TH1D* hPt = new TH1D("hPt", "Pt", 60, 0, 60); hPt->Sumw2();
  TH1D* hM = new TH1D("hM", "PM", 40, 0, 10); hM->Sumw2();

  //Do some dummy jetfinding ...
  fjcore::JetDefinition jet_def(fjcore::antikt_algorithm, 0.7);

  vector<shared_ptr<PartonShower>> mShowers;

  // Hide Template (see class declarations in reader/JetScapeReader.h) ...
  auto reader=make_shared<JetScapeReaderAscii>(fNameIn);
  //auto reader=make_shared<JetScapeReaderAsciiGZ>("test_out.dat.gz");

  while (!reader->Finished())
    {
      reader->Next();

      cout<<"Analyze current event = "<<reader->GetCurrentEvent()<<endl;
      mShowers=reader->GetPartonShowers();

      /*
      int finals = 0;
      for (int i=0;i<mShowers.size();i++) {
	  cout<<" Analyze parton shower = "<<i<<endl;

	  //mShowers[i]->PrintVertices();
	  //mShowers[i]->PrintPartons();

	  finals += mShowers[i]->GetFinalPartonsForFastJet().size();

	  fjcore::ClusterSequence cs(mShowers[i]->GetFinalPartonsForFastJet(), jet_def);

	  vector<fjcore::PseudoJet> jets = fjcore::sorted_by_pt(cs.inclusive_jets(2));
	  cout<<endl;
	  cout<<jet_def.description()<<endl;
	  // Output of found jets ...
	  //cout<<endl;
	  for (int k=0;k<jets.size();k++)
	    cout<<"Anti-kT jet "<<k<<" : "<<jets[k]<<endl;
	  cout<<endl;
	  cout<<"Shower initiating parton : "<<*(mShowers[i]->GetPartonAt(0))<<endl;
	  cout<<endl;

	*/

    //cout << " Found " << finals << " final state partons." << endl;

      auto hadrons = reader->GetHadrons();
      cout<<"Number of hadrons is: " << hadrons.size() << endl;

      fjcore::ClusterSequence hcs(reader->GetHadronsForFastJet(), jet_def);
      vector<fjcore::PseudoJet> hjets = fjcore::sorted_by_pt(hcs.inclusive_jets(2));
      //cout<<"AT HADRONIC LEVEL " << endl;
      for (int k=0;k<hjets.size();k++) {
          if (k>0) break;
          cout<<"Anti-kT jet "<<k<<" : "<<hjets[k]<<endl;
          hPt->Fill(hjets[k].pt());
          hM->Fill(hjets[k].m());
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
