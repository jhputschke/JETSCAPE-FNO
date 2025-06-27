void plot_jet_ref(TString fFNO="./test_ana_jet_fno_nReuse_10_val.root", TString fHydro="./test_ana_jet_hydro_nReuse_10_val.root", TString fNull = "./test_ana_jet_hydro_null_medium.root")
{
    gStyle->SetOptStat(0);

    TFile *fF = new TFile(fFNO,"READ");
    TFile *fH = new TFile(fHydro,"READ");
    TFile *fN = new TFile(fNull,"READ");

    auto hPtFno = (TH1D*) fF->Get("hPt");
    auto hPtHydro = (TH1D*) fH->Get("hPt");
    auto hPtNull = (TH1D*) fN->Get("hPt");
    auto hzFno = (TH1D*) fF->Get("hz");
    auto hzNull = (TH1D*) fN->Get("hz");
    auto hzHydro = (TH1D*) fH->Get("hz");
    auto hMFno = (TH1D*) fF->Get("hM");
    auto hMHydro = (TH1D*) fH->Get("hM");
    auto hMNull = (TH1D*) fN->Get("hM");

    TCanvas *c1 = new TCanvas("c1", "Canvas #1", 800, 600);
    hPtFno->SetMarkerColor(2);
    hPtFno->SetMarkerStyle(22);
    hPtFno->SetLineColor(2);
    hPtHydro->SetMarkerStyle(29);
    hPtNull->SetMarkerStyle(24);

    hPtFno->Scale(1/(double) hPtFno->GetEntries());
    hPtHydro->Scale(1/(double) hPtHydro->GetEntries());
    hPtNull->Scale(1/(double) hPtNull->GetEntries());

    hPtNull->DrawCopy("");
    hPtFno->DrawCopy("same");
    hPtHydro->DrawCopy("same");

    TCanvas *c2 = new TCanvas("c2", "Canvas #2", 800, 600);
    hPtFno->Divide(hPtHydro);
    hPtFno->DrawCopy("");

    TCanvas *c3 = new TCanvas("c3", "Canvas #3", 800, 600);
    hzFno->SetMarkerColor(2);
    hzFno->SetMarkerStyle(22);
    hzFno->SetLineColor(2);
    hzHydro->SetMarkerStyle(29);
    hzNull->SetMarkerStyle(24);

    hzFno->Rebin(2);hzHydro->Rebin(2);hzNull->Rebin(2);

    hzFno->DrawCopy("");
    hzHydro->DrawCopy("same");
    hzNull->DrawCopy("same");

    TCanvas *c4 = new TCanvas("c4", "Canvas #4", 800, 600);
    hzFno->Divide(hzHydro);
    hzFno->DrawCopy("");
    hzHydro->Divide(hzNull);
    hzHydro->DrawCopy("same");

    TCanvas *c5 = new TCanvas("c5", "Canvas #5", 800, 600);
    hMFno->SetMarkerColor(2);
    hMFno->SetMarkerStyle(22);
    hMFno->SetLineColor(2);
    hMHydro->SetMarkerStyle(29);
    hMNull->SetMarkerStyle(24);

    hMFno->Scale(1/(double) hMFno->GetEntries());
    hMHydro->Scale(1/(double) hMHydro->GetEntries());
    hMNull->Scale(1/(double) hMNull->GetEntries());

    hMNull->DrawCopy("");
    hMFno->DrawCopy("same");
    hMHydro->DrawCopy("same");


}
