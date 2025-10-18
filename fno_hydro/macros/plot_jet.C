void plot_jet(TString fFNO="./test_ana_jet_fno_val.root", TString fHydro="./test_ana_jet_hydro_val.root")
{
    gStyle->SetOptStat(0);

    TFile *fF = new TFile(fFNO,"READ");
    TFile *fH = new TFile(fHydro,"READ");

    auto hPtFno = (TH1D*) fF->Get("hPt");
    auto hPtHydro = (TH1D*) fH->Get("hPt");
    auto hzFno = (TH1D*) fF->Get("hz");
    auto hzHydro = (TH1D*) fH->Get("hz");
    auto hMFno = (TH1D*) fF->Get("hM");
    auto hMHydro = (TH1D*) fH->Get("hM");

    TCanvas *c1 = new TCanvas("c1", "Canvas #1", 800, 600);
    hPtFno->SetMarkerColor(2);
    hPtFno->SetMarkerStyle(22);
    hPtFno->SetLineColor(2);
    hPtHydro->SetMarkerStyle(26);

    hPtFno->DrawCopy("");
    hPtHydro->DrawCopy("same");

    TCanvas *c2 = new TCanvas("c2", "Canvas #2", 800, 600);
    hPtFno->Divide(hPtHydro);
    hPtFno->DrawCopy("");

    TCanvas *c3 = new TCanvas("c3", "Canvas #3", 800, 600);
    hzFno->SetMarkerColor(2);
    hzFno->SetMarkerStyle(22);
    hzFno->SetLineColor(2);
    hzHydro->SetMarkerStyle(26);

    hzFno->Rebin(2);hzHydro->Rebin(2);

    hzFno->DrawCopy("");
    hzHydro->DrawCopy("same");

    TCanvas *c4 = new TCanvas("c4", "Canvas #4", 800, 600);
    hzFno->Divide(hzHydro);
    hzFno->DrawCopy("");

    TCanvas *c5 = new TCanvas("c5", "Canvas #5", 800, 600);
    hMFno->SetMarkerColor(2);
    hMFno->SetMarkerStyle(22);
    hMFno->SetLineColor(2);
    hMHydro->SetMarkerStyle(29);

    hMFno->Scale(1/(double) hMFno->GetEntries());
    hMHydro->Scale(1/(double) hMHydro->GetEntries());

    hMFno->DrawCopy("");
    hMHydro->DrawCopy("same");
}
