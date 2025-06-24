void plot_bulk(TString fFNO="./test_ana_fno.root", TString fHydro="./test_ana_hydro.root")
{
    gStyle->SetOptStat(0);

    TFile *fF = new TFile(fFNO,"READ");
    TFile *fH = new TFile(fHydro,"READ");

    auto hPhiFno = (TH1D*) fF->Get("hPhi");
    auto hPhiHydro = (TH1D*) fH->Get("hPhi");

    auto hPtFno = (TH1D*) fF->Get("hPt");
    auto hPtHydro = (TH1D*) fH->Get("hPt");

    TCanvas *c1 = new TCanvas("c1", "Canvas #1", 800, 600);
    hPhiFno->SetLineColor(2);

    hPhiFno->DrawCopy("");
    hPhiHydro->DrawCopy("same");

    TCanvas *c2 = new TCanvas("c2", "Canvas #2", 800, 600);
    hPhiFno->Divide(hPhiHydro);
    hPhiFno->DrawCopy("");

    TCanvas *c3 = new TCanvas("c3", "Canvas #3", 800, 600);
    hPtFno->SetLineColor(2);

    hPtFno->DrawCopy("");
    hPtHydro->DrawCopy("same");

}
