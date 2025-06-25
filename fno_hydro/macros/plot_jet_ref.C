void plot_jet_ref(TString fFNO="./test_ana_jet_fno_val.root", TString fHydro="./test_ana_jet_hydro_val.root", TString fNull = "./test_ana_jet_hydro_null_medium.root")
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

    TCanvas *c1 = new TCanvas("c1", "Canvas #1", 800, 600);
    hPtFno->SetMarkerColor(2);
    hPtFno->SetMarkerStyle(22);
    hPtFno->SetLineColor(2);
    hPtHydro->SetMarkerStyle(23);
    hPtNull->SetMarkerStyle(24);

    hPtFno->DrawCopy("");
    hPtHydro->DrawCopy("same");
    hPtNull->DrawCopy("same");

    TCanvas *c2 = new TCanvas("c2", "Canvas #2", 800, 600);
    hPtFno->Divide(hPtHydro);
    hPtFno->DrawCopy("");

    TCanvas *c3 = new TCanvas("c3", "Canvas #3", 800, 600);
    hzFno->SetMarkerColor(2);
    hzFno->SetMarkerStyle(22);
    hzFno->SetLineColor(2);
    hzHydro->SetMarkerStyle(23);
    hzNull->SetMarkerStyle(24);

    hzFno->DrawCopy("");
    hzHydro->DrawCopy("same");
    hzNull->DrawCopy("same");

    TCanvas *c4 = new TCanvas("c4", "Canvas #4", 800, 600);
    hzFno->Divide(hzHydro);
    hzFno->DrawCopy("");


}
