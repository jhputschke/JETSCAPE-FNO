void plot_bulk(TString fFNO="./test_ana_bulk_fno_2k.root", TString fHydro="./test_ana_bulk_hydro_2k.root")
{
    gStyle->SetOptStat(0);

    TFile *fF = new TFile(fFNO,"READ");
    TFile *fH = new TFile(fHydro,"READ");

    auto hPhiFno = (TH1D*) fF->Get("hPhi");
    auto hPhiHydro = (TH1D*) fH->Get("hPhi");
    auto hPhiFnoMid = (TH1D*) fF->Get("hPhiMid");
    auto hPhiHydroMid = (TH1D*) fH->Get("hPhiMid");
    //hPhiFno->Rebin(3);hPhiHydro->Rebin(3);

    auto hPtFno = (TH1D*) fF->Get("hPt");
    auto hPtHydro = (TH1D*) fH->Get("hPt");

    auto hPhiFnoFit = (TH1D*) hPhiFno->Clone("hPhiFnoFit");
    auto hPhiHydroFit = (TH1D*) hPhiHydro->Clone("hPhiHydroFit");

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

    TCanvas *c4 = new TCanvas("c4", "Canvas #4", 800, 600);
    hPtFno->Divide(hPtHydro);
    hPtFno->DrawCopy("");

    TCanvas *c5 = new TCanvas("c5", "Canvas #5", 800, 600);
    hPhiFnoFit->SetMarkerColor(2);
    hPhiFnoFit->SetMarkerStyle(22);
    hPhiFnoFit->SetLineColor(2);
    hPhiHydroFit->SetMarkerStyle(26);

    TF1 *vnFno = new TF1("vnFno","[0] * (1+[1]*TMath::Cos(2*x)+[2]*TMath::Cos(3*x)+[3]*TMath::Cos(4*x))",-TMath::Pi(),TMath::Pi());
    vnFno->SetParameters(10,0.1,0.05,0.05);

    TF1 *vnHydro = new TF1("vnHydro","[0] * (1+[1]*TMath::Cos(2*x)+[2]*TMath::Cos(3*x)+[3]*TMath::Cos(4*x))",-TMath::Pi(),TMath::Pi());
    vnHydro->SetParameters(10,0.1,0.05,0.05);
    vnHydro->SetLineColor(1);

    hPhiFnoFit->Fit("vnFno","R+");
    hPhiHydroFit->Fit("vnHydro","R+");

    hPhiFnoFit->DrawCopy("");
    hPhiHydroFit->DrawCopy("same");

    TCanvas *c6 = new TCanvas("c6", "Canvas #6", 800, 600);
    hPhiFnoMid->SetMarkerColor(2);
    hPhiFnoMid->SetMarkerStyle(22);
    hPhiFnoMid->SetLineColor(2);
    hPhiHydroMid->SetMarkerStyle(26);

    hPhiFnoMid->DrawCopy();
    hPhiHydroMid->DrawCopy("same");
}
