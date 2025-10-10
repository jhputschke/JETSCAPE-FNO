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
// This is a causal liquefier with the JETSCAPE framework
// -----------------------------------------
//
// Put in a gaussian source that will return energy w.r.t. <tau, x, y, eta>

#include "GausLiquefier.h"
#include "JetScapeLogger.h"
#include "JetScapeXML.h"
#include <cfloat>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

namespace Jetscape {

GausLiquefier::GausLiquefier() {
    Init(); // read from XML;
};
    
void GausLiquefier::Init(){
    JSINFO << "Initialize GausLiquefier ...";

    sigma_Rxy2 = pow(JetScapeXML::Instance()->GetElementDouble({"Liquefier","GausLiquefier","sigma_xy"}),2.);
    sigma_tau2 = pow(JetScapeXML::Instance()->GetElementDouble({"Liquefier","GausLiquefier","sigma_tau"}),2.);

    
    // Get the parameters to add droplets from the input.
    // Parse droplets from XML in the format: <droplets>{{a,b,c,d},{e,f,g,h}},{{i,j,k,l},{m,n,o,p}},...</droplets>
    std::string droplets_str = JetScapeXML::Instance()->GetElementText({"Liquefier","GausLiquefier","droplets"}, " ");
    droplets_str.erase(std::remove_if(droplets_str.begin(), droplets_str.end(), ::isspace), droplets_str.end());
    if (!droplets_str.empty()) {
        // Remove all whitespace for easier parsing

        size_t pos = 0;
        while ((pos = droplets_str.find("{{")) != std::string::npos) {
            size_t end = droplets_str.find("}}", pos);
            if (end == std::string::npos) break;
            std::string droplet = droplets_str.substr(pos + 2, end - pos - 2);
            droplets_str = droplets_str.substr(end + 2);

            size_t mid = droplet.find("},{");
            if (mid == std::string::npos) continue;
            std::string xmu_str = droplet.substr(0, mid);
            std::string pmu_str = droplet.substr(mid + 3);

            std::array<Jetscape::real, 4> xmu, pmu;
            std::istringstream xmu_ss(xmu_str);
            std::istringstream pmu_ss(pmu_str);
            for (int i = 0; i < 4; ++i) xmu_ss >> xmu[i], xmu_ss.ignore(1, ',');
            for (int i = 0; i < 4; ++i) pmu_ss >> pmu[i], pmu_ss.ignore(1, ',');

            VERBOSE(3) << " Added droplet: xmu(" << xmu[0] << "," << xmu[1] << "," << xmu[2] << "," << xmu[3] << ") pmu(" << pmu[0] << "," << pmu[1] << "," << pmu[2] << "," << pmu[3] << ")";
            add_a_droplet({xmu, pmu}); //{{xmu[0],xmu[1],xmu[2],xmu[3]}, {pmu[0],pmu[1],pmu[2],pmu[3]}});
        }
    }
};


void GausLiquefier::smearing_kernel(Jetscape::real tau, Jetscape::real x,
                         Jetscape::real y, Jetscape::real eta,
                         const Droplet drop,
                         std::array<Jetscape::real, 4>& jmu) const
{
  jmu = {0.0, 0.0, 0.0, 0.0};
  const auto& x_drop = drop.get_xmu();
  const auto& p_drop = drop.get_pmu();
  auto dtau2 = pow(tau - x_drop[0],2.);
  auto dR2 = pow(x - x_drop[1], 2.) + pow(y - x_drop[2], 2.);
  auto factor = exp(-dtau2 / 2. / sigma_tau2 - dR2 / 2. / sigma_Rxy2);
  for (int i = 0; i < 4; i++) {
      jmu[i] += p_drop[i] * factor;
  }
};

};
