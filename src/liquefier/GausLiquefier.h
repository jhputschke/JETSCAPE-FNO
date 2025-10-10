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

#ifndef GAUSLIQUEFIER_H
#define GAUSLIQUEFIER_H

#include "LiquefierBase.h"
#include "RealType.h"
#include <array>

#include <cmath>
#include <gsl/gsl_sf_bessel.h>

namespace Jetscape {

class GausLiquefier: public Jetscape::LiquefierBase {
 private:
     float sigma_tau;
     float sigma_Rxy;
    
 public:
    //parameters (to be moved to xml)---------------------------
    float sigma_Rxy2 {1.3};// in [fm^2]
    float sigma_tau2 {1.1};// in [fm^2]

    GausLiquefier();

    ~GausLiquefier() {};
    
    void Init();
    
    void smearing_kernel(Jetscape::real, Jetscape::real,
                         Jetscape::real, Jetscape::real,
                         const Droplet,
                         std::array<Jetscape::real, 4>&) const;

    double dumping(double t) const;
};

};

#endif  // GAUSLIQUEFIER_H
