/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Modular, task-based framework for simulating all aspects of heavy-ion
 *collisions
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

#ifndef PARTONSHOWERGENERATOR_H
#define PARTONSHOWERGENERATOR_H

namespace Jetscape {

class JetEnergyLoss;

/**
 * @class PartonShowerGenerator
 * @brief Base class for generating parton showers.
 *
 * The PartonShowerGenerator class encapsulates the algorithm that
 * evolves partons into a full parton shower. The shower starts from
 * a single initiating hard parton and is recursively evolved in time,
 * with splittings and energy-loss effects modeled by a JetEnergyLoss module.
 *
 * The parton shower is represented as a graph consisting of
 * vertices (splitting points) and partons (edges).
 */
class PartonShowerGenerator {
 public:
  /**
   * @brief Default constructor.
   */
  PartonShowerGenerator(){};

  /**
   * @brief Virtual destructor.
   */
  virtual ~PartonShowerGenerator(){};

  /**
   * @brief Perform the parton shower evolution.
   *
   * This method takes a reference to a JetEnergyLoss module, which
   * provides the splitting functions and energy-loss dynamics.
   * The parton shower starts with the initiating parton provided by
   * the JetEnergyLoss object, and evolves recursively in discrete
   * time steps until the maximum evolution time is reached.
   *
   * The generated shower is recorded in the PartonShower object
   * associated with the JetEnergyLoss module.
   *
   * @param j Reference to the JetEnergyLoss module controlling
   *          the shower evolution.
   */
  virtual void DoShower(JetEnergyLoss &j);
};

}  // end namespace Jetscape

#endif
