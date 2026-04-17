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

#ifndef JETENERGYLOSSMANAGER_H
#define JETENERGYLOSSMANAGER_H

#include <vector>

#include "JetClass.h"
#include "JetScapeTask.h"
#include "sigslot.h"

namespace Jetscape {

/**
 * @class JetEnergyLossManager
 * @brief Manager for jet energy loss tasks in the JETSCAPE framework.
 *
 * This class manages a collection of jet energy loss tasks, handling their
 * initialization, execution, clearing, and output writing. It also implements
 * the signal-slot philosophy to connect tasks such as `JetEnergyLoss` and
 * `HardProcess` with relevant functions for energy loss and parton list
 * retrieval.
 */
class JetEnergyLossManager
    : public JetScapeTask,
      public std::enable_shared_from_this<JetEnergyLossManager> {
 public:
  /**
   * @brief Default constructor.
   *
   * Creates a JetEnergyLossManager with task ID `"JLossManager"`.
   * The flag `GetHardPartonListConnected` is set to `false`.
   */
  JetEnergyLossManager();

  /**
   * @brief Destructor.
   */
  virtual ~JetEnergyLossManager();

  /**
   * @brief Initializes attached jet energy loss tasks.
   *
   * Sends a signal to connect a `JetEnergyLoss` object to the
   * `GetHardPartonList()` function of the `HardProcess` class.
   * Can be overridden by derived tasks.
   *
   * @sa JetScapeSignalManager
   */
  virtual void Init();

  /**
   * @brief Executes the jet energy loss tasks.
   *
   * Reads the hard parton list, calls `CreateSignalSlots()`, and executes the
   * attached energy loss tasks. Supports parallel execution.
   * Can be overridden by derived tasks.
   */
  virtual void Exec();

  /**
   * @brief Clears attached energy loss tasks.
   *
   * Erases all tasks attached to the manager.
   * Can be overridden by derived tasks.
   */
  virtual void Clear();

  /**
   * @brief Writes output from the jet energy loss tasks.
   *
   * Writes output information relevant to energy loss tasks/subtasks into
   * the given writer.
   *
   * @param w Weak pointer to a JetScapeWriter instance.
   * @sa JetScapeWriter
   */
  virtual void WriteTask(weak_ptr<JetScapeWriter> w);

  /**
   * @brief Get the number of registered signals.
   * @return Number of signals.
   */
  int GetNumSignals();

  /**
   * @brief Create and connect signal slots.
   *
   * Ensures that tasks are connected via signals to:
   * - `UpdateEnergyDeposit()`
   * - `GetEnergyDensity()`
   * - `GetHydroCell()` (from the FluidDynamics class)
   * - `DoEnergyLoss()` (from the JetEnergyLoss class)
   *
   * If not connected, sends the required signals.
   * Can be extended or overridden by derived tasks.
   *
   * @sa JetScapeSignalManager
   */
  void CreateSignalSlots();

  /**
   * @brief Signal to connect JetEnergyLossManager to HardProcess.
   *
   * This signal connects the JetEnergyLossManager to the
   * `GetHardPartonList()` function of the `HardProcess` class.
   */
  sigslot::signal1<vector<shared_ptr<Parton>> &> GetHardPartonList;

  /**
   * @brief Set the connection status of GetHardPartonList.
   * @param m_GetHardPartonListConnected `true` if the signal was sent to
   *        `HardProcess::GetHardPartonList()`.
   */
  void SetGetHardPartonListConnected(bool m_GetHardPartonListConnected) {
    GetHardPartonListConnected = m_GetHardPartonListConnected;
  }

  /**
   * @brief Get the connection status of GetHardPartonList.
   * @return `true` if the signal was sent to
   * `HardProcess::GetHardPartonList()`.
   */
  const bool GetGetHardPartonListConnected() {
    return GetHardPartonListConnected;
  }

 private:
  bool GetHardPartonListConnected;  ///< Flag indicating if GetHardPartonList is
                                    ///< connected.
  vector<shared_ptr<Parton>> hp;    ///< Hard parton list managed by this class.
};

}  // end namespace Jetscape

#endif