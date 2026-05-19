#pragma once
#include "legged_wbc/WbcBase.h"
#include <casadi/casadi.hpp>
#include <cmath>
#include <vector>

namespace legged {

class RneaWbc : public WbcBase {
 public:
  using WbcBase::WbcBase;

  vector_t update(const vector_t& stateDesired, const vector_t& inputDesired, const vector_t& rbdStateMeasured, size_t mode,
                  scalar_t period) override;

  void loadTasksSetting(const std::string& taskFile, bool verbose) override;

  void setExternalInitialGuess(const vector_t& u0);

 private:
  Task formulateWeightedTasks(const vector_t& stateDesired, const vector_t& inputDesired, scalar_t period);

  void buildNlp();
  vector_t solveNlp(const vector_t& q, const vector_t& v,
                    const contact_flag_t& contactFlag,
                    const vector_t& stateDesired, const vector_t& inputDesired,
                    scalar_t period);

  casadi::Function nlpSolver_;

  casadi::DM prevX_, prevLamG_, prevLamX_;
  bool hasWarmStart_     = false;
  bool hasExternalGuess_ = false;

  // Planning horizon parameters — loaded from task.info [rneaWbc] at startup
  int    numNodes_     = 1;
  int    tauNodes_     = 1;
  double gamma_        = 1.2;
  double totalHorizon_ = 0.56;

  // Cached NLP dimensions and per-node time steps set by buildNlp()
  int nz_  = 0;
  int ndx_ = 0;  // delta-state dimension (2*nv); offset of U[0] in the Fatrop z vector
  std::vector<double> dts_;

  scalar_t weightSwingLeg_{}, weightBaseAccel_{}, weightContactForce_{};
};

}  // namespace legged