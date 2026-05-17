#pragma once
#include "legged_wbc/WbcBase.h"
#include <casadi/casadi.hpp>

namespace legged {

class RneaWbc : public WbcBase {
 public:
  using WbcBase::WbcBase;

  vector_t update(const vector_t& stateDesired, const vector_t& inputDesired, const vector_t& rbdStateMeasured, size_t mode,
                  scalar_t period) override;

  void loadTasksSetting(const std::string& taskFile, bool verbose) override;

  // Feed an external primal guess (e.g. from WeightedWbc) to bootstrap the
  // first NLP solve.  Call before the first update().
  void setExternalInitialGuess(const vector_t& u0);

 private:
  Task formulateWeightedTasks(const vector_t& stateDesired, const vector_t& inputDesired, scalar_t period);

  void buildNlp();
  vector_t solveNlp(const vector_t& q, const vector_t& v,
                    const contact_flag_t& contactFlag,
                    const vector_t& stateDesired, const vector_t& inputDesired,
                    scalar_t period);

  casadi::Function nlpSolver_;

  // Warm-start: cache primal and dual variables from the previous solve.
  // Dual variables (lam_g, lam_x) are only valid after a successful solve;
  // externalGuess_ holds a primal-only hint (no dual) from an external solver.
  casadi::DM prevX_, prevLamG_, prevLamX_;
  bool hasWarmStart_     = false;
  bool hasExternalGuess_ = false;

  scalar_t weightSwingLeg_{}, weightBaseAccel_{}, weightContactForce_{};
};

}  // namespace legged