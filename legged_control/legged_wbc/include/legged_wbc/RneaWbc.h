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

 private:
  void buildNlp();
  vector_t solveNlp(const vector_t& q, const vector_t& v,
                    const contact_flag_t& contactFlag,
                    const vector_t& stateDesired, const vector_t& inputDesired,
                    scalar_t period);

  casadi::Function nlpSolver_;
  bool nlpBuilt_ = false;

  scalar_t weightSwingLeg_{}, weightBaseAccel_{}, weightContactForce_{};
};

}  // namespace legged