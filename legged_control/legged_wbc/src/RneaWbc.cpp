#include <pinocchio/fwd.hpp>

#include "legged_wbc/RneaWbc.h"

#include <ocs2_core/misc/LoadData.h>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <ros/console.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/autodiff/casadi.hpp>

namespace legged {

vector_t RneaWbc::update(const vector_t& stateDesired, const vector_t& inputDesired,
                          const vector_t& rbdStateMeasured, size_t mode, scalar_t period) {
  WbcBase::update(stateDesired, inputDesired, rbdStateMeasured, mode, period);

  return solveNlp(qMeasured_, vMeasured_, contactFlag_, stateDesired, inputDesired, period);
}

void RneaWbc::buildNlp() {
  using ADScalar = casadi::SX;
  using ADModel  = pinocchio::ModelTpl<ADScalar>;
  using ADData   = pinocchio::DataTpl<ADScalar>;
  using ADVec    = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;
  using ADVec3   = Eigen::Matrix<ADScalar, 3, 1>;
  using ADMat3   = Eigen::Matrix<ADScalar, 3, 3>;
  using ADForce  = pinocchio::ForceTpl<ADScalar>;

  const auto& model = pinocchioInterfaceMeasured_.getModel();
  ADModel ad_model  = model.template cast<ADScalar>();
  ADData  ad_data(ad_model);

  const int nq = model.nq;  // == info_.generalizedCoordinatesNum (Euler-angle floating base)
  const int nv = model.nv;
  const int nj = static_cast<int>(info_.actuatedDofNum);
  const int nc = static_cast<int>(info_.numThreeDofContacts);
  const int nu = nv + 3 * nc + nj;  // decision: [a(nv), Fc(3*nc), tau_j(nj)]

  // ── NLP parameters:
  //    p = [q(nq); v(nv); H_flat(nu²); g_obj(nu); J_nc_flat(3*nc*nv); dJv_nc(3*nc)]
  //    H_flat / g_obj  : weighted task cost matrices (computed per step)
  //    J_nc_flat       : contact Jacobian, 3*nc×nv, column-major;
  //                      swing-foot rows are zero so their constraint degenerates to 0=0
  //    dJv_nc          : Jdot * v bias for no-contact-motion, also zeroed for swing feet
  ADScalar cs_q      = ADScalar::sym("q",        nq);
  ADScalar cs_v      = ADScalar::sym("v",        nv);
  ADScalar cs_H_flat = ADScalar::sym("H_flat",   nu * nu);
  ADScalar cs_g_obj  = ADScalar::sym("g_obj",    nu);
  ADScalar cs_J_nc_flat = ADScalar::sym("J_nc_flat", 3 * nc * nv);
  ADScalar cs_dJv_nc    = ADScalar::sym("dJv_nc",    3 * nc);
  ADScalar cs_p = ADScalar::vertcat({cs_q, cs_v, cs_H_flat, cs_g_obj,
                                     cs_J_nc_flat, cs_dJv_nc});

  // ── NLP decision variables: u = [a(nv), Fc(3*nc), tau_j(nj)]
  ADScalar cs_u   = ADScalar::sym("u",  nu);
  ADScalar cs_a   = cs_u(casadi::Slice(0, nv));
  ADScalar cs_fc  = cs_u(casadi::Slice(nv, nv + 3 * nc));
  ADScalar cs_tau = cs_u(casadi::Slice(nv + 3 * nc, nu));

  // ── Map CasADi symbolic columns → Eigen vectors so pinocchio can consume them
  ADVec q_ad(nq), v_ad(nv), a_ad(nv);
  q_ad = Eigen::Map<ADVec>(static_cast<std::vector<ADScalar>>(cs_q).data(), nq, 1);
  v_ad = Eigen::Map<ADVec>(static_cast<std::vector<ADScalar>>(cs_v).data(), nv, 1);
  a_ad = Eigen::Map<ADVec>(static_cast<std::vector<ADScalar>>(cs_a).data(), nv, 1);

  // ── Forward kinematics: populates oMi (joint placements) and oMf (frame placements)
  pinocchio::forwardKinematics(ad_model, ad_data, q_ad);
  pinocchio::updateFramePlacements(ad_model, ad_data);

  // ── Build f_ext: external wrench on each joint expressed in joint local frame.
  //    Contact forces Fc are in world frame; the transformation is:
  //      f_local = R_world_joint^T * f_world
  //    where R_world_joint = ad_data.oMi[joint_id].rotation().
  pinocchio::container::aligned_vector<ADForce> f_ext(
      static_cast<size_t>(ad_model.njoints), ADForce::Zero());

  for (int i = 0; i < nc; i++) {
    const auto frame_idx = info_.endEffectorFrameIndices[i];
    const auto joint_id  = ad_model.frames[frame_idx].parent;

    // Symbolic rotation of joint frame in world frame (3×3)
    const ADMat3 R = ad_data.oMi[joint_id].rotation();

    // Contact force for foot i in world frame
    ADVec3 fc_world;
    for (int k = 0; k < 3; k++) fc_world(k) = cs_fc(3 * i + k);

    // Point contact → linear force only, no torque at contact
    f_ext[joint_id].linear()  = R.transpose() * fc_world;

    ADVec3 r_world = ad_data.oMf[frame_idx].translation() - ad_data.oMi[joint_id].translation();
    ADVec3 r_local = R.transpose() * r_world;
    f_ext[joint_id].angular() = r_local.cross(R.transpose() * fc_world);
  }

  // ── Symbolic RNEA: tau_rnea = M*a + C(q,v) - g(q) - J^T * f_ext
  //    Result is stored in ad_data.tau (Eigen vector of ADScalar, size nv).
  pinocchio::rnea(ad_model, ad_data, q_ad, v_ad, a_ad, f_ext);

  ADScalar cs_tau_rnea(nv, 1);
  pinocchio::casadi::copy(ad_data.tau, cs_tau_rnea);

  // ── No-contact-motion: J_nc * a + dJv_nc = 0
  //    Swing-foot rows of J_nc_flat are zero, so their rows degenerate to 0 = 0.
  ADScalar cs_J_nc_mat = ADScalar::reshape(cs_J_nc_flat, 3 * nc, nv);
  ADScalar cs_nc_res   = ADScalar::mtimes(cs_J_nc_mat, cs_a) + cs_dJv_nc;

  // ── Friction cone: mu² * Fc_z² - Fc_x² - Fc_y² >= 0  (one per foot)
  //    Swing feet are already fixed to Fc=0 by box constraints, so the
  //    constraint degenerates to 0 >= 0 and causes no numerical issue.
  const double mu = frictionCoeff_;
  std::vector<ADScalar> friction_exprs;
  friction_exprs.reserve(nc);
  for (int i = 0; i < nc; i++) {
    ADScalar Fc_x = cs_fc(3 * i);
    ADScalar Fc_y = cs_fc(3 * i + 1);
    ADScalar Fc_z = cs_fc(3 * i + 2);
    friction_exprs.push_back(mu * mu * Fc_z * Fc_z - Fc_x * Fc_x - Fc_y * Fc_y);
  }
  ADScalar cs_friction = ADScalar::vertcat(friction_exprs);

  // ── Constraints g:
  //    [0 : 6]            equality  — floating base: tau_rnea = 0
  //    [6 : nv]           equality  — actuated joints: tau_rnea = tau_j
  //    [nv : nv+3*nc]     equality  — contact: J_nc * a = -dJv_nc
  //    [nv+3*nc : ...]    inequality (>= 0) — friction cone per foot
  ADScalar cs_g = ADScalar::vertcat({
      cs_tau_rnea(casadi::Slice(0, 6)),
      cs_tau_rnea(casadi::Slice(6, nv)) - cs_tau,
      cs_nc_res,
      cs_friction
  });

  // Weighted task objective: 0.5 * u^T * H * u + g_obj^T * u
  // H and g_obj are numeric parameters computed per time-step from the task Jacobians.
  // A small regularization term keeps the Hessian positive-definite when H is rank-deficient.
  ADScalar cs_H_mat = ADScalar::reshape(cs_H_flat, nu, nu);
  ADScalar obj = 0.5 * ADScalar::dot(cs_u, ADScalar::mtimes(cs_H_mat, cs_u))
               + ADScalar::dot(cs_g_obj, cs_u)
               + 1e-6 * ADScalar::dot(cs_u, cs_u);

  casadi::SXDict nlp = {{"x", cs_u}, {"f", obj}, {"g", cs_g}, {"p", cs_p}};
  casadi::Dict opts = {
      {"print_time", false},
      {"ipopt.print_level", 0},
      {"ipopt.max_iter", 50},
      {"ipopt.warm_start_init_point", "yes"},
      {"ipopt.warm_start_bound_push", 1e-9},
      {"ipopt.warm_start_bound_frac", 1e-9},
      {"ipopt.warm_start_slack_bound_frac", 1e-9},
      {"ipopt.warm_start_slack_bound_push", 1e-9},
      {"ipopt.warm_start_mult_bound_push", 1e-9},
  };
  nlpSolver_ = casadi::nlpsol("wbc_rnea", "ipopt", nlp, opts);
}

Task RneaWbc::formulateWeightedTasks(const vector_t& stateDesired, const vector_t& inputDesired,
                                      scalar_t period) {
  return formulateSwingLegTask() * weightSwingLeg_
       + formulateBaseAccelTask(stateDesired, inputDesired, period) * weightBaseAccel_
       + formulateContactForceTask(inputDesired) * weightContactForce_;
}

vector_t RneaWbc::solveNlp(const vector_t& q, const vector_t& v,
                           const contact_flag_t& contactFlag,
                           const vector_t& stateDesired,
                           const vector_t& inputDesired,
                           scalar_t period) {
  const int nq = static_cast<int>(q.size());
  const int nv = static_cast<int>(v.size());
  const int nj = static_cast<int>(info_.actuatedDofNum);
  const int nc = static_cast<int>(info_.numThreeDofContacts);
  const int nu = nv + 3 * nc + nj;

  Task task = formulateWeightedTasks(stateDesired, inputDesired, period);
  matrix_t H_task = task.a_.transpose() * task.a_;
  vector_t g_task = -(task.a_.transpose() * task.b_);

  // No-contact-motion: J_nc * a = -dJv_nc
  // Swing-foot rows are zeroed so their constraint is trivially 0 = 0.
  matrix_t J_nc   = matrix_t::Zero(3 * nc, nv);
  vector_t dJv_nc = vector_t::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    if (contactFlag[i]) {
      J_nc.block(3 * i, 0, 3, nv) = j_.block(3 * i, 0, 3, nv);
      dJv_nc.segment(3 * i, 3)    = dj_.block(3 * i, 0, 3, nv) * vMeasured_;
    }
  }

  // Pack NLP parameters: p = [q; v; H_flat; g_obj; J_nc_flat; dJv_nc]
  // All matrices use column-major order (Eigen default == CasADi reshape default)
  const int p_size = nq + nv + nu * nu + nu + 3 * nc * nv + 3 * nc;
  casadi::DM p_dm(p_size, 1);
  int off = 0;
  for (int i = 0; i < nq; i++)       p_dm(off++)  = q(i);
  for (int i = 0; i < nv; i++)       p_dm(off++)  = v(i);
  for (int i = 0; i < nu * nu; i++)  p_dm(off++)  = H_task.data()[i];
  for (int i = 0; i < nu; i++)       p_dm(off++)  = g_task(i);
  for (int i = 0; i < 3*nc*nv; i++) p_dm(off++)  = J_nc.data()[i];
  for (int i = 0; i < 3*nc; i++)    p_dm(off++)  = dJv_nc(i);

  // Equality bounds: RNEA (6+nj rows) + no-contact-motion (3*nc rows), all = 0
  // Inequality bounds: friction cone (nc rows) in [0, +inf)
  const int n_eq   = 6 + nj + 3 * nc;
  const int n_ineq = nc;
  casadi::DM lbg = casadi::DM::zeros(n_eq + n_ineq, 1);
  casadi::DM ubg = casadi::DM::zeros(n_eq + n_ineq, 1);
  for (int i = 0; i < n_ineq; i++) ubg(n_eq + i) = 1e20;

  // Box constraints on u
  casadi::DM lbu = -1e9 * casadi::DM::ones(nu, 1);
  casadi::DM ubu =  1e9 * casadi::DM::ones(nu, 1);

  // Torque limits
  const int lim_size = static_cast<int>(torqueLimits_.size());
  for (int j = 0; j < nj; j++) {
    const double lim = torqueLimits_(j % lim_size);
    lbu(nv + 3 * nc + j) = -lim;
    ubu(nv + 3 * nc + j) =  lim;
  }
  // Contact force feasibility:
  //   contact foot → normal force must be upward (Fc_z >= 0)
  //   swing foot   → zero force (Fc = 0)
  for (int i = 0; i < nc; i++) {
    if (contactFlag[i]) {
      lbu(nv + 3 * i + 2) = 0.0;   // Fc_z >= 0
    } else {
      lbu(nv + 3 * i)     = 0.0;  ubu(nv + 3 * i)     = 0.0;
      lbu(nv + 3 * i + 1) = 0.0;  ubu(nv + 3 * i + 1) = 0.0;
      lbu(nv + 3 * i + 2) = 0.0;  ubu(nv + 3 * i + 2) = 0.0;
    }
  }

  casadi::DMDict args = {{"lbg", lbg}, {"ubg", ubg}, {"lbx", lbu}, {"ubx", ubu}, {"p", p_dm}};
  if (hasWarmStart_) {
    // Full warm-start: primal + dual variables from a previous successful solve
    args["x0"]     = prevX_;
    args["lam_g0"] = prevLamG_;
    args["lam_x0"] = prevLamX_;
  } else if (hasExternalGuess_) {
    // Primal-only hint from WeightedWbc — no dual variables available,
    // but far better than starting from zero.
    args["x0"] = prevX_;
  } else {
    args["x0"] = casadi::DM::zeros(nu, 1);
  }

  casadi::DMDict sol = nlpSolver_(args);

  // ── Solver diagnostics (throttled to 1 Hz) ─────────────────────────────
  const auto& stats       = nlpSolver_.stats();
  const std::string status = stats.at("return_status");
  const int    iters      = static_cast<int>(stats.at("iter_count"));
  const double obj        = static_cast<double>(casadi::DM(sol.at("f")));

  // RNEA equality residual: g should be zero; report L-inf norm
  casadi::DM g_val = sol.at("g");
  double g_inf = 0.0;
  for (int i = 0; i < g_val.size1(); i++)
    g_inf = std::max(g_inf, std::abs(static_cast<double>(g_val(i))));

  if (status != "Solve_Succeeded" && status != "Solved_To_Acceptable_Level") {
    ROS_WARN_STREAM_THROTTLE(1.0,
        "[RneaWbc] IPOPT status: " << status
        << "  iters=" << iters << "  obj=" << obj
        << "  |g|_inf=" << g_inf);
  } else {
    ROS_INFO_STREAM_THROTTLE(1.0,
        "[RneaWbc] IPOPT status: " << status
        << "  iters=" << iters << "  obj=" << obj
        << "  |g|_inf=" << g_inf);
  }
  // ───────────────────────────────────────────────────────────────────────

  if (status == "Solve_Succeeded" || status == "Solved_To_Acceptable_Level") {
    // Good solve: save full primal + dual for next iteration
    prevX_        = sol.at("x");
    prevLamG_     = sol.at("lam_g");
    prevLamX_     = sol.at("lam_x");
    hasWarmStart_     = true;
    hasExternalGuess_ = false;
  } else {
    // Bad solve: discard this result, fall back to external guess next time
    prevX_ = sol.at("x");
    hasWarmStart_ = false;
  }

  vector_t result(nu);
  for (int i = 0; i < nu; i++) result(i) = static_cast<double>(prevX_(i));
  return result;
}

void RneaWbc::setExternalInitialGuess(const vector_t& u0) {
  const int nu = static_cast<int>(u0.size());
  prevX_ = casadi::DM(nu, 1);
  for (int i = 0; i < nu; i++) prevX_(i) = u0(i);
  hasExternalGuess_ = true;
}

void RneaWbc::loadTasksSetting(const std::string& taskFile, bool verbose) {
  WbcBase::loadTasksSetting(taskFile, verbose);

  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile, pt);
  std::string prefix = "weight.";
  if (verbose) {
    std::cerr << "\n #### RneaWbc weight:";
    std::cerr << "\n #### =============================================================================\n";
  }
  loadData::loadPtreeValue(pt, weightSwingLeg_,     prefix + "swingLeg",     verbose);
  loadData::loadPtreeValue(pt, weightBaseAccel_,    prefix + "baseAccel",    verbose);
  loadData::loadPtreeValue(pt, weightContactForce_, prefix + "contactForce", verbose);

  buildNlp();
}

}  // namespace legged
