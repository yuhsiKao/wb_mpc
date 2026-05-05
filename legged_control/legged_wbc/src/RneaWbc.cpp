#include <pinocchio/fwd.hpp>

#include "legged_wbc/RneaWbc.h"

#include <ocs2_core/misc/LoadData.h>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/autodiff/casadi.hpp>

namespace legged {

vector_t RneaWbc::update(const vector_t& stateDesired, const vector_t& inputDesired,
                          const vector_t& rbdStateMeasured, size_t mode, scalar_t period) {
  WbcBase::update(stateDesired, inputDesired, rbdStateMeasured, mode, period);

  if (!nlpBuilt_) { buildNlp(); nlpBuilt_ = true; }

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

  // ── NLP parameters: p = [q(nq); v(nv); H_flat(nu*nu); g_obj(nu)]
  //    H_flat and g_obj encode the weighted task cost: 0.5 u^T H u + g_obj^T u
  ADScalar cs_q     = ADScalar::sym("q",      nq);
  ADScalar cs_v     = ADScalar::sym("v",      nv);
  ADScalar cs_H_flat = ADScalar::sym("H_flat", nu * nu);
  ADScalar cs_g_obj  = ADScalar::sym("g_obj",  nu);
  ADScalar cs_p = ADScalar::vertcat({cs_q, cs_v, cs_H_flat, cs_g_obj});

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

  // ── Forward kinematics to get symbolic joint placements (populates ad_data.oMi)
  pinocchio::forwardKinematics(ad_model, ad_data, q_ad);

  // ── Build f_ext: external wrench on each joint expressed in joint local frame.
  //    Contact forces Fc are in world frame; the transformation is:
  //      f_local = R_world_joint^T * f_world
  //    where R_world_joint = ad_data.oMi[joint_id].rotation().
  pinocchio::container::aligned_vector<ADForce> f_ext(
      static_cast<size_t>(ad_model.njoints), ADForce::Zero());

  for (int i = 0; i < nc; i++) {
    const auto frame_idx = info_.endEffectorFrameIndices[i];
    const auto joint_id  = ad_model.frames[frame_idx].parentJoint;

    // Symbolic rotation of joint frame in world frame (3×3)
    const ADMat3 R = ad_data.oMi[joint_id].rotation();

    // Contact force for foot i in world frame
    ADVec3 fc_world;
    for (int k = 0; k < 3; k++) fc_world(k) = cs_fc(3 * i + k);

    // Point contact → linear force only, no torque at contact
    f_ext[joint_id].linear()  = R.transpose() * fc_world;
    f_ext[joint_id].angular() = ADVec3::Zero();
  }

  // ── Symbolic RNEA: tau_rnea = M*a + C(q,v) - g(q) - J^T * f_ext
  //    Result is stored in ad_data.tau (Eigen vector of ADScalar, size nv).
  pinocchio::rnea(ad_model, ad_data, q_ad, v_ad, a_ad, f_ext);

  ADScalar cs_tau_rnea(nv, 1);
  pinocchio::casadi::copy(ad_data.tau, cs_tau_rnea);

  // ── Equality constraints g == 0:
  //    [0:6]  floating base (unactuated) → tau_rnea must vanish
  //    [6:nv] actuated joints            → tau_rnea must equal decision tau_j
  ADScalar cs_g = ADScalar::vertcat({
      cs_tau_rnea(casadi::Slice(0, 6)),
      cs_tau_rnea(casadi::Slice(6, nv)) - cs_tau
  });

  // Weighted task objective: 0.5 * u^T * H * u + g_obj^T * u
  // H and g_obj are numeric parameters computed per time-step from the task Jacobians.
  // A small regularization term keeps the Hessian positive-definite when H is rank-deficient.
  ADScalar cs_H_mat = ADScalar::reshape(cs_H_flat, nu, nu);
  ADScalar obj = 0.5 * ADScalar::dot(cs_u, ADScalar::mtimes(cs_H_mat, cs_u))
               + ADScalar::dot(cs_g_obj, cs_u)
               + 1e-6 * ADScalar::dot(cs_u, cs_u);

  casadi::SXDict nlp = {{"x", cs_u}, {"f", obj}, {"g", cs_g}, {"p", cs_p}};
  casadi::Dict   opts = {{"ipopt.print_level", 0}, {"ipopt.max_iter", 50}};
  nlpSolver_ = casadi::nlpsol("wbc_rnea", "ipopt", nlp, opts);
}

vector_t RneaWbc::solveNlp(const vector_t& q, const vector_t& v,
                             const contact_flag_t& /*contactFlag*/,
                             const vector_t& stateDesired,
                             const vector_t& inputDesired,
                             scalar_t period) {
  const int nq = static_cast<int>(q.size());
  const int nv = static_cast<int>(v.size());
  const int nj = static_cast<int>(info_.actuatedDofNum);
  const int nc = static_cast<int>(info_.numThreeDofContacts);
  const int nu = nv + 3 * nc + nj;

  // Compute weighted task cost matrices (same tasks as WeightedWbc::formulateWeightedTasks)
  Task task = formulateSwingLegTask()    * weightSwingLeg_
            + formulateBaseAccelTask(stateDesired, inputDesired, period) * weightBaseAccel_
            + formulateContactForceTask(inputDesired) * weightContactForce_;
  matrix_t H_task = task.a_.transpose() * task.a_;           // (nu x nu)
  vector_t g_task = -(task.a_.transpose() * task.b_);        // (nu)

  // Pack NLP parameters: p = [q(nq); v(nv); H_flat(nu*nu); g_obj(nu)]
  // H_flat uses column-major order (matches both Eigen and CasADi reshape defaults)
  casadi::DM p_dm(nq + nv + nu * nu + nu, 1);
  for (int i = 0; i < nq; i++) p_dm(i)      = q(i);
  for (int i = 0; i < nv; i++) p_dm(nq + i) = v(i);
  for (int i = 0; i < nu * nu; i++) p_dm(nq + nv + i)          = H_task.data()[i];
  for (int i = 0; i < nu; i++)      p_dm(nq + nv + nu * nu + i) = g_task(i);

  casadi::DM u0  = casadi::DM::zeros(nu, 1);   // initial guess (cold start)
  casadi::DM lbg = casadi::DM::zeros(6 + nj, 1);
  casadi::DM ubg = casadi::DM::zeros(6 + nj, 1);

  casadi::DM lbu = -1e9 * casadi::DM::ones(nu, 1);
  casadi::DM ubu =  1e9 * casadi::DM::ones(nu, 1);
  const int lim_size = static_cast<int>(torqueLimits_.size());
  for (int j = 0; j < nj; j++) {
    const double lim = torqueLimits_(j % lim_size);
    lbu(nv + 3 * nc + j) = -lim;
    ubu(nv + 3 * nc + j) =  lim;
  }

  casadi::DMDict sol = nlpSolver_(casadi::DMDict{
      {"x0", u0}, {"lbg", lbg}, {"ubg", ubg},
      {"lbx", lbu}, {"ubx", ubu}, {"p", p_dm}
  });

  casadi::DM u_sol = sol.at("x");
  vector_t result(nu);
  for (int i = 0; i < nu; i++) result(i) = static_cast<double>(u_sol(i));
  return result;
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
}

}  // namespace legged
