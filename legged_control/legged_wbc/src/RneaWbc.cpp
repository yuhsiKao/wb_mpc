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

  const int nq  = model.nq;
  const int nv  = model.nv;
  const int nj  = static_cast<int>(info_.actuatedDofNum);
  const int nc  = static_cast<int>(info_.numThreeDofContacts);
  const int nu_full = nv + 3 * nc + nj;  // k < tauNodes_: [a(nv), Fc(3*nc), τ_j(nj)]
  const int nu_red  = nv + 3 * nc;       // k ≥ tauNodes_: [a(nv), Fc(3*nc)]
  const int nu      = nu_full;           // alias for parameter-vector sizing (always full)
  const int ndx     = 2 * nv;           // delta state: [δq(nv), δv(nv)]
  const int N       = numNodes_;
  const double mu = frictionCoeff_;

  // ── Precompute geometric time steps: δt_k = γ^k * δt_0
  //    Σ δt_k = δt_0 * (γ^N − 1)/(γ−1) = T_total  →  δt_0 solved analytically
  dts_.resize(N);
  {
    double dt0;
    if (std::abs(gamma_ - 1.0) < 1e-9)
        dt0 = totalHorizon_ / N;
    else
        dt0 = totalHorizon_ * (gamma_ - 1.0) / (std::pow(gamma_, N) - 1.0);
    for (int k = 0; k < N; k++)
        dts_[k] = dt0 * std::pow(gamma_, k);
  }

  // ── Parameter vector layout:
  //    [q0(nq), v0(nv),  <node-0-params>, ..., <node-(N-1)-params>]
  //    per-node: [H_flat(nu²), g_obj(nu), J_nc_flat(3*nc*nv), dJv_nc(3*nc)]
  //    (δt_k are baked as numeric constants — no need to pass them at each solve)
  const int p_node  = nu * nu + nu + 3 * nc * nv + 3 * nc;
  const int p_total = nq + nv + N * p_node;
  ADScalar cs_p = ADScalar::sym("p", p_total);

  int p_off = 0;
  ADScalar cs_q0 = cs_p(casadi::Slice(p_off, p_off + nq)); p_off += nq;
  ADScalar cs_v0 = cs_p(casadi::Slice(p_off, p_off + nv)); p_off += nv;

  std::vector<ADScalar> cs_H_flat(N), cs_g_obj(N), cs_J_nc_flat(N), cs_dJv_nc(N);
  for (int k = 0; k < N; k++) {
    cs_H_flat[k]    = cs_p(casadi::Slice(p_off, p_off + nu * nu));      p_off += nu * nu;
    cs_g_obj[k]     = cs_p(casadi::Slice(p_off, p_off + nu));           p_off += nu;
    cs_J_nc_flat[k] = cs_p(casadi::Slice(p_off, p_off + 3 * nc * nv)); p_off += 3 * nc * nv;
    cs_dJv_nc[k]    = cs_p(casadi::Slice(p_off, p_off + 3 * nc));      p_off += 3 * nc;
  }

  // ── Decision variable layout (Fatrop stage-wise interleaved):
  //    z = [DX[0](ndx), U[0](nu_0), DX[1](ndx), U[1](nu_1), ..., U[N-1](nu_{N-1}), DX[N](ndx)]
  //    k < tauNodes_ : U[k] = [a(nv), Fc(3*nc), τ_j(nj)]  size nu_full
  //    k ≥ tauNodes_ : U[k] = [a(nv), Fc(3*nc)]            size nu_red
  //    DX[k] = [δq(nv), δv(nv)]                            size ndx
  //    DX[0] is included as a decision variable and pinned to 0 via box constraints
  //    in solveNlp(), giving Fatrop a clean [state, control] pair at every stage.
  ndx_ = ndx;
  nz_ = (N + 1) * ndx
      + std::min(N, tauNodes_) * nu_full
      + std::max(0, N - tauNodes_) * nu_red;
  ADScalar cs_z = ADScalar::sym("z", nz_);

  std::vector<ADScalar> U(N), DX(N + 1);
  int z_off = 0;
  DX[0] = cs_z(casadi::Slice(z_off, z_off + ndx));  z_off += ndx;  // pinned to 0 at solve time
  for (int k = 0; k < N; k++) {
    const int nu_k = (k < tauNodes_) ? nu_full : nu_red;
    U[k]    = cs_z(casadi::Slice(z_off, z_off + nu_k));  z_off += nu_k;
    DX[k+1] = cs_z(casadi::Slice(z_off, z_off + ndx));   z_off += ndx;
  }

  // ── Map q0, v0 to Eigen so pinocchio can integrate from them
  ADVec q0_ad(nq), v0_ad(nv);
  for (int i = 0; i < nq; i++) q0_ad(i) = cs_q0(i);
  for (int i = 0; i < nv; i++) v0_ad(i) = cs_v0(i);

  // ── Build NLP per shooting node
  ADScalar obj = ADScalar::zeros(1);
  std::vector<ADScalar> g_list;
  g_list.reserve(N);

  // ── FK at q0 once; all nodes linearise dynamics at the measured configuration.
  //    pinocchio::integrate is intentionally avoided — it triggers boost::bad_get
  //    via boost::variant dispatch when used with CasADi ADScalar joint types.
  ADData ad_data_fk(ad_model);
  pinocchio::forwardKinematics(ad_model, ad_data_fk, q0_ad);
  pinocchio::updateFramePlacements(ad_model, ad_data_fk);

  for (int k = 0; k < N; k++) {
    // Extract delta state at node k: DX[k] = [δq_k(nv), δv_k(nv)]
    ADScalar dq_k = DX[k](casadi::Slice(0, nv));   // (nv×1) — used only in shooting gap
    ADScalar dv_k = DX[k](casadi::Slice(nv, ndx)); // (nv×1) — feeds absolute v_k

    // ── Absolute velocity: v_k = v0 + δv_k
    ADVec v_k_ad(nv);
    for (int i = 0; i < nv; i++) v_k_ad(i) = v0_ad(i) + dv_k(i);

    // ── Extract inputs at node k (τ_j only for tau nodes)
    ADScalar cs_a_k  = U[k](casadi::Slice(0, nv));
    ADScalar cs_fc_k = U[k](casadi::Slice(nv, nv + 3 * nc));
    ADVec a_k_ad(nv);
    for (int i = 0; i < nv; i++) a_k_ad(i) = cs_a_k(i);

    // ── External wrenches at q0 (frozen linearisation point)
    pinocchio::container::aligned_vector<ADForce> f_ext(
        static_cast<size_t>(ad_model.njoints), ADForce::Zero());
    for (int i = 0; i < nc; i++) {
      const auto frame_idx = info_.endEffectorFrameIndices[i];
      const auto joint_id  = ad_model.frames[frame_idx].parent;
      const ADMat3 R = ad_data_fk.oMi[joint_id].rotation();
      ADVec3 fc_world;
      for (int j = 0; j < 3; j++) fc_world(j) = cs_fc_k(3 * i + j);
      f_ext[joint_id].linear()  = R.transpose() * fc_world;
      ADVec3 r_world = ad_data_fk.oMf[frame_idx].translation() - ad_data_fk.oMi[joint_id].translation();
      ADVec3 r_local = R.transpose() * r_world;
      f_ext[joint_id].angular() = r_local.cross(R.transpose() * fc_world);
    }

    // ── RNEA at (q0, v_k, a_k)
    ADData ad_data(ad_model);
    pinocchio::rnea(ad_model, ad_data, q0_ad, v_k_ad, a_k_ad, f_ext);
    ADScalar cs_tau_rnea(nv, 1);
    pinocchio::casadi::copy(ad_data.tau, cs_tau_rnea);

    // ── No-contact-motion: J_nc_k * a_k + dJv_nc_k = 0
    ADScalar cs_J_nc_mat = ADScalar::reshape(cs_J_nc_flat[k], 3 * nc, nv);
    ADScalar cs_nc_res   = ADScalar::mtimes(cs_J_nc_mat, cs_a_k) + cs_dJv_nc[k];

    // ── Friction cone: mu²*Fz² - Fx² - Fy² >= 0
    std::vector<ADScalar> frict;
    frict.reserve(nc);
    for (int i = 0; i < nc; i++) {
      ADScalar Fx = cs_fc_k(3 * i), Fy = cs_fc_k(3 * i + 1), Fz = cs_fc_k(3 * i + 2);
      frict.push_back(mu * mu * Fz * Fz - Fx * Fx - Fy * Fy);
    }
    ADScalar cs_friction_k = ADScalar::vertcat(frict);

    // ── Shooting gap: DX[k+1] = DX[k] + [v_k·δt_k ; a_k·δt_k]
    //    δt_k is baked in as a numeric constant (geometric schedule)
    //    v_k in CasADi: cs_v0 + dv_k  (avoids Eigen round-trip)
    const double dt_k = dts_[k];
    ADScalar v_k_cs  = cs_v0 + dv_k;                                    // (nv×1)
    ADScalar dq_pred = dq_k + v_k_cs * dt_k;                            // (nv×1)
    ADScalar dv_pred = dv_k + cs_a_k * dt_k;                            // (nv×1)
    ADScalar gap     = DX[k+1] - ADScalar::vertcat({dq_pred, dv_pred}); // (ndx×1)

    // ── Joint RNEA constraint (tauNodes_-aware)
    //    k < tauNodes_ : τ_rnea[6:] == τ_j  (τ_j is a decision variable)
    //    k ≥ tauNodes_ : τ_rnea[6:] == 0    (no torque variable at reduced nodes)
    ADScalar joint_eq = (k < tauNodes_)
        ? cs_tau_rnea(casadi::Slice(6, nv)) - U[k](casadi::Slice(nv + 3 * nc, nu_full))
        : cs_tau_rnea(casadi::Slice(6, nv));

    g_list.push_back(gap);
    g_list.push_back(cs_tau_rnea(casadi::Slice(0, 6)));
    g_list.push_back(joint_eq);
    g_list.push_back(cs_nc_res);
    g_list.push_back(cs_friction_k);

    // ── Objective (tauNodes_-aware)
    //    k < tauNodes_ : full H (nu_full × nu_full)
    //    k ≥ tauNodes_ : top-left nu_red × nu_red block (τ_j columns dropped)
    ADScalar cs_H_full = ADScalar::reshape(cs_H_flat[k], nu_full, nu_full);
    if (k < tauNodes_) {
      obj += 0.5 * ADScalar::dot(U[k], ADScalar::mtimes(cs_H_full, U[k]))
           + ADScalar::dot(cs_g_obj[k], U[k])
           + 1e-6 * ADScalar::dot(U[k], U[k]);
    } else {
      ADScalar cs_H_red = cs_H_full(casadi::Slice(0, nu_red), casadi::Slice(0, nu_red));
      ADScalar cs_g_red = cs_g_obj[k](casadi::Slice(0, nu_red));
      obj += 0.5 * ADScalar::dot(U[k], ADScalar::mtimes(cs_H_red, U[k]))
           + ADScalar::dot(cs_g_red, U[k])
           + 1e-6 * ADScalar::dot(U[k], U[k]);
    }
  }

  ADScalar cs_g = ADScalar::vertcat(g_list);

  // ── Fatrop structure: every stage has ndx states; controls and lincon may vary per stage.
  //    nx  : state sizes,  length N+1 (terminal stage is DX[N], no controls)
  //    nu  : control sizes, length N+1 (terminal entry = 0 by CasADi 3.6 convention)
  //    ng  : non-dynamic constraint count per stage, length N+1
  //          gap constraints (ndx rows each) are detected automatically from the structure.
  //          lincon per stage = RNEA(nv) + contact(3*nc) + friction(nc) = nv + 4*nc
  const int ng_stage = nv + 4 * nc;
  std::vector<casadi_int> nx_v(N + 1, ndx);
  std::vector<casadi_int> nu_v(N + 1, 0);
  std::vector<casadi_int> ng_v(N + 1, 0);
  for (int k = 0; k < N; k++) {
    nu_v[k] = (k < tauNodes_) ? nu_full : nu_red;
    ng_v[k] = ng_stage;
  }

  casadi::SXDict nlp = {{"x", cs_z}, {"f", obj}, {"g", cs_g}, {"p", cs_p}};
  casadi::Dict fatrop_opts;
  fatrop_opts["print_level"] = casadi_int(0);
  fatrop_opts["max_iter"]    = casadi_int(100);
  casadi::Dict opts = {
      {"print_time",          false},
      {"expand",              false},  // must NOT expand: Fatrop relies on SX sparsity structure
      {"structure_detection", std::string("manual")},
      {"N",                   casadi_int(N)},
      {"nx",                  nx_v},
      {"nu",                  nu_v},
      {"ng",                  ng_v},
      {"fatrop",              fatrop_opts},
  };
  nlpSolver_ = casadi::nlpsol("wbc_rnea_mpc", "fatrop", nlp, opts);
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
  const int nq      = static_cast<int>(q.size());
  const int nv      = static_cast<int>(v.size());
  const int nj      = static_cast<int>(info_.actuatedDofNum);
  const int nc      = static_cast<int>(info_.numThreeDofContacts);
  const int nu_full = nv + 3 * nc + nj;
  const int nu_red  = nv + 3 * nc;
  const int nu      = nu_full;  // alias for parameter-vector sizing
  const int ndx     = 2 * nv;
  const int N       = numNodes_;

  // ── Task matrices (same for all nodes — refine per-node if MPC provides a trajectory)
  Task task = formulateWeightedTasks(stateDesired, inputDesired, period);
  matrix_t H_task = task.a_.transpose() * task.a_;
  vector_t g_task = -(task.a_.transpose() * task.b_);

  matrix_t J_nc   = matrix_t::Zero(3 * nc, nv);
  vector_t dJv_nc = vector_t::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    if (contactFlag[i]) {
      J_nc.block(3 * i, 0, 3, nv) = j_.block(3 * i, 0, 3, nv);
      dJv_nc.segment(3 * i, 3)    = dj_.block(3 * i, 0, 3, nv) * vMeasured_;
    }
  }

  // ── Pack parameter vector: [q0, v0, per-node x N]
  //    δt_k are baked into the NLP graph — not passed here
  const int p_node  = nu * nu + nu + 3 * nc * nv + 3 * nc;
  const int p_total = nq + nv + N * p_node;
  casadi::DM p_dm(p_total, 1);
  int off = 0;
  for (int i = 0; i < nq; i++)      p_dm(off++) = q(i);
  for (int i = 0; i < nv; i++)      p_dm(off++) = v(i);
  for (int k = 0; k < N; k++) {
    for (int i = 0; i < nu * nu; i++) p_dm(off++) = H_task.data()[i];
    for (int i = 0; i < nu; i++)      p_dm(off++) = g_task(i);
    for (int i = 0; i < 3*nc*nv; i++) p_dm(off++) = J_nc.data()[i];
    for (int i = 0; i < 3*nc; i++)    p_dm(off++) = dJv_nc(i);
  }

  // ── Constraint bounds
  // Per node (in g_list order): gap(ndx eq) + RNEA-base(6 eq) + RNEA-joints(nj eq)
  //                             + contact(3*nc eq) + friction(nc ineq)
  const int n_eq_per   = ndx + nv + 3 * nc;   // ndx + 6 + nj + 3*nc = 2nv+nv+3nc
  const int n_ineq_per = nc;
  const int n_per      = n_eq_per + n_ineq_per;
  casadi::DM lbg = casadi::DM::zeros(N * n_per, 1);
  casadi::DM ubg = casadi::DM::zeros(N * n_per, 1);
  for (int k = 0; k < N; k++) {
    const int friction_off = k * n_per + n_eq_per;
    for (int i = 0; i < n_ineq_per; i++) ubg(friction_off + i) = 1e20;
  }
  // For k >= tauNodes_: τ_j is not a decision variable, so relax the joint torque equality
  // to an actuator-limit inequality |tau_rnea[6:nv]| <= lim instead of forcing it to zero.
  // Offset within each node's constraint block: gap(ndx) + base(6) = ndx+6
  {
    const int lim_sz = static_cast<int>(torqueLimits_.size());
    for (int k = tauNodes_; k < N; k++) {
      const int joint_off = k * n_per + ndx + 6;
      for (int j = 0; j < nj; j++) {
        const double lim = torqueLimits_(j % lim_sz);
        lbg(joint_off + j) = -lim;
        ubg(joint_off + j) =  lim;
      }
    }
  }

  // ── Decision variable box constraints:
  //    z = [DX[0](ndx), U[0](nu_0), DX[1](ndx), U[1](nu_1), ..., U[N-1](nu_{N-1}), DX[N](ndx)]
  //    DX[0] is pinned to zero: it is the initial delta-state (always 0 at the current instant).
  casadi::DM lbz = -1e9 * casadi::DM::ones(nz_, 1);
  casadi::DM ubz =  1e9 * casadi::DM::ones(nz_, 1);
  for (int i = 0; i < ndx; i++) { lbz(i) = 0.0; ubz(i) = 0.0; }

  const int lim_size = static_cast<int>(torqueLimits_.size());
  int z_off = ndx;  // skip DX[0]
  for (int k = 0; k < N; k++) {
    const int nu_k = (k < tauNodes_) ? nu_full : nu_red;

    // Torque limits: only for tau nodes
    if (k < tauNodes_) {
      for (int j = 0; j < nj; j++) {
        const double lim = torqueLimits_(j % lim_size);
        lbz(z_off + nv + 3 * nc + j) = -lim;
        ubz(z_off + nv + 3 * nc + j) =  lim;
      }
    }
    // Force feasibility: contact Fc_z >= 0; swing Fc = 0
    for (int i = 0; i < nc; i++) {
      if (contactFlag[i]) {
        lbz(z_off + nv + 3 * i + 2) = 0.0;
      } else {
        lbz(z_off + nv + 3*i)   = 0.0; ubz(z_off + nv + 3*i)   = 0.0;
        lbz(z_off + nv + 3*i+1) = 0.0; ubz(z_off + nv + 3*i+1) = 0.0;
        lbz(z_off + nv + 3*i+2) = 0.0; ubz(z_off + nv + 3*i+2) = 0.0;
      }
    }
    z_off += nu_k;
    z_off += ndx;  // skip DX[k+1] (unbounded)
  }

  // ── Solve
  casadi::DMDict args = {{"lbg", lbg}, {"ubg", ubg}, {"lbx", lbz}, {"ubx", ubz}, {"p", p_dm}};
  if (hasWarmStart_) {
    args["x0"]     = prevX_;
    args["lam_g0"] = prevLamG_;
    args["lam_x0"] = prevLamX_;
  } else if (hasExternalGuess_) {
    args["x0"] = prevX_;
  } else {
    args["x0"] = casadi::DM::zeros(nz_, 1);
  }

  casadi::DMDict sol = nlpSolver_(args);

  // ── Diagnostics (throttled to 1 Hz)
  const auto& stats        = nlpSolver_.stats();
  const std::string status = static_cast<std::string>(stats.at("unified_return_status"));
  const int    iters       = static_cast<int>(stats.at("iter_count"));
  const double obj_val     = static_cast<double>(casadi::DM(sol.at("f")));

  casadi::DM g_val = sol.at("g");
  double g_inf = 0.0;
  for (int i = 0; i < g_val.size1(); i++)
    g_inf = std::max(g_inf, std::abs(static_cast<double>(g_val(i))));

  const bool solved_ok = (status == "SOLVER_RET_SUCCESS" || status == "SOLVER_RET_LIMITED");
  if (!solved_ok) {
    ROS_WARN_STREAM_THROTTLE(1.0,
        "[RneaWbc] Fatrop status: " << status
        << "  iters=" << iters << "  obj=" << obj_val << "  |g|_inf=" << g_inf);
  } else {
    ROS_INFO_STREAM_THROTTLE(1.0,
        "[RneaWbc] Fatrop status: " << status
        << "  iters=" << iters << "  obj=" << obj_val << "  |g|_inf=" << g_inf);
  }

  if (solved_ok) {
    prevX_            = sol.at("x");
    prevLamG_         = sol.at("lam_g");
    prevLamX_         = sol.at("lam_x");
    hasWarmStart_     = true;
    hasExternalGuess_ = false;
  } else {
    prevX_        = sol.at("x");
    hasWarmStart_ = false;
  }

  // Return U[0] — starts at offset ndx_ in the stage-wise layout [DX[0], U[0], ...]
  vector_t result(nu);
  for (int i = 0; i < nu; i++) result(i) = static_cast<double>(prevX_(ndx_ + i));

  // ── Diagnostic: contactFlag + τ_j per leg (throttled to 1 Hz)
  {
    std::string cf_str;
    for (int i = 0; i < nc; i++) cf_str += contactFlag[i] ? "1" : "0";
    const vector_t tau_j = result.tail(nj);
    ROS_DEBUG_STREAM_THROTTLE(1.0,
        "[RneaWbc] contactFlag=" << cf_str
        << "  tau_j=" << tau_j.transpose().format(
               Eigen::IOFormat(3, 0, " ", " ")));
  }

  return result;
}

void RneaWbc::setExternalInitialGuess(const vector_t& u0) {
  // U[0] starts at offset ndx_ in the stage-wise layout [DX[0], U[0], ...]
  prevX_ = casadi::DM::zeros(nz_, 1);
  const int nu = static_cast<int>(u0.size());
  for (int i = 0; i < nu; i++) prevX_(ndx_ + i) = u0(i);
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

  prefix = "rneaWbc.";
  if (verbose) {
    std::cerr << "\n #### RneaWbc planning:";
    std::cerr << "\n #### =============================================================================\n";
  }
  loadData::loadPtreeValue(pt, numNodes_, prefix + "numNodes", verbose);
  loadData::loadPtreeValue(pt, tauNodes_, prefix + "tauNodes", verbose);
  loadData::loadPtreeValue(pt, gamma_,    prefix + "gamma",    verbose);
  tauNodes_ = std::min(tauNodes_, numNodes_);  // guard: tauNodes ≤ numNodes

  // Determine base dt: rneaWbc.dt takes priority; falls back to sqp.dt.
  // totalHorizon = numNodes × dt; gamma controls adaptive spacing within it.
  double wbc_dt = 0.015;
  loadData::loadPtreeValue(pt, wbc_dt, "sqp.dt",         false);  // default from MPC
  loadData::loadPtreeValue(pt, wbc_dt, prefix + "dt",    verbose); // override if set
  totalHorizon_ = numNodes_ * wbc_dt;

  if (verbose) {
    std::cerr << " #### [RneaWbc] dt=" << wbc_dt
              << "  totalHorizon=" << totalHorizon_
              << "  gamma=" << gamma_ << "\n";
  }

  buildNlp();
}

}  // namespace legged
