// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "ReactiveLJForceComputeGPU.h"

#include "hoomd/Autotuner.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace hoomd
    {
namespace md
    {

ReactiveLJForceComputeGPU::ReactiveLJForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<NeighborList> nlist,
                                                     const std::string& reactive_type,
                                                     Scalar sigma,
                                                     Scalar epsilon,
                                                     Scalar r_cut,
                                                     Scalar weakening_inner,
                                                     Scalar weakening_outer,
                                                     Scalar weakening_exponent,
                                                     bool smooth_elbow,
                                                     Scalar smooth_kappa,
                                                     Scalar smooth_beta)
    : ReactiveLJForceCompute(sysdef,
                             nlist,
                             reactive_type,
                             sigma,
                             epsilon,
                             r_cut,
                             weakening_inner,
                             weakening_outer,
                             weakening_exponent,
                             smooth_elbow,
                             smooth_kappa,
                             smooth_beta)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a ReactiveLJForceComputeGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing ReactiveLJForceComputeGPU");
        }

    m_tuner_coord.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                          AutotunerBase::getTppListPow2(m_exec_conf)},
                                         m_exec_conf,
                                         "reactivelj_coord"));
    m_tuner_lambda.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                           AutotunerBase::getTppListPow2(m_exec_conf)},
                                          m_exec_conf,
                                          "reactivelj_lambda"));
    m_tuner_force.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                          AutotunerBase::getTppListPow2(m_exec_conf)},
                                         m_exec_conf,
                                         "reactivelj_force"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_coord, m_tuner_lambda, m_tuner_force});

#ifdef ENABLE_MPI
    m_tuner_coord->setSync(bool(m_pdata->getDomainDecomposition()));
    m_tuner_lambda->setSync(bool(m_pdata->getDomainDecomposition()));
    m_tuner_force->setSync(bool(m_pdata->getDomainDecomposition()));
#endif
    }

void ReactiveLJForceComputeGPU::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);
    ensureScratchSize();

    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error()
            << "ReactiveLJForceComputeGPU cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in ReactiveLJForceComputeGPU");
        }

    ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                              access_location::device,
                              access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_coord(m_coordination, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_lambda(m_lambda, access_location::device, access_mode::readwrite);

    const unsigned int N = m_pdata->getN();
    const size_t virial_pitch = m_virial.getPitch();
    const BoxDim box = m_pdata->getBox();

    PDataFlags flags = m_pdata->getFlags();
    const bool compute_virial = flags[pdata_flag::pressure_tensor];

    m_force.zeroFill();
    m_virial.zeroFill();
    m_coordination.zeroFill();
    m_lambda.zeroFill();

    kernel::reactive_lj_params params;
    params.sigma = m_sigma;
    params.epsilon = m_epsilon;
    params.r_cut = m_r_cut;
    params.r_cut_sq = m_r_cut_sq;
    params.neighbor_rcut_sq = m_neighbor_rcut_sq;
    params.energy_shift = m_energy_shift;
    params.weakening_inner = m_weakening_inner;
    params.weakening_outer = m_weakening_outer;
    params.weakening_exponent = m_weakening_exponent;
    params.weakening_exponent_int = m_weakening_exponent_int;
    params.inverse_span = m_inverse_span;
    params.pi_inverse_span = m_pi_inverse_span;
    params.r_elbow = m_r_elbow;
    params.smooth_elbow = m_smooth_elbow;
    params.smooth_kappa = m_smooth_kappa;
    params.smooth_beta = m_smooth_beta;
    params.smooth_beta_is_one = m_smooth_beta_is_one ? 1u : 0u;
    params.smooth_kappa_sigma = m_smooth_kappa_sigma;
    params.smooth_delta_tol = m_smooth_delta_tol;
    params.smooth_r_min = m_smooth_r_min;
    params.lj_energy12 = m_lj_energy12;
    params.lj_energy6 = m_lj_energy6;
    params.lj_force12 = m_lj_force12;
    params.lj_force6 = m_lj_force6;
    params.lj_force_deriv12 = m_lj_force_deriv12;
    params.lj_force_deriv6 = m_lj_force_deriv6;

    m_exec_conf->setDevice();

    // Pass 1: compute per-particle coordination numbers.
    m_tuner_coord->begin();
    const auto coord_params = m_tuner_coord->getParam();
    hipError_t err = kernel::gpu_reactivelj_compute_coordination(N,
                                                                 d_pos.data,
                                                                 d_n_neigh.data,
                                                                 d_nlist.data,
                                                                 d_head_list.data,
                                                                 d_coord.data,
                                                                 m_reactive_typeid,
                                                                 box,
                                                                 params,
                                                                 coord_params[0],
                                                                 coord_params[1]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("ReactiveLJForceComputeGPU coordination kernel "
                                             "launch failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_coord->end();

    // Pass 2: accumulate lambda on each endpoint (pair-once, CPU-equivalent semantics).
    m_tuner_lambda->begin();
    const auto lambda_params = m_tuner_lambda->getParam();
    err = kernel::gpu_reactivelj_compute_lambda(N,
                                                d_pos.data,
                                                d_n_neigh.data,
                                                d_nlist.data,
                                                d_head_list.data,
                                                d_coord.data,
                                                d_lambda.data,
                                                m_reactive_typeid,
                                                box,
                                                params,
                                                lambda_params[0],
                                                lambda_params[1]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("ReactiveLJForceComputeGPU lambda kernel launch "
                                             "failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_lambda->end();

    // Pass 3: evaluate forces and optionally the virial tensor.
    m_tuner_force->begin();
    const auto force_params = m_tuner_force->getParam();
    err = kernel::gpu_reactivelj_compute_forces(N,
                                                d_pos.data,
                                                d_n_neigh.data,
                                                d_nlist.data,
                                                d_head_list.data,
                                                d_coord.data,
                                                d_lambda.data,
                                                d_force.data,
                                                d_virial.data,
                                                virial_pitch,
                                                m_reactive_typeid,
                                                box,
                                                params,
                                                compute_virial,
                                                force_params[0],
                                                force_params[1]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("ReactiveLJForceComputeGPU force kernel launch "
                                             "failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_force->end();
    }

namespace detail
    {
void export_ReactiveLJForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<ReactiveLJForceComputeGPU,
                     ReactiveLJForceCompute,
                     std::shared_ptr<ReactiveLJForceComputeGPU>>(m,
                                                                 "ReactiveLJForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            const std::string&,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            bool,
                            Scalar,
                            Scalar>(),
             pybind11::arg("sysdef"),
             pybind11::arg("nlist"),
             pybind11::arg("reactive_type"),
             pybind11::arg("sigma"),
             pybind11::arg("epsilon"),
             pybind11::arg("r_cut"),
             pybind11::arg("weakening_inner"),
             pybind11::arg("weakening_outer"),
             pybind11::arg("weakening_exponent"),
             pybind11::arg("smooth_elbow") = true,
             pybind11::arg("smooth_kappa") = Scalar(0.05),
             pybind11::arg("smooth_beta") = Scalar(1.0));
    }

    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
