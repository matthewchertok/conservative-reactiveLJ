// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "LiuOConnorTersoffForceComputeGPU.h"

#include "hoomd/Autotuner.h"

#include <stdexcept>
#include <string>

namespace hoomd
    {
namespace md
    {

LiuOConnorTersoffForceComputeGPU::LiuOConnorTersoffForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    const std::string& sticky_type,
    Scalar A1,
    Scalar A2,
    Scalar lambda1,
    Scalar lambda2,
    Scalar dimer_r,
    Scalar cutoff_thickness,
    Scalar r_cut,
    Scalar alpha,
    Scalar n,
    Scalar gamma)
    : LiuOConnorTersoffForceCompute(sysdef,
                                    nlist,
                                    sticky_type,
                                    A1,
                                    A2,
                                    lambda1,
                                    lambda2,
                                    dimer_r,
                                    cutoff_thickness,
                                    r_cut,
                                    alpha,
                                    n,
                                    gamma)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a LiuOConnorTersoffForceComputeGPU with no GPU in the execution "
               "configuration"
            << std::endl;
        throw std::runtime_error("Error initializing LiuOConnorTersoffForceComputeGPU");
        }

    m_tuner_coord.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                          AutotunerBase::getTppListPow2(m_exec_conf)},
                                         m_exec_conf,
                                         "liu_o_connor_tersoff_coord"));
    m_tuner_lambda.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                           AutotunerBase::getTppListPow2(m_exec_conf)},
                                          m_exec_conf,
                                          "liu_o_connor_tersoff_lambda"));
    m_tuner_force.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                          AutotunerBase::getTppListPow2(m_exec_conf)},
                                         m_exec_conf,
                                         "liu_o_connor_tersoff_force"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_coord, m_tuner_lambda, m_tuner_force});

#ifdef ENABLE_MPI
    m_tuner_coord->setSync(bool(m_pdata->getDomainDecomposition()));
    m_tuner_lambda->setSync(bool(m_pdata->getDomainDecomposition()));
    m_tuner_force->setSync(bool(m_pdata->getDomainDecomposition()));
#endif
    }

void LiuOConnorTersoffForceComputeGPU::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);
    ensureScratchSize();

    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error()
            << "LiuOConnorTersoffForceComputeGPU cannot handle a half neighborlist"
            << std::endl;
        throw std::runtime_error("Error computing forces in LiuOConnorTersoffForceComputeGPU");
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
    const bool compute_virial = m_pdata->getFlags()[pdata_flag::pressure_tensor];

    m_force.zeroFill();
    m_virial.zeroFill();
    m_coordination.zeroFill();
    m_lambda.zeroFill();

    kernel::liu_o_connor_tersoff_params params;
    params.A1 = m_A1;
    params.A2 = m_A2;
    params.lambda1 = m_lambda1;
    params.lambda2 = m_lambda2;
    params.dimer_r = m_dimer_r;
    params.cutoff_thickness = m_cutoff_thickness;
    params.r_cut = m_r_cut;
    params.r_cut_sq = m_r_cut_sq;
    params.r_inner = m_r_inner;
    params.inv_cutoff_thickness = m_inv_cutoff_thickness;
    params.cutoff_alpha = m_cutoff_alpha;
    params.n = m_n;
    params.gamma = m_gamma;
    params.gamma_n = m_gamma_n;
    params.b_exponent = m_b_exponent;
    params.sticky_typeid = m_sticky_typeid;

    m_exec_conf->setDevice();

    m_tuner_coord->begin();
    const auto coord_params = m_tuner_coord->getParam();
    hipError_t err = kernel::gpu_liu_o_connor_tersoff_compute_coordination(
        N,
        d_pos.data,
        d_n_neigh.data,
        d_nlist.data,
        d_head_list.data,
        d_coord.data,
        box,
        params,
        coord_params[0],
        coord_params[1]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("LiuOConnorTersoffForceComputeGPU coordination "
                                             "kernel launch failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_coord->end();

    m_tuner_lambda->begin();
    const auto lambda_params = m_tuner_lambda->getParam();
    err = kernel::gpu_liu_o_connor_tersoff_compute_lambda(N,
                                                          d_pos.data,
                                                          d_n_neigh.data,
                                                          d_nlist.data,
                                                          d_head_list.data,
                                                          d_coord.data,
                                                          d_lambda.data,
                                                          box,
                                                          params,
                                                          lambda_params[0],
                                                          lambda_params[1]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("LiuOConnorTersoffForceComputeGPU lambda kernel "
                                             "launch failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_lambda->end();

    m_tuner_force->begin();
    const auto force_params = m_tuner_force->getParam();
    err = kernel::gpu_liu_o_connor_tersoff_compute_forces(N,
                                                          d_pos.data,
                                                          d_n_neigh.data,
                                                          d_nlist.data,
                                                          d_head_list.data,
                                                          d_coord.data,
                                                          d_lambda.data,
                                                          d_force.data,
                                                          d_virial.data,
                                                          virial_pitch,
                                                          box,
                                                          params,
                                                          compute_virial,
                                                          force_params[0],
                                                          force_params[1]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("LiuOConnorTersoffForceComputeGPU force kernel "
                                             "launch failed: ")
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
void export_LiuOConnorTersoffForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<LiuOConnorTersoffForceComputeGPU,
                     LiuOConnorTersoffForceCompute,
                     std::shared_ptr<LiuOConnorTersoffForceComputeGPU>>(
        m,
        "LiuOConnorTersoffForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            const std::string&,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>(),
             pybind11::arg("sysdef"),
             pybind11::arg("nlist"),
             pybind11::arg("sticky_type"),
             pybind11::arg("A1"),
             pybind11::arg("A2"),
             pybind11::arg("lambda1"),
             pybind11::arg("lambda2"),
             pybind11::arg("dimer_r"),
             pybind11::arg("cutoff_thickness"),
             pybind11::arg("r_cut"),
             pybind11::arg("alpha"),
             pybind11::arg("n"),
             pybind11::arg("gamma"));
    }

    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
