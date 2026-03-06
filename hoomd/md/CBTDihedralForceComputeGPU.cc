// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CBTDihedralForceComputeGPU.cc
    \brief Defines CBTDihedralForceComputeGPU.
*/

#include "CBTDihedralForceComputeGPU.h"

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute dihedral forces on.
 */
CBTDihedralForceComputeGPU::CBTDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : CBTDihedralForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a CBTDihedralForceComputeGPU with no GPU in execution configuration"
            << endl;
        throw std::runtime_error("Error initializing CBTDihedralForceComputeGPU");
        }

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "cbt_dihedral"));
    m_autotuners.push_back(m_tuner);
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces.

    \param timestep Current time step of the simulation.
*/
void CBTDihedralForceComputeGPU::computeForces(uint64_t timestep)
    {
    (void)timestep;

    ArrayHandle<DihedralData::members_t> d_gpu_dihedral_list(m_dihedral_data->getGPUTable(),
                                                             access_location::device,
                                                             access_mode::read);
    ArrayHandle<unsigned int> d_n_dihedrals(m_dihedral_data->getNGroupsArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_dihedrals_ABCD(m_dihedral_data->getGPUPosTable(),
                                               access_location::device,
                                               access_mode::read);

    // the dihedral table is up to date: call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<dihedral_cbt_params> d_params(m_params, access_location::device, access_mode::read);

    m_tuner->begin();
    kernel::gpu_compute_cbt_dihedral_forces(d_force.data,
                                            d_virial.data,
                                            m_virial.getPitch(),
                                            m_pdata->getN(),
                                            d_pos.data,
                                            box,
                                            d_gpu_dihedral_list.data,
                                            d_dihedrals_ABCD.data,
                                            m_dihedral_data->getGPUTableIndexer().getW(),
                                            d_n_dihedrals.data,
                                            d_params.data,
                                            m_dihedral_data->getNTypes(),
                                            m_tuner->getParam()[0],
                                            m_exec_conf->dev_prop.warpSize);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_CBTDihedralForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<CBTDihedralForceComputeGPU,
                     CBTDihedralForceCompute,
                     std::shared_ptr<CBTDihedralForceComputeGPU>>(m, "CBTDihedralForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
