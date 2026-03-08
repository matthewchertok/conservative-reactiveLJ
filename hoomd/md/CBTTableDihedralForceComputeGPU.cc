// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CBTTableDihedralForceComputeGPU.cc
    \brief Defines CBTTableDihedralForceComputeGPU.
*/

#include "CBTTableDihedralForceComputeGPU.h"

#include <algorithm>

using namespace std;

namespace hoomd
    {
namespace md
    {
CBTTableDihedralForceComputeGPU::CBTTableDihedralForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    unsigned int table_width)
    : CBTTableDihedralForceCompute(sysdef, table_width)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a CBTTableDihedralForceComputeGPU with no GPU in execution configuration"
            << endl;
        throw std::runtime_error("Error initializing CBTTableDihedralForceComputeGPU");
        }

    auto block_size_range = AutotunerBase::makeBlockSizeRange(m_exec_conf);
    unsigned int kernel_max_block_size = 0;
    const hipError_t err = kernel::gpu_cbt_table_dihedral_get_max_block_size(kernel_max_block_size);
    if (err != hipSuccess || kernel_max_block_size == 0)
        {
        m_exec_conf->msg->warning()
            << "CBTTableDihedralForceComputeGPU: failed to query max block size for CBT table "
               "dihedral kernel; using default autotuner range. Error: "
            << hipGetErrorString(err) << endl;
        }
    else
        {
        block_size_range.erase(remove_if(block_size_range.begin(),
                                         block_size_range.end(),
                                         [kernel_max_block_size](unsigned int block_size)
                                         { return block_size > kernel_max_block_size; }),
                               block_size_range.end());

        if (block_size_range.empty())
            {
            block_size_range.push_back(kernel_max_block_size);
            }
        }

    m_tuner.reset(new Autotuner<1>({block_size_range},
                                   m_exec_conf,
                                   "cbt_table_dihedral"));
    m_autotuners.push_back(m_tuner);
    }

void CBTTableDihedralForceComputeGPU::computeForces(uint64_t timestep)
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

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_tables(m_tables, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_angle_exponents(m_angle_exponents,
                                          access_location::device,
                                          access_mode::read);

    m_tuner->begin();
    const hipError_t err = kernel::gpu_compute_cbt_table_dihedral_forces(d_force.data,
                                                                          d_virial.data,
                                                                          m_virial.getPitch(),
                                                                          m_pdata->getN(),
                                                                          d_pos.data,
                                                                          box,
                                                                          d_gpu_dihedral_list.data,
                                                                          d_dihedrals_ABCD.data,
                                                                          m_dihedral_data
                                                                              ->getGPUTableIndexer()
                                                                              .getW(),
                                                                          d_n_dihedrals.data,
                                                                          d_tables.data,
                                                                          m_table_width,
                                                                          m_table_value,
                                                                          d_angle_exponents.data,
                                                                          m_tuner->getParam()[0]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("CBTTableDihedralForceComputeGPU kernel launch "
                                             "failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_CBTTableDihedralForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<CBTTableDihedralForceComputeGPU,
                     CBTTableDihedralForceCompute,
                     std::shared_ptr<CBTTableDihedralForceComputeGPU>>(m,
                                                                       "CBTTableDihedralForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
