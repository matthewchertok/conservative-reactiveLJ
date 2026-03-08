// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hip/hip_runtime.h"

/*! \file CBTTableDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating tabulated CBT dihedral forces.
*/

#ifndef __CBTTABLEDIHEDRALFORCEGPU_CUH__
#define __CBTTABLEDIHEDRALFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_cbt_table_dihedral_get_max_block_size(unsigned int& max_block_size);

hipError_t gpu_compute_cbt_table_dihedral_forces(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const size_t virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4* d_pos,
                                                 const BoxDim& box,
                                                 const group_storage<4>* tlist,
                                                 const unsigned int* dihedral_ABCD,
                                                 const unsigned int pitch,
                                                 const unsigned int* n_dihedrals_list,
                                                 const Scalar2* d_tables,
                                                 const unsigned int table_width,
                                                 const Index2D& table_value,
                                                 const Scalar* d_angle_exponents,
                                                 const unsigned int block_size);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
