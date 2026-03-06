// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CBTDihedralForceTypes.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file CBTDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating CBT dihedral forces.
*/

#ifndef __CBTDIHEDRALFORCEGPU_CUH__
#define __CBTDIHEDRALFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes CBT dihedral forces for CBTDihedralForceComputeGPU
hipError_t gpu_compute_cbt_dihedral_forces(Scalar4* d_force,
                                           Scalar* d_virial,
                                           const size_t virial_pitch,
                                           const unsigned int N,
                                           const Scalar4* d_pos,
                                           const BoxDim& box,
                                           const group_storage<4>* tlist,
                                           const unsigned int* dihedral_ABCD,
                                           const unsigned int pitch,
                                           const unsigned int* n_dihedrals_list,
                                           const dihedral_cbt_params* d_params,
                                           const unsigned int n_dihedral_types,
                                           const int block_size,
                                           const int warp_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
