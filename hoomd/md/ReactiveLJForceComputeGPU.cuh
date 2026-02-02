// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __REACTIVE_LJ_FORCE_COMPUTE_GPU_CUH__
#define __REACTIVE_LJ_FORCE_COMPUTE_GPU_CUH__

#ifdef ENABLE_HIP

#include "hip/hip_runtime.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

struct reactive_lj_params
    {
    Scalar sigma;
    Scalar epsilon;
    Scalar r_cut;
    Scalar r_cut_sq;
    Scalar neighbor_rcut_sq;
    Scalar energy_shift;
    Scalar weakening_inner;
    Scalar weakening_outer;
    Scalar weakening_exponent;
    Scalar inverse_span;
    Scalar r_elbow;
    bool smooth_elbow;
    Scalar smooth_kappa;
    Scalar smooth_beta;
    Scalar smooth_delta_tol;
    Scalar smooth_r_min;
    };

hipError_t gpu_reactivelj_compute_coordination(const unsigned int N,
                                               const Scalar4* d_pos,
                                               const unsigned int* d_n_neigh,
                                               const unsigned int* d_nlist,
                                               const size_t* d_head_list,
                                               Scalar* d_coord,
                                               unsigned int reactive_typeid,
                                               BoxDim box,
                                               reactive_lj_params params,
                                               unsigned int block_size);

hipError_t gpu_reactivelj_compute_lambda(const unsigned int N,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_n_neigh,
                                         const unsigned int* d_nlist,
                                         const size_t* d_head_list,
                                         const Scalar* d_coord,
                                         Scalar* d_lambda,
                                         unsigned int* d_error_flag,
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_lj_params params,
                                         unsigned int block_size);

hipError_t gpu_reactivelj_compute_forces(const unsigned int N,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_n_neigh,
                                         const unsigned int* d_nlist,
                                         const size_t* d_head_list,
                                         const Scalar* d_coord,
                                         const Scalar* d_lambda,
                                         Scalar4* d_force,
                                         Scalar* d_virial,
                                         const size_t virial_pitch,
                                         unsigned int* d_error_flag,
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_lj_params params,
                                         unsigned int compute_virial,
                                         unsigned int block_size);

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP

#endif // __REACTIVE_LJ_FORCE_COMPUTE_GPU_CUH__
