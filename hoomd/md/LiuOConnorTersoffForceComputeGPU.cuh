// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_GPU_CUH__
#define __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_GPU_CUH__

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
#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_liu_o_connor_tersoff_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
const int gpu_liu_o_connor_tersoff_max_tpp = 64;
#else
const int gpu_liu_o_connor_tersoff_max_tpp = 32;
#endif

struct liu_o_connor_tersoff_params
    {
    Scalar A1;
    Scalar A2;
    Scalar lambda1;
    Scalar lambda2;
    Scalar dimer_r;
    Scalar cutoff_thickness;
    Scalar r_cut;
    Scalar r_cut_sq;
    Scalar r_inner;
    Scalar inv_cutoff_thickness;
    Scalar cutoff_alpha;
    Scalar n;
    Scalar gamma;
    Scalar gamma_n;
    Scalar b_exponent;
    unsigned int sticky_typeid;
    };

hipError_t gpu_liu_o_connor_tersoff_compute_coordination(
    const unsigned int N,
    const Scalar4* d_pos,
    const unsigned int* d_n_neigh,
    const unsigned int* d_nlist,
    const size_t* d_head_list,
    Scalar* d_coord,
    BoxDim box,
    liu_o_connor_tersoff_params params,
    unsigned int block_size,
    unsigned int threads_per_particle);

hipError_t gpu_liu_o_connor_tersoff_compute_lambda(
    const unsigned int N,
    const Scalar4* d_pos,
    const unsigned int* d_n_neigh,
    const unsigned int* d_nlist,
    const size_t* d_head_list,
    const Scalar* d_coord,
    Scalar* d_lambda,
    BoxDim box,
    liu_o_connor_tersoff_params params,
    unsigned int block_size,
    unsigned int threads_per_particle);

hipError_t gpu_liu_o_connor_tersoff_compute_forces(
    const unsigned int N,
    const Scalar4* d_pos,
    const unsigned int* d_n_neigh,
    const unsigned int* d_nlist,
    const size_t* d_head_list,
    const Scalar* d_coord,
    const Scalar* d_lambda,
    Scalar4* d_force,
    Scalar* d_virial,
    const size_t virial_pitch,
    BoxDim box,
    liu_o_connor_tersoff_params params,
    unsigned int compute_virial,
    unsigned int block_size,
    unsigned int threads_per_particle);

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP

#endif // __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_GPU_CUH__
