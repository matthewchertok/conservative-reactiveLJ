// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "ReactiveLJForceComputeGPU.cuh"

#include "hoomd/HOOMDMath.h"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"
#include "hoomd/WarpTools.cuh"

#include <algorithm>
#include <mutex>
#include <unordered_map>

#ifdef DEVICE
#undef DEVICE
#endif
#define DEVICE __device__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
namespace
    {
hipError_t getCachedMaxThreadsPerBlock(const void* kernel_ptr,
                                       std::unordered_map<int, unsigned int>& cache,
                                       std::mutex& cache_mutex,
                                       unsigned int& max_threads_per_block)
    {
    int device = -1;
    hipError_t err = hipGetDevice(&device);
    if (err != hipSuccess)
        {
        return err;
        }

    {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(device);
    if (it != cache.end())
        {
        max_threads_per_block = it->second;
        return hipSuccess;
        }
    }

    hipFuncAttributes attr;
    err = hipFuncGetAttributes(&attr, kernel_ptr);
    if (err != hipSuccess)
        {
        return err;
        }

    max_threads_per_block = static_cast<unsigned int>(attr.maxThreadsPerBlock);

    {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.emplace(device, max_threads_per_block);
    }

    return hipSuccess;
    }
    } // end namespace

#if HOOMD_LONGREAL_SIZE == 64
#if (__CUDA_ARCH__ < 600)
DEVICE inline double atomicAddScalar(double* address, double val)
    {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
        {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
    }
#else
DEVICE inline double atomicAddScalar(double* address, double val)
    {
    return atomicAdd(address, val);
    }
#endif
#else
#ifdef __HIP_PLATFORM_HCC__
DEVICE inline float atomicAddScalar(float* address, float val)
    {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do
        {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val + __uint_as_float(assumed)));
        } while (assumed != old);
    return __uint_as_float(old);
    }
#else
DEVICE inline float atomicAddScalar(float* address, float val)
    {
    return atomicAdd(address, val);
    }
#endif
#endif

// Smoothly project raw excess coordination onto max(raw, 0). The gate term is
// dC_exc/draw and is reused in chain-rule terms later on.
DEVICE inline void evaluateCexcSmooth(Scalar raw, Scalar& C_exc, Scalar& gate)
    {
    const Scalar eps = Scalar(1e-6);
    const Scalar s = slow::sqrt(raw * raw + eps * eps);
    C_exc = Scalar(0.5) * (raw + s);
    gate = Scalar(0.5) * (Scalar(1.0) + raw / s);
    }

DEVICE inline Scalar powIntegerOneToNine(Scalar base, unsigned int exponent)
    {
    switch (exponent)
        {
    case 1:
        return base;
    case 2:
        return base * base;
    case 3:
        return base * base * base;
    case 4:
        {
        const Scalar base2 = base * base;
        return base2 * base2;
        }
    case 5:
        {
        const Scalar base2 = base * base;
        return base2 * base2 * base;
        }
    case 6:
        {
        const Scalar base2 = base * base;
        return base2 * base2 * base2;
        }
    case 7:
        {
        const Scalar base2 = base * base;
        return base2 * base2 * base2 * base;
        }
    case 8:
        {
        const Scalar base2 = base * base;
        const Scalar base4 = base2 * base2;
        return base4 * base4;
        }
    case 9:
        {
        const Scalar base2 = base * base;
        const Scalar base4 = base2 * base2;
        return base4 * base4 * base;
        }
    default:
        return slow::pow(base, Scalar(exponent));
        }
    }

DEVICE inline Scalar evaluateWeakening(Scalar C_exc, const reactive_lj_params& params)
    {
    if (params.weakening_exponent == Scalar(0.0))
        {
        return Scalar(1.0);
        }

    const Scalar inv_coord = Scalar(1.0) / (Scalar(1.0) + C_exc);
    if (params.weakening_exponent_int != 0)
        {
        return powIntegerOneToNine(inv_coord, params.weakening_exponent_int);
        }
    return slow::pow(Scalar(1.0) + C_exc, -params.weakening_exponent);
    }

DEVICE inline Scalar evaluateSmoothDelta(Scalar weakening, const reactive_lj_params& params)
    {
    const Scalar one_minus_w
        = (Scalar(1.0) > weakening) ? (Scalar(1.0) - weakening) : Scalar(0.0);
    if (params.smooth_beta_is_one)
        {
        return params.smooth_kappa_sigma * one_minus_w;
        }
    return params.smooth_kappa_sigma * slow::pow(one_minus_w, params.smooth_beta);
    }

DEVICE inline Scalar evaluateSmoothDeltaDerivative(Scalar weakening,
                                                   const reactive_lj_params& params)
    {
    const Scalar one_minus_w
        = (Scalar(1.0) > weakening) ? (Scalar(1.0) - weakening) : Scalar(0.0);
    if (params.smooth_beta_is_one)
        {
        return -params.smooth_kappa_sigma;
        }
    if (one_minus_w > Scalar(0.0))
        {
        return -params.smooth_kappa_sigma * params.smooth_beta
               * slow::pow(one_minus_w, params.smooth_beta - Scalar(1.0));
        }
    return Scalar(0.0);
    }

// Cosine window used for coordination weights between inner and outer radii.
DEVICE inline Scalar computeWeight(Scalar distance, const reactive_lj_params& params)
    {
    if (distance >= params.weakening_outer)
        {
        return Scalar(0.0);
        }
    if (distance <= params.weakening_inner)
        {
        return Scalar(1.0);
        }

    const Scalar fraction = (distance - params.weakening_inner) * params.inverse_span;
    Scalar sin_fraction = Scalar(0.0);
    Scalar cos_fraction = Scalar(0.0);
    fast::sincospi(fraction, sin_fraction, cos_fraction);
    (void)sin_fraction;
    return Scalar(0.5) * (Scalar(1.0) + cos_fraction);
    }

// Derivative of the cosine coordination window with respect to distance.
DEVICE inline Scalar evaluateWeightDerivative(Scalar distance, const reactive_lj_params& params)
    {
    if (distance >= params.weakening_outer || distance <= params.weakening_inner)
        {
        return Scalar(0.0);
        }

    const Scalar fraction = (distance - params.weakening_inner) * params.inverse_span;
    Scalar sin_fraction = Scalar(0.0);
    Scalar cos_fraction = Scalar(0.0);
    fast::sincospi(fraction, sin_fraction, cos_fraction);
    (void)cos_fraction;
    return -Scalar(0.5) * sin_fraction * params.pi_inverse_span;
    }

DEVICE inline void computeWeightAndDerivative(Scalar distance,
                                              const reactive_lj_params& params,
                                              Scalar& weight,
                                              Scalar& dw_dr)
    {
    if (distance >= params.weakening_outer)
        {
        weight = Scalar(0.0);
        dw_dr = Scalar(0.0);
        return;
        }
    if (distance <= params.weakening_inner)
        {
        weight = Scalar(1.0);
        dw_dr = Scalar(0.0);
        return;
        }

    const Scalar fraction = (distance - params.weakening_inner) * params.inverse_span;
    Scalar sin_fraction = Scalar(0.0);
    Scalar cos_fraction = Scalar(0.0);
    fast::sincospi(fraction, sin_fraction, cos_fraction);
    weight = Scalar(0.5) * (Scalar(1.0) + cos_fraction);
    dw_dr = -Scalar(0.5) * sin_fraction * params.pi_inverse_span;
    }

// Shifted Lennard-Jones energy used by the ReactiveLJ pair branch.
DEVICE inline Scalar evaluateEnergy(Scalar distance, const reactive_lj_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar inv_r2 = inv_r * inv_r;
    const Scalar inv_r6 = inv_r2 * inv_r2 * inv_r2;
    const Scalar inv_r12 = inv_r6 * inv_r6;
    return params.lj_energy12 * inv_r12 - params.lj_energy6 * inv_r6 - params.energy_shift;
    }

// Positive magnitude of the pair force, F(r), before multiplying by direction.
DEVICE inline Scalar evaluateForceMagnitude(Scalar distance, const reactive_lj_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar inv_r2 = inv_r * inv_r;
    const Scalar inv_r6 = inv_r2 * inv_r2 * inv_r2;
    const Scalar inv_r12 = inv_r6 * inv_r6;
    return params.lj_force12 * inv_r12 * inv_r - params.lj_force6 * inv_r6 * inv_r;
    }

// Radial derivative dF/dr used in the smoothing-region chain rule.
DEVICE inline Scalar evaluateForceDerivative(Scalar distance, const reactive_lj_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar inv_r2 = inv_r * inv_r;
    const Scalar inv_r6 = inv_r2 * inv_r2 * inv_r2;
    const Scalar inv_r12 = inv_r6 * inv_r6;

    return params.lj_force_deriv12 * inv_r12 * inv_r2
           + params.lj_force_deriv6 * inv_r6 * inv_r2;
    }

// Compute dU/dC_exc for one directed pair. This drives the many-body lambda
// accumulation and, through lambda, the coordination force in pass 3.
DEVICE inline Scalar evaluate_dU_dCexc(Scalar r_in,
                                      Scalar C_exc,
                                      const reactive_lj_params& params)
    {
    const Scalar r = (r_in > params.smooth_r_min) ? r_in : params.smooth_r_min;

    const Scalar W = evaluateWeakening(C_exc, params);

    Scalar dW_dC = Scalar(0.0);
    if (params.weakening_exponent > Scalar(0.0))
        {
        const Scalar denom = Scalar(1.0) + C_exc;
        dW_dC = -params.weakening_exponent * W / denom;
        }

    bool smoothing_active = false;
    Scalar delta = Scalar(0.0);
    Scalar r1 = Scalar(0.0);
    Scalar r2 = Scalar(0.0);
    Scalar L = Scalar(0.0);

    if (params.smooth_elbow)
        {
        // As weakening grows, the Hermite blending window opens around r_elbow.
        delta = evaluateSmoothDelta(W, params);

        r1 = params.r_elbow - delta;
        r2 = params.r_elbow + delta;
        L = r2 - r1;
        // Outside this narrow region we fall back to the simpler branch.
        if (delta > params.smooth_delta_tol && L > params.smooth_delta_tol && r > r1 && r < r2)
            {
            smoothing_active = true;
            }
        }

    if (!smoothing_active)
        {
        // Unsmoothed branch: only the attractive part is weakened by W(C_exc).
        const Scalar U_LJ = evaluateEnergy(r, params);
        const Scalar U_att = (U_LJ < Scalar(0.0)) ? U_LJ : Scalar(0.0);
        return U_att * dW_dC;
        }

    const Scalar ddelta_dW = evaluateSmoothDeltaDerivative(W, params);

    const Scalar dr1_dW = -ddelta_dW;
    const Scalar dr2_dW = ddelta_dW;
    const Scalar dL_dW = Scalar(2.0) * ddelta_dW;

    const Scalar t = (r - r1) / L;
    const Scalar L2 = L * L;
    const Scalar dt_dW = (-(dr1_dW) * L - (r - r1) * dL_dW) / L2;

    const Scalar t2 = t * t;
    const Scalar t3 = t2 * t;

    const Scalar h00 = Scalar(2.0) * t3 - Scalar(3.0) * t2 + Scalar(1.0);
    const Scalar h10 = t3 - Scalar(2.0) * t2 + t;
    const Scalar h01 = -Scalar(2.0) * t3 + Scalar(3.0) * t2;
    const Scalar h11 = t3 - t2;

    const Scalar dh00_dt = Scalar(6.0) * t2 - Scalar(6.0) * t;
    const Scalar dh10_dt = Scalar(3.0) * t2 - Scalar(4.0) * t + Scalar(1.0);
    const Scalar dh01_dt = -Scalar(6.0) * t2 + Scalar(6.0) * t;
    const Scalar dh11_dt = Scalar(3.0) * t2 - Scalar(2.0) * t;

    const Scalar dh00_dW = dh00_dt * dt_dW;
    const Scalar dh10_dW = dh10_dt * dt_dW;
    const Scalar dh01_dW = dh01_dt * dt_dW;
    const Scalar dh11_dW = dh11_dt * dt_dW;

    const Scalar U1 = evaluateEnergy(r1, params);
    const Scalar F1 = evaluateForceMagnitude(r1, params);
    const Scalar dU1 = -F1;

    const Scalar U_LJ2 = evaluateEnergy(r2, params);
    const Scalar F2 = evaluateForceMagnitude(r2, params);
    const Scalar U2 = W * U_LJ2;
    const Scalar dU2 = -W * F2;

    const Scalar dF1_dr = evaluateForceDerivative(r1, params);
    const Scalar d2U1_dr2 = -dF1_dr;

    const Scalar dF2_dr = evaluateForceDerivative(r2, params);

    const Scalar dU1_dW = (-F1) * dr1_dW;
    const Scalar ddU1_dW = d2U1_dr2 * dr1_dW;

    const Scalar dU2_dW = U_LJ2 + W * (-F2) * dr2_dW;
    const Scalar ddU2_dW = (-F2) - W * dF2_dr * dr2_dW;

    const Scalar dH_dW
        = dh00_dW * U1 + h00 * dU1_dW + dh10_dW * L * dU1 + h10 * dL_dW * dU1
          + h10 * L * ddU1_dW + dh01_dW * U2 + h01 * dU2_dW + dh11_dW * L * dU2
          + h11 * dL_dW * dU2 + h11 * L * ddU2_dW;

    return dH_dW * dW_dC;
    }

template<int tpp>
__global__ void gpu_reactivelj_compute_coordination_kernel(const unsigned int N,
                                                           const Scalar4* d_pos,
                                                           const unsigned int* d_n_neigh,
                                                           const unsigned int* d_nlist,
                                                           const size_t* d_head_list,
                                                           Scalar* d_coord,
                                                           unsigned int reactive_typeid,
                                                           BoxDim box,
                                                           reactive_lj_params params)
    {
    // Pass 1: one logical warp computes one particle's scalar coordination.
    const unsigned int lane = threadIdx.x % tpp;
    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        if (lane == 0)
            {
            d_coord[idx] = Scalar(0.0);
            }
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];

    Scalar coord = Scalar(0.0);

    for (unsigned int k = lane; k < n_neigh; k += tpp)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        const unsigned int type_j = __scalar_as_int(pos_j.w);
        if (type_j != reactive_typeid)
            {
            continue;
            }

        Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);
        if (rsq >= params.neighbor_rcut_sq)
            {
            continue;
            }

        Scalar r = Scalar(0.0);
        if (rsq > Scalar(0.0))
            {
            const Scalar inv_r = fast::rsqrt(rsq);
            r = rsq * inv_r;
            }
        const Scalar weight = computeWeight(r, params);
        if (weight == Scalar(0.0))
            {
            continue;
            }

        coord += weight;
        }

    coord = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(coord);
    if (lane == 0)
        {
        d_coord[idx] = coord;
        }
    }

template<int tpp>
__global__ void gpu_reactivelj_compute_lambda_kernel(const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const unsigned int* d_n_neigh,
                                                     const unsigned int* d_nlist,
                                                     const size_t* d_head_list,
                                                     const Scalar* d_coord,
                                                     Scalar* d_lambda,
                                                     unsigned int reactive_typeid,
                                                     BoxDim box,
                                                     reactive_lj_params params)
    {
    // Pass 2: process each pair once (j > i). Accumulate i's contribution in a
    // thread-local register and atomically add once at the end; atomically
    // update j per pair. This preserves CPU-equivalent pair-once semantics.
    const unsigned int lane = threadIdx.x % tpp;
    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];
    const Scalar Ci = __ldg(d_coord + idx);
    Scalar lambda_local = Scalar(0.0);

    for (unsigned int k = lane; k < n_neigh; k += tpp)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N || j <= idx)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        const unsigned int type_j = __scalar_as_int(pos_j.w);
        if (type_j != reactive_typeid)
            {
            continue;
            }

        Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);
        if (rsq >= params.r_cut_sq)
            {
            continue;
            }

        Scalar r = Scalar(0.0);
        if (rsq > Scalar(0.0))
            {
            const Scalar inv_r = fast::rsqrt(rsq);
            r = rsq * inv_r;
            }
        Scalar w_ij = Scalar(0.0);
        if (r < params.weakening_outer)
            {
            w_ij = computeWeight(r, params);
            }

        const Scalar Cj = __ldg(d_coord + j);
        const Scalar raw = (Ci - w_ij) + (Cj - w_ij);

        Scalar C_exc = Scalar(0.0);
        Scalar gate = Scalar(0.0);
        evaluateCexcSmooth(raw, C_exc, gate);

        const Scalar dU_dCexc = evaluate_dU_dCexc(r, C_exc, params);
        const Scalar contrib = gate * dU_dCexc;

        lambda_local += contrib;
        atomicAddScalar(d_lambda + j, contrib);
        }

    lambda_local = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(lambda_local);
    if (lane == 0 && lambda_local != Scalar(0.0))
        {
        atomicAddScalar(d_lambda + idx, lambda_local);
        }
    }

template<bool compute_virial, int tpp>
__global__ void gpu_reactivelj_compute_forces_kernel(const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const unsigned int* d_n_neigh,
                                                     const unsigned int* d_nlist,
                                                     const size_t* d_head_list,
                                                     const Scalar* d_coord,
                                                     const Scalar* d_lambda,
                                                     Scalar4* d_force,
                                                     Scalar* d_virial,
                                                     const size_t virial_pitch,
                                                     unsigned int reactive_typeid,
                                                     BoxDim box,
                                                     reactive_lj_params params)
    {
    // Pass 3: one logical warp evaluates pair force, coordination force, and energy/virial.
    const unsigned int lane = threadIdx.x % tpp;
    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        return;
        }

    const Scalar coordination_i = __ldg(d_coord + idx);
    const Scalar lambda_i = __ldg(d_lambda + idx);

    Scalar3 fi = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar pei = Scalar(0.0);
    Scalar virial_xx = Scalar(0.0);
    Scalar virial_xy = Scalar(0.0);
    Scalar virial_xz = Scalar(0.0);
    Scalar virial_yy = Scalar(0.0);
    Scalar virial_yz = Scalar(0.0);
    Scalar virial_zz = Scalar(0.0);

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];

    for (unsigned int k = lane; k < n_neigh; k += tpp)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        const unsigned int type_j = __scalar_as_int(pos_j.w);
        if (type_j != reactive_typeid)
            {
            continue;
            }

        Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);
        if (rsq >= params.neighbor_rcut_sq)
            {
            continue;
            }

        const Scalar small_r = Scalar(1e-7);
        Scalar r = Scalar(0.0);
        if (rsq > Scalar(0.0))
            {
            const Scalar inv_r = fast::rsqrt(rsq);
            r = rsq * inv_r;
            }
        if (r < small_r)
            {
            r = small_r;
            }

        // Early-out gate: if neither pair nor coordination terms can contribute,
        // skip all expensive weakening / smoothing math.
        const bool in_pair = rsq < params.r_cut_sq;
        const bool in_coord_window = r > params.weakening_inner && r < params.weakening_outer;
        if (!in_pair && !in_coord_window)
            {
            continue;
            }

        Scalar pair_energy = Scalar(0.0);
        Scalar force_divr = Scalar(0.0);

        Scalar weight = Scalar(0.0);
        Scalar dw_dr = Scalar(0.0);
        if (in_coord_window)
            {
            computeWeightAndDerivative(r, params, weight, dw_dr);
            }
        else if (r < params.weakening_outer)
            {
            weight = computeWeight(r, params);
            }
        const Scalar coordination_j = __ldg(d_coord + j);
        const Scalar raw = (coordination_i - weight) + (coordination_j - weight);

        Scalar C_exc = Scalar(0.0);
        Scalar gate = Scalar(0.0);
        evaluateCexcSmooth(raw, C_exc, gate);

        if (in_pair)
            {
            const Scalar weakening = evaluateWeakening(C_exc, params);

            const Scalar r2inv = Scalar(1.0) / (r * r);
            const Scalar r6inv = r2inv * r2inv * r2inv;
            const Scalar r12inv = r6inv * r6inv;
            const Scalar base_energy
                = params.lj_energy12 * r12inv - params.lj_energy6 * r6inv - params.energy_shift;
            const Scalar repulsive_energy
                = (base_energy > Scalar(0.0)) ? base_energy : Scalar(0.0);
            const Scalar attractive_energy
                = (base_energy < Scalar(0.0)) ? base_energy : Scalar(0.0);
            pair_energy = repulsive_energy + weakening * attractive_energy;

            const Scalar force_scale = (base_energy <= Scalar(0.0)) ? weakening : Scalar(1.0);
            const Scalar base_force_divr
                = params.lj_force12 * r12inv * r2inv - params.lj_force6 * r6inv * r2inv;
            force_divr = force_scale * base_force_divr;

            if (params.smooth_elbow)
                {
                // Inside this interval we replace the sharp elbow with a Hermite
                // segment to keep force/energy transitions well-behaved.
                const Scalar delta = evaluateSmoothDelta(weakening, params);
                const Scalar r1 = params.r_elbow - delta;
                const Scalar r2 = params.r_elbow + delta;
                const Scalar width = r2 - r1;
                if (delta > params.smooth_delta_tol && width > params.smooth_delta_tol && r > r1
                    && r < r2)
                    {
                    const Scalar U1 = evaluateEnergy(r1, params);
                    const Scalar dU1 = -evaluateForceMagnitude(r1, params);
                    const Scalar U2 = weakening * evaluateEnergy(r2, params);
                    const Scalar dU2 = -weakening * evaluateForceMagnitude(r2, params);
                    const Scalar t = (r - r1) / width;

                    const Scalar h00
                        = Scalar(2.0) * t * t * t - Scalar(3.0) * t * t + Scalar(1.0);
                    const Scalar h10 = t * t * t - Scalar(2.0) * t * t + t;
                    const Scalar h01
                        = -Scalar(2.0) * t * t * t + Scalar(3.0) * t * t;
                    const Scalar h11 = t * t * t - t * t;

                    pair_energy = h00 * U1 + h10 * width * dU1 + h01 * U2 + h11 * width * dU2;

                    const Scalar dh00 = Scalar(6.0) * t * t - Scalar(6.0) * t;
                    const Scalar dh10 = Scalar(3.0) * t * t - Scalar(4.0) * t + Scalar(1.0);
                    const Scalar dh01 = -Scalar(6.0) * t * t + Scalar(6.0) * t;
                    const Scalar dh11 = Scalar(3.0) * t * t - Scalar(2.0) * t;

                    const Scalar dU_dr = (dh00 * U1 + dh10 * width * dU1 + dh01 * U2
                                          + dh11 * width * dU2)
                                         / width;
                    const Scalar force_mag = -dU_dr;
                    force_divr = force_mag / r;
                    }
                }
            }

        Scalar force_divr_coord = Scalar(0.0);
        if (in_coord_window)
            {
            Scalar dU_dCexc_pair = Scalar(0.0);
            if (in_pair)
                {
                dU_dCexc_pair = evaluate_dU_dCexc(r, C_exc, params);
                }
            const Scalar pair_self_contrib = gate * dU_dCexc_pair;

            const Scalar lambda_sum = lambda_i + __ldg(d_lambda + j);
            const Scalar lambda_other_sum
                = lambda_sum - Scalar(2.0) * pair_self_contrib;
            const Scalar force_mag_coord = -(lambda_other_sum * dw_dr);
            force_divr_coord = force_mag_coord / r;
            }

        const Scalar total_force_divr = force_divr + force_divr_coord;

        fi.x += dx.x * total_force_divr;
        fi.y += dx.y * total_force_divr;
        fi.z += dx.z * total_force_divr;

        pei += Scalar(0.5) * pair_energy;

        if constexpr (compute_virial)
            {
            const Scalar force_div2r = Scalar(0.5) * total_force_divr;
            virial_xx += force_div2r * dx.x * dx.x;
            virial_xy += force_div2r * dx.x * dx.y;
            virial_xz += force_div2r * dx.x * dx.z;
            virial_yy += force_div2r * dx.y * dx.y;
            virial_yz += force_div2r * dx.y * dx.z;
            virial_zz += force_div2r * dx.z * dx.z;
            }
        }

    fi.x = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(fi.x);
    fi.y = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(fi.y);
    fi.z = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(fi.z);
    pei = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(pei);

    if constexpr (compute_virial)
        {
        virial_xx = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(virial_xx);
        virial_xy = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(virial_xy);
        virial_xz = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(virial_xz);
        virial_yy = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(virial_yy);
        virial_yz = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(virial_yz);
        virial_zz = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(virial_zz);
        }

    if (lane == 0)
        {
        d_force[idx].x += fi.x;
        d_force[idx].y += fi.y;
        d_force[idx].z += fi.z;
        d_force[idx].w += pei;

        if constexpr (compute_virial)
            {
            d_virial[0 * virial_pitch + idx] += virial_xx;
            d_virial[1 * virial_pitch + idx] += virial_xy;
            d_virial[2 * virial_pitch + idx] += virial_xz;
            d_virial[3 * virial_pitch + idx] += virial_yy;
            d_virial[4 * virial_pitch + idx] += virial_yz;
            d_virial[5 * virial_pitch + idx] += virial_zz;
            }
        }
    }

hipError_t gpu_reactivelj_get_coordination_max_block_size(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock((const void*)gpu_reactivelj_compute_coordination_kernel<1>,
                                       max_threads_cache,
                                       cache_mutex,
                                       max_block_size);
    }

hipError_t gpu_reactivelj_get_lambda_max_block_size(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock((const void*)gpu_reactivelj_compute_lambda_kernel<1>,
                                       max_threads_cache,
                                       cache_mutex,
                                       max_block_size);
    }

namespace
    {
unsigned int getRunBlockSize(unsigned int block_size,
                             unsigned int max_threads_per_block,
                             unsigned int threads_per_particle)
    {
    unsigned int run_block_size = std::min(block_size, max_threads_per_block);
    run_block_size -= run_block_size % threads_per_particle;
    return run_block_size;
    }

template<int tpp> hipError_t getCoordinationMaxBlockSize(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock((const void*)gpu_reactivelj_compute_coordination_kernel<tpp>,
                                       max_threads_cache,
                                       cache_mutex,
                                       max_block_size);
    }

template<int tpp> hipError_t getLambdaMaxBlockSize(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock((const void*)gpu_reactivelj_compute_lambda_kernel<tpp>,
                                       max_threads_cache,
                                       cache_mutex,
                                       max_block_size);
    }

template<bool compute_virial, int tpp>
hipError_t getForceMaxBlockSize(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock(
        (const void*)gpu_reactivelj_compute_forces_kernel<compute_virial, tpp>,
        max_threads_cache,
        cache_mutex,
        max_block_size);
    }

hipError_t gpu_reactivelj_get_force_variant_max_block_size(bool compute_virial,
                                                           unsigned int& max_block_size)
    {
    if (compute_virial)
        {
        return getForceMaxBlockSize<true, 1>(max_block_size);
        }

    return getForceMaxBlockSize<false, 1>(max_block_size);
    }
    } // end namespace

hipError_t gpu_reactivelj_get_force_max_block_size(unsigned int& max_block_size)
    {
    unsigned int max_block_size_with_virial = 0;
    hipError_t err
        = gpu_reactivelj_get_force_variant_max_block_size(true, max_block_size_with_virial);
    if (err != hipSuccess)
        {
        return err;
        }

    unsigned int max_block_size_without_virial = 0;
    err = gpu_reactivelj_get_force_variant_max_block_size(false, max_block_size_without_virial);
    if (err != hipSuccess)
        {
        return err;
        }

    // Use the stricter limit so one autotuner range is valid for both variants.
    max_block_size = std::min(max_block_size_with_virial, max_block_size_without_virial);
    return hipSuccess;
    }

namespace
    {
template<int tpp> struct CoordinationLauncher
    {
    static hipError_t launch(const unsigned int N,
                             const Scalar4* d_pos,
                             const unsigned int* d_n_neigh,
                             const unsigned int* d_nlist,
                             const size_t* d_head_list,
                             Scalar* d_coord,
                             unsigned int reactive_typeid,
                             BoxDim box,
                             reactive_lj_params params,
                             unsigned int block_size,
                             unsigned int threads_per_particle)
        {
        if (threads_per_particle == tpp)
            {
            unsigned int max_threads_per_block = 0;
            hipError_t cache_err = getCoordinationMaxBlockSize<tpp>(max_threads_per_block);
            if (cache_err != hipSuccess)
                {
                return cache_err;
                }
            const unsigned int run_block_size
                = getRunBlockSize(block_size, max_threads_per_block, tpp);
            if (run_block_size == 0)
                {
                return hipErrorInvalidConfiguration;
                }

            const unsigned int particles_per_block = run_block_size / tpp;
            dim3 grid((N + particles_per_block - 1) / particles_per_block, 1, 1);
            dim3 threads(run_block_size, 1, 1);
            hipLaunchKernelGGL((gpu_reactivelj_compute_coordination_kernel<tpp>),
                               grid,
                               threads,
                               0,
                               0,
                               N,
                               d_pos,
                               d_n_neigh,
                               d_nlist,
                               d_head_list,
                               d_coord,
                               reactive_typeid,
                               box,
                               params);
            return hipGetLastError();
            }
        return CoordinationLauncher<tpp / 2>::launch(N,
                                                     d_pos,
                                                     d_n_neigh,
                                                     d_nlist,
                                                     d_head_list,
                                                     d_coord,
                                                     reactive_typeid,
                                                     box,
                                                     params,
                                                     block_size,
                                                     threads_per_particle);
        }
    };

template<> struct CoordinationLauncher<0>
    {
    static hipError_t launch(const unsigned int,
                             const Scalar4*,
                             const unsigned int*,
                             const unsigned int*,
                             const size_t*,
                             Scalar*,
                             unsigned int,
                             BoxDim,
                             reactive_lj_params,
                             unsigned int,
                             unsigned int)
        {
        return hipErrorInvalidConfiguration;
        }
    };

template<int tpp> struct LambdaLauncher
    {
    static hipError_t launch(const unsigned int N,
                             const Scalar4* d_pos,
                             const unsigned int* d_n_neigh,
                             const unsigned int* d_nlist,
                             const size_t* d_head_list,
                             const Scalar* d_coord,
                             Scalar* d_lambda,
                             unsigned int reactive_typeid,
                             BoxDim box,
                             reactive_lj_params params,
                             unsigned int block_size,
                             unsigned int threads_per_particle)
        {
        if (threads_per_particle == tpp)
            {
            unsigned int max_threads_per_block = 0;
            hipError_t cache_err = getLambdaMaxBlockSize<tpp>(max_threads_per_block);
            if (cache_err != hipSuccess)
                {
                return cache_err;
                }
            const unsigned int run_block_size
                = getRunBlockSize(block_size, max_threads_per_block, tpp);
            if (run_block_size == 0)
                {
                return hipErrorInvalidConfiguration;
                }

            const unsigned int particles_per_block = run_block_size / tpp;
            dim3 grid((N + particles_per_block - 1) / particles_per_block, 1, 1);
            dim3 threads(run_block_size, 1, 1);
            hipLaunchKernelGGL((gpu_reactivelj_compute_lambda_kernel<tpp>),
                               grid,
                               threads,
                               0,
                               0,
                               N,
                               d_pos,
                               d_n_neigh,
                               d_nlist,
                               d_head_list,
                               d_coord,
                               d_lambda,
                               reactive_typeid,
                               box,
                               params);
            return hipGetLastError();
            }
        return LambdaLauncher<tpp / 2>::launch(N,
                                               d_pos,
                                               d_n_neigh,
                                               d_nlist,
                                               d_head_list,
                                               d_coord,
                                               d_lambda,
                                               reactive_typeid,
                                               box,
                                               params,
                                               block_size,
                                               threads_per_particle);
        }
    };

template<> struct LambdaLauncher<0>
    {
    static hipError_t launch(const unsigned int,
                             const Scalar4*,
                             const unsigned int*,
                             const unsigned int*,
                             const size_t*,
                             const Scalar*,
                             Scalar*,
                             unsigned int,
                             BoxDim,
                             reactive_lj_params,
                             unsigned int,
                             unsigned int)
        {
        return hipErrorInvalidConfiguration;
        }
    };

template<bool compute_virial, int tpp> struct ForceLauncher
    {
    static hipError_t launch(const unsigned int N,
                             const Scalar4* d_pos,
                             const unsigned int* d_n_neigh,
                             const unsigned int* d_nlist,
                             const size_t* d_head_list,
                             const Scalar* d_coord,
                             const Scalar* d_lambda,
                             Scalar4* d_force,
                             Scalar* d_virial,
                             const size_t virial_pitch,
                             unsigned int reactive_typeid,
                             BoxDim box,
                             reactive_lj_params params,
                             unsigned int block_size,
                             unsigned int threads_per_particle)
        {
        if (threads_per_particle == tpp)
            {
            unsigned int max_threads_per_block = 0;
            hipError_t cache_err = getForceMaxBlockSize<compute_virial, tpp>(max_threads_per_block);
            if (cache_err != hipSuccess)
                {
                return cache_err;
                }
            const unsigned int run_block_size
                = getRunBlockSize(block_size, max_threads_per_block, tpp);
            if (run_block_size == 0)
                {
                return hipErrorInvalidConfiguration;
                }

            const unsigned int particles_per_block = run_block_size / tpp;
            dim3 grid((N + particles_per_block - 1) / particles_per_block, 1, 1);
            dim3 threads(run_block_size, 1, 1);
            hipLaunchKernelGGL((gpu_reactivelj_compute_forces_kernel<compute_virial, tpp>),
                               grid,
                               threads,
                               0,
                               0,
                               N,
                               d_pos,
                               d_n_neigh,
                               d_nlist,
                               d_head_list,
                               d_coord,
                               d_lambda,
                               d_force,
                               d_virial,
                               virial_pitch,
                               reactive_typeid,
                               box,
                               params);
            return hipGetLastError();
            }
        return ForceLauncher<compute_virial, tpp / 2>::launch(N,
                                                              d_pos,
                                                              d_n_neigh,
                                                              d_nlist,
                                                              d_head_list,
                                                              d_coord,
                                                              d_lambda,
                                                              d_force,
                                                              d_virial,
                                                              virial_pitch,
                                                              reactive_typeid,
                                                              box,
                                                              params,
                                                              block_size,
                                                              threads_per_particle);
        }
    };

template<bool compute_virial> struct ForceLauncher<compute_virial, 0>
    {
    static hipError_t launch(const unsigned int,
                             const Scalar4*,
                             const unsigned int*,
                             const unsigned int*,
                             const size_t*,
                             const Scalar*,
                             const Scalar*,
                             Scalar4*,
                             Scalar*,
                             const size_t,
                             unsigned int,
                             BoxDim,
                             reactive_lj_params,
                             unsigned int,
                             unsigned int)
        {
        return hipErrorInvalidConfiguration;
        }
    };
    } // namespace

hipError_t gpu_reactivelj_compute_coordination(const unsigned int N,
                                               const Scalar4* d_pos,
                                               const unsigned int* d_n_neigh,
                                               const unsigned int* d_nlist,
                                               const size_t* d_head_list,
                                               Scalar* d_coord,
                                               unsigned int reactive_typeid,
                                               BoxDim box,
                                               reactive_lj_params params,
                                               unsigned int block_size,
                                               unsigned int threads_per_particle)
    {
    return CoordinationLauncher<gpu_reactivelj_max_tpp>::launch(N,
                                                                d_pos,
                                                                d_n_neigh,
                                                                d_nlist,
                                                                d_head_list,
                                                                d_coord,
                                                                reactive_typeid,
                                                                box,
                                                                params,
                                                                block_size,
                                                                threads_per_particle);
    }

hipError_t gpu_reactivelj_compute_lambda(const unsigned int N,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_n_neigh,
                                         const unsigned int* d_nlist,
                                         const size_t* d_head_list,
                                         const Scalar* d_coord,
                                         Scalar* d_lambda,
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_lj_params params,
                                         unsigned int block_size,
                                         unsigned int threads_per_particle)
    {
    return LambdaLauncher<gpu_reactivelj_max_tpp>::launch(N,
                                                          d_pos,
                                                          d_n_neigh,
                                                          d_nlist,
                                                          d_head_list,
                                                          d_coord,
                                                          d_lambda,
                                                          reactive_typeid,
                                                          box,
                                                          params,
                                                          block_size,
                                                          threads_per_particle);
    }

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
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_lj_params params,
                                         unsigned int compute_virial,
                                         unsigned int block_size,
                                         unsigned int threads_per_particle)
    {
    const bool do_compute_virial = (compute_virial != 0);

    if (do_compute_virial)
        {
        return ForceLauncher<true, gpu_reactivelj_max_tpp>::launch(N,
                                                                   d_pos,
                                                                   d_n_neigh,
                                                                   d_nlist,
                                                                   d_head_list,
                                                                   d_coord,
                                                                   d_lambda,
                                                                   d_force,
                                                                   d_virial,
                                                                   virial_pitch,
                                                                   reactive_typeid,
                                                                   box,
                                                                   params,
                                                                   block_size,
                                                                   threads_per_particle);
        }
    return ForceLauncher<false, gpu_reactivelj_max_tpp>::launch(N,
                                                                d_pos,
                                                                d_n_neigh,
                                                                d_nlist,
                                                                d_head_list,
                                                                d_coord,
                                                                d_lambda,
                                                                d_force,
                                                                d_virial,
                                                                virial_pitch,
                                                                reactive_typeid,
                                                                box,
                                                                params,
                                                                block_size,
                                                                threads_per_particle);
    }

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
