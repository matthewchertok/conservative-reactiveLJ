// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "ReactiveLJForceComputeGPU.cuh"

#include "hoomd/HOOMDMath.h"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"

#define DEVICE __device__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
// Smoothly project raw excess coordination onto max(raw, 0). The gate term is
// dC_exc/draw and is reused in chain-rule terms later on.
DEVICE inline void evaluateCexcSmooth(Scalar raw, Scalar& C_exc, Scalar& gate)
    {
    const Scalar eps = Scalar(1e-6);
    const Scalar s = slow::sqrt(raw * raw + eps * eps);
    C_exc = Scalar(0.5) * (raw + s);
    gate = Scalar(0.5) * (Scalar(1.0) + raw / s);
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

    const Scalar pi = Scalar(3.141592653589793238462643383279502884L);
    const Scalar fraction = (distance - params.weakening_inner) * params.inverse_span;
    const Scalar angle = pi * fraction;
    return Scalar(0.5) * (Scalar(1.0) + slow::cos(angle));
    }

// Derivative of the cosine coordination window with respect to distance.
DEVICE inline Scalar evaluateWeightDerivative(Scalar distance, const reactive_lj_params& params)
    {
    if (distance >= params.weakening_outer || distance <= params.weakening_inner)
        {
        return Scalar(0.0);
        }

    const Scalar pi = Scalar(3.141592653589793238462643383279502884L);
    const Scalar fraction = (distance - params.weakening_inner) * params.inverse_span;
    const Scalar angle = pi * fraction;
    return -Scalar(0.5) * slow::sin(angle) * (pi * params.inverse_span);
    }

// Shifted Lennard-Jones energy used by the ReactiveLJ pair branch.
DEVICE inline Scalar evaluateEnergy(Scalar distance, const reactive_lj_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar sr = params.sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;
    return Scalar(4.0) * params.epsilon * (sr12 - sr6) - params.energy_shift;
    }

// Positive magnitude of the pair force, F(r), before multiplying by direction.
DEVICE inline Scalar evaluateForceMagnitude(Scalar distance, const reactive_lj_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar sr = params.sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;
    return Scalar(24.0) * params.epsilon * inv_r * (Scalar(2.0) * sr12 - sr6);
    }

// Radial derivative dF/dr used in the smoothing-region chain rule.
DEVICE inline Scalar evaluateForceDerivative(Scalar distance, const reactive_lj_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar inv_r2 = inv_r * inv_r;

    const Scalar sr = params.sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;

    return Scalar(24.0) * params.epsilon * inv_r2
           * (Scalar(-26.0) * sr12 + Scalar(7.0) * sr6);
    }

// Compute dU/dC_exc for one directed pair. This drives the many-body lambda
// accumulation and, through lambda, the coordination force in pass 3.
DEVICE inline Scalar evaluate_dU_dCexc(Scalar r_in,
                                      Scalar C_exc,
                                      const reactive_lj_params& params,
                                      unsigned int* error_flag)
    {
    const Scalar r = (r_in > params.smooth_r_min) ? r_in : params.smooth_r_min;

    const Scalar W = slow::pow(Scalar(1.0) + C_exc, -params.weakening_exponent);

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
        const Scalar one_minus_W
            = (Scalar(1.0) > W) ? (Scalar(1.0) - W) : Scalar(0.0);
        delta = params.smooth_kappa * params.sigma * slow::pow(one_minus_W, params.smooth_beta);

        r1 = params.r_elbow - delta;
        r2 = params.r_elbow + delta;

        if (r1 <= params.smooth_r_min)
            {
            if (atomicCAS(error_flag, 0u, 1u) == 0u)
                {
                }
            return Scalar(0.0);
            }
        if (r2 >= params.r_cut)
            {
            if (atomicCAS(error_flag, 0u, 2u) == 0u)
                {
                }
            return Scalar(0.0);
            }
        else
            {
            L = r2 - r1;
            // Outside this narrow region we fall back to the simpler branch.
            if (delta > params.smooth_delta_tol && L > params.smooth_delta_tol && r > r1
                && r < r2)
                {
                smoothing_active = true;
                }
            }
        }

    if (!smoothing_active)
        {
        // Unsmoothed branch: only the attractive part is weakened by W(C_exc).
        const Scalar U_LJ = evaluateEnergy(r, params);
        const Scalar U_att = (U_LJ < Scalar(0.0)) ? U_LJ : Scalar(0.0);
        return U_att * dW_dC;
        }

    const Scalar one_minus_W
        = (Scalar(1.0) > W) ? (Scalar(1.0) - W) : Scalar(0.0);

    Scalar ddelta_dW = Scalar(0.0);
    if (one_minus_W > Scalar(0.0))
        {
        ddelta_dW = -params.smooth_kappa * params.sigma * params.smooth_beta
                    * slow::pow(one_minus_W, params.smooth_beta - Scalar(1.0));
        }

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
    // Pass 1: each thread computes one particle's scalar coordination.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        d_coord[idx] = Scalar(0.0);
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];

    Scalar coord = Scalar(0.0);

    for (unsigned int k = 0; k < n_neigh; ++k)
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

    d_coord[idx] = coord;
    }

__global__ void gpu_reactivelj_compute_lambda_kernel(const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const unsigned int* d_n_neigh,
                                                     const unsigned int* d_nlist,
                                                     const size_t* d_head_list,
                                                     const Scalar* d_coord,
                                                     Scalar* d_lambda,
                                                     Scalar* d_pair_contrib,
                                                     unsigned int* d_error_flag,
                                                     unsigned int reactive_typeid,
                                                     BoxDim box,
                                                     reactive_lj_params params)
    {
    // Pass 2: lambda_i = sum_j gate_ij * dU_ij/dC_exc, and cache each directed
    // term so pass 3 can reuse it.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        d_lambda[idx] = Scalar(0.0);
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];
    const Scalar Ci = __ldg(d_coord + idx);

    Scalar lambda = Scalar(0.0);

    for (unsigned int k = 0; k < n_neigh; ++k)
        {
        const size_t nlist_idx = head + k;
        // Default to zero so pass 3 can safely read this entry.
        d_pair_contrib[nlist_idx] = Scalar(0.0);

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

        const Scalar dU_dCexc = evaluate_dU_dCexc(r, C_exc, params, d_error_flag);
        const Scalar contrib = gate * dU_dCexc;

        lambda += contrib;
        d_pair_contrib[nlist_idx] = contrib;
        }

    d_lambda[idx] = lambda;
    }

__global__ void gpu_reactivelj_compute_forces_kernel(const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const unsigned int* d_n_neigh,
                                                     const unsigned int* d_nlist,
                                                     const size_t* d_head_list,
                                                     const Scalar* d_coord,
                                                     const Scalar* d_lambda,
                                                     const Scalar* d_pair_contrib,
                                                     Scalar4* d_force,
                                                     Scalar* d_virial,
                                                     const size_t virial_pitch,
                                                     unsigned int* d_error_flag,
                                                     unsigned int reactive_typeid,
                                                     BoxDim box,
                                                     reactive_lj_params params,
                                                     unsigned int compute_virial)
    {
    // Pass 3: evaluate pair force, coordination force, and energy/virial.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

    for (unsigned int k = 0; k < n_neigh; ++k)
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

        const Scalar weight = (r < params.weakening_outer) ? computeWeight(r, params) : Scalar(0.0);
        const Scalar coordination_j = __ldg(d_coord + j);
        const Scalar raw = (coordination_i - weight) + (coordination_j - weight);

        Scalar C_exc = Scalar(0.0);
        Scalar gate = Scalar(0.0);
        evaluateCexcSmooth(raw, C_exc, gate);

        if (in_pair)
            {
            const Scalar weakening
                = slow::pow(Scalar(1.0) + C_exc, -params.weakening_exponent);

            const Scalar base_energy = evaluateEnergy(r, params);
            const Scalar repulsive_energy
                = (base_energy > Scalar(0.0)) ? base_energy : Scalar(0.0);
            const Scalar attractive_energy
                = (base_energy < Scalar(0.0)) ? base_energy : Scalar(0.0);
            pair_energy = repulsive_energy + weakening * attractive_energy;

            const Scalar force_scale = (base_energy <= Scalar(0.0)) ? weakening : Scalar(1.0);
            const Scalar inv_r = Scalar(1.0) / r;
            const Scalar base_force_divr = evaluateForceMagnitude(r, params) * inv_r;
            force_divr = force_scale * base_force_divr;

            if (params.smooth_elbow)
                {
                // Inside this interval we replace the sharp elbow with a Hermite
                // segment to keep force/energy transitions well-behaved.
                const Scalar one_minus_w
                    = (Scalar(1.0) > weakening) ? (Scalar(1.0) - weakening) : Scalar(0.0);
                const Scalar delta
                    = params.smooth_kappa * params.sigma * slow::pow(one_minus_w, params.smooth_beta);
                const Scalar r1 = params.r_elbow - delta;
                const Scalar r2 = params.r_elbow + delta;
                const Scalar width = r2 - r1;
                if (r1 <= params.smooth_r_min)
                    {
                    if (atomicCAS(d_error_flag, 0u, 1u) == 0u)
                        {
                        }
                    return;
                    }
                if (r2 >= params.r_cut)
                    {
                    if (atomicCAS(d_error_flag, 0u, 2u) == 0u)
                        {
                        }
                    return;
                    }
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
            Scalar pair_self_contrib = Scalar(0.0);
            if (in_pair)
                {
                const size_t nlist_idx = head + k;
                // Fast path: reuse gate*dU/dCexc already computed in pass 2.
                pair_self_contrib = __ldg(d_pair_contrib + nlist_idx);
                }
            else
                {
                // Fallback for neighbors that contribute to dw/dr but were
                // outside the pair cutoff in pass 2.
                const Scalar dU_dCexc_pair = evaluate_dU_dCexc(r, C_exc, params, d_error_flag);
                pair_self_contrib = gate * dU_dCexc_pair;
                }

            const Scalar lambda_sum = lambda_i + __ldg(d_lambda + j);
            const Scalar lambda_other_sum
                = lambda_sum - Scalar(2.0) * pair_self_contrib;
            const Scalar dw_dr = evaluateWeightDerivative(r, params);
            const Scalar force_mag_coord = -(lambda_other_sum * dw_dr);
            force_divr_coord = force_mag_coord / r;
            }

        const Scalar total_force_divr = force_divr + force_divr_coord;

        fi.x += dx.x * total_force_divr;
        fi.y += dx.y * total_force_divr;
        fi.z += dx.z * total_force_divr;

        pei += Scalar(0.5) * pair_energy;

        if (compute_virial)
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

    d_force[idx].x += fi.x;
    d_force[idx].y += fi.y;
    d_force[idx].z += fi.z;
    d_force[idx].w += pei;

    if (compute_virial)
        {
        d_virial[0 * virial_pitch + idx] += virial_xx;
        d_virial[1 * virial_pitch + idx] += virial_xy;
        d_virial[2 * virial_pitch + idx] += virial_xz;
        d_virial[3 * virial_pitch + idx] += virial_yy;
        d_virial[4 * virial_pitch + idx] += virial_yz;
        d_virial[5 * virial_pitch + idx] += virial_zz;
        }
    }

hipError_t gpu_reactivelj_compute_coordination(const unsigned int N,
                                               const Scalar4* d_pos,
                                               const unsigned int* d_n_neigh,
                                               const unsigned int* d_nlist,
                                               const size_t* d_head_list,
                                               Scalar* d_coord,
                                               unsigned int reactive_typeid,
                                               BoxDim box,
                                               reactive_lj_params params,
                                               unsigned int block_size)
    {
    dim3 grid((N + block_size - 1) / block_size, 1, 1);
    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL((gpu_reactivelj_compute_coordination_kernel),
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

hipError_t gpu_reactivelj_compute_lambda(const unsigned int N,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_n_neigh,
                                         const unsigned int* d_nlist,
                                         const size_t* d_head_list,
                                         const Scalar* d_coord,
                                         Scalar* d_lambda,
                                         Scalar* d_pair_contrib,
                                         unsigned int* d_error_flag,
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_lj_params params,
                                         unsigned int block_size)
    {
    dim3 grid((N + block_size - 1) / block_size, 1, 1);
    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL((gpu_reactivelj_compute_lambda_kernel),
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
                       d_pair_contrib,
                       d_error_flag,
                       reactive_typeid,
                       box,
                       params);
    return hipGetLastError();
    }

hipError_t gpu_reactivelj_compute_forces(const unsigned int N,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_n_neigh,
                                         const unsigned int* d_nlist,
                                         const size_t* d_head_list,
                                         const Scalar* d_coord,
                                         const Scalar* d_lambda,
                                         const Scalar* d_pair_contrib,
                                         Scalar4* d_force,
                                         Scalar* d_virial,
                                         const size_t virial_pitch,
                                         unsigned int* d_error_flag,
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_lj_params params,
                                         unsigned int compute_virial,
                                         unsigned int block_size)
    {
    dim3 grid((N + block_size - 1) / block_size, 1, 1);
    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL((gpu_reactivelj_compute_forces_kernel),
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
                       d_pair_contrib,
                       d_force,
                       d_virial,
                       virial_pitch,
                       d_error_flag,
                       reactive_typeid,
                       box,
                       params,
                       compute_virial);
    return hipGetLastError();
    }

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
