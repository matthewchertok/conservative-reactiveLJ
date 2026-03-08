// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"

#include "CBTTableDihedralForceGPU.cuh"

#include <algorithm>
#include <assert.h>
#include <mutex>
#include <unordered_map>

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

__device__ inline Scalar3 cross3(const Scalar3& a, const Scalar3& b)
    {
    return make_scalar3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

__device__ inline bool compute_angle_terms(const Scalar3& u,
                                           const Scalar3& v,
                                           const Scalar eps,
                                           Scalar& sin_theta,
                                           Scalar& cos_theta,
                                           Scalar3& dtheta_du,
                                           Scalar3& dtheta_dv)
    {
    const Scalar u_norm_sq = dot(u, u);
    const Scalar v_norm_sq = dot(v, v);
    const Scalar u_norm = fast::sqrt(u_norm_sq);
    const Scalar v_norm = fast::sqrt(v_norm_sq);

    if (u_norm <= eps || v_norm <= eps)
        {
        return false;
        }

    const Scalar inv_uv = Scalar(1.0) / (u_norm * v_norm);
    cos_theta = dot(u, v) * inv_uv;
    if (cos_theta > Scalar(1.0))
        {
        cos_theta = Scalar(1.0);
        }
    else if (cos_theta < Scalar(-1.0))
        {
        cos_theta = Scalar(-1.0);
        }

    const Scalar3 uxv = cross3(u, v);
    sin_theta = fast::sqrt(dot(uxv, uxv)) * inv_uv;
    if (sin_theta < Scalar(0.0))
        {
        sin_theta = Scalar(0.0);
        }
    else if (sin_theta > Scalar(1.0))
        {
        sin_theta = Scalar(1.0);
        }

    if (sin_theta <= eps)
        {
        return false;
        }

    const Scalar3 term_u = v * inv_uv - (cos_theta / u_norm_sq) * u;
    const Scalar3 term_v = u * inv_uv - (cos_theta / v_norm_sq) * v;

    dtheta_du = -(Scalar(1.0) / sin_theta) * term_u;
    dtheta_dv = -(Scalar(1.0) / sin_theta) * term_v;
    return true;
    }

__device__ inline void interpolate_table(const Scalar2* d_tables,
                                         const Index2D table_value,
                                         const unsigned int table_width,
                                         const unsigned int type,
                                         const Scalar phi,
                                         Scalar& U,
                                         Scalar& tau)
    {
    const Scalar delta_phi = Scalar(2.0 * M_PI) / Scalar(table_width - 1);
    Scalar value_f = (Scalar(M_PI) + phi) / delta_phi;

    unsigned int value_i;
    Scalar f;
    if (value_f <= Scalar(0.0))
        {
        value_i = 0;
        f = Scalar(0.0);
        }
    else if (value_f >= Scalar(table_width - 1))
        {
        value_i = table_width - 2;
        f = Scalar(1.0);
        }
    else
        {
        value_i = static_cast<unsigned int>(value_f);
        f = value_f - Scalar(value_i);
        }

    const Scalar2 UT0 = __ldg(d_tables + table_value(value_i, type));
    const Scalar2 UT1 = __ldg(d_tables + table_value(value_i + 1, type));
    U = UT0.x + f * (UT1.x - UT0.x);
    tau = UT0.y + f * (UT1.y - UT0.y);
    }

    } // end namespace

__global__ void gpu_compute_cbt_table_dihedral_forces_kernel(Scalar4* d_force,
                                                             Scalar* d_virial,
                                                             const size_t virial_pitch,
                                                             const unsigned int N,
                                                             const Scalar4* d_pos,
                                                             BoxDim box,
                                                             const group_storage<4>* tlist,
                                                             const unsigned int* dihedral_ABCD,
                                                             const unsigned int pitch,
                                                             const unsigned int* n_dihedrals_list,
                                                             const Scalar2* d_tables,
                                                             const unsigned int table_width,
                                                             const Index2D table_value,
                                                             const Scalar* d_angle_exponents)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int n_dihedrals = n_dihedrals_list[idx];

    Scalar4 idx_postype = d_pos[idx];
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 pos_a, pos_b, pos_c, pos_d;

    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial_idx[6];
    for (unsigned int i = 0; i < 6; i++)
        virial_idx[i] = Scalar(0.0);

    const Scalar eps = Scalar(1e-12);

    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        group_storage<4> cur_dihedral = tlist[pitch * dihedral_idx + idx];
        unsigned int cur_ABCD = dihedral_ABCD[pitch * dihedral_idx + idx];

        int cur_dihedral_x_idx = cur_dihedral.idx[0];
        int cur_dihedral_y_idx = cur_dihedral.idx[1];
        int cur_dihedral_z_idx = cur_dihedral.idx[2];
        int cur_dihedral_type = cur_dihedral.idx[3];

        Scalar4 x_postype = d_pos[cur_dihedral_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);

        Scalar4 y_postype = d_pos[cur_dihedral_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

        Scalar4 z_postype = d_pos[cur_dihedral_z_idx];
        Scalar3 z_pos = make_scalar3(z_postype.x, z_postype.y, z_postype.z);

        if (cur_ABCD == 0)
            {
            pos_a = idx_pos;
            pos_b = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        else if (cur_ABCD == 1)
            {
            pos_b = idx_pos;
            pos_a = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        else if (cur_ABCD == 2)
            {
            pos_c = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_d = z_pos;
            }
        else
            {
            pos_d = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_c = z_pos;
            }

        const Scalar3 b1 = box.minImage(pos_b - pos_a);
        const Scalar3 b2 = box.minImage(pos_c - pos_b);
        const Scalar3 b3 = box.minImage(pos_d - pos_c);

        const Scalar3 n1 = cross3(b1, b2);
        const Scalar3 n2 = cross3(b2, b3);

        const Scalar n1_sq = dot(n1, n1);
        const Scalar n2_sq = dot(n2, n2);
        const Scalar b2_sq = dot(b2, b2);

        if (n1_sq <= eps || n2_sq <= eps || b2_sq <= eps)
            {
            continue;
            }

        const Scalar b2_norm = fast::sqrt(b2_sq);
        const Scalar3 b2_hat = b2 / b2_norm;

        const Scalar x = dot(n1, n2);
        const Scalar y = dot(cross3(n1, n2), b2_hat);
        const Scalar phi = atan2(y, x);

        const Scalar3 dphi_dr1 = -(b2_norm / n1_sq) * n1;
        const Scalar3 dphi_dr4 = +(b2_norm / n2_sq) * n2;
        const Scalar3 dphi_db2 = (dot(b1, b2) / b2_sq) * dphi_dr1 - (dot(b3, b2) / b2_sq) * dphi_dr4;
        const Scalar3 dphi_dr2 = -dphi_dr1 - dphi_db2;
        const Scalar3 dphi_dr3 = dphi_db2 - dphi_dr4;

        Scalar sin1, cos1;
        Scalar sin2, cos2;
        Scalar3 dtheta1_du, dtheta1_dv;
        Scalar3 dtheta2_du, dtheta2_dv;

        const bool valid_angle1 = compute_angle_terms(-b1, b2, eps, sin1, cos1, dtheta1_du, dtheta1_dv);
        const bool valid_angle2 = compute_angle_terms(-b2, b3, eps, sin2, cos2, dtheta2_du, dtheta2_dv);
        if (!valid_angle1 || !valid_angle2)
            {
            continue;
            }

        const Scalar m = __ldg(d_angle_exponents + cur_dihedral_type);

        Scalar torsion_energy;
        Scalar torsion_tau;
        interpolate_table(d_tables,
                          table_value,
                          table_width,
                          cur_dihedral_type,
                          phi,
                          torsion_energy,
                          torsion_tau);

        const Scalar sin1_pow_m = pow(sin1, m);
        const Scalar sin2_pow_m = pow(sin2, m);

        const Scalar gate = sin1_pow_m * sin2_pow_m;
        const Scalar energy = gate * torsion_energy;
        const Scalar dU_dphi = gate * (-torsion_tau);

        Scalar dU_dtheta1 = Scalar(0.0);
        Scalar dU_dtheta2 = Scalar(0.0);
        if (m != Scalar(0.0))
            {
            dU_dtheta1 = m * pow(sin1, m - Scalar(1.0)) * sin2_pow_m * cos1 * torsion_energy;
            dU_dtheta2 = m * sin1_pow_m * pow(sin2, m - Scalar(1.0)) * cos2 * torsion_energy;
            }

        const Scalar3 dtheta1_dr1 = dtheta1_du;
        const Scalar3 dtheta1_dr2 = -dtheta1_du - dtheta1_dv;
        const Scalar3 dtheta1_dr3 = dtheta1_dv;

        const Scalar3 dtheta2_dr2 = dtheta2_du;
        const Scalar3 dtheta2_dr3 = -dtheta2_du - dtheta2_dv;
        const Scalar3 dtheta2_dr4 = dtheta2_dv;

        const Scalar3 dU_dr1 = dU_dphi * dphi_dr1 + dU_dtheta1 * dtheta1_dr1;
        const Scalar3 dU_dr2 = dU_dphi * dphi_dr2 + dU_dtheta1 * dtheta1_dr2 + dU_dtheta2 * dtheta2_dr2;
        const Scalar3 dU_dr3 = dU_dphi * dphi_dr3 + dU_dtheta1 * dtheta1_dr3 + dU_dtheta2 * dtheta2_dr3;
        const Scalar3 dU_dr4 = dU_dphi * dphi_dr4 + dU_dtheta2 * dtheta2_dr4;

        const Scalar3 f1 = -dU_dr1;
        const Scalar3 f2 = -dU_dr2;
        const Scalar3 f3 = -dU_dr3;
        const Scalar3 f4 = -dU_dr4;

        const Scalar3 vb1 = -b1;
        const Scalar3 vb2 = b2;
        const Scalar3 vb3 = b3;

        Scalar dihedral_virial[6];
        dihedral_virial[0]
            = Scalar(0.25) * (vb1.x * f1.x + vb2.x * f3.x + (vb3.x + vb2.x) * f4.x);
        dihedral_virial[1]
            = Scalar(0.25) * (vb1.y * f1.x + vb2.y * f3.x + (vb3.y + vb2.y) * f4.x);
        dihedral_virial[2]
            = Scalar(0.25) * (vb1.z * f1.x + vb2.z * f3.x + (vb3.z + vb2.z) * f4.x);
        dihedral_virial[3]
            = Scalar(0.25) * (vb1.y * f1.y + vb2.y * f3.y + (vb3.y + vb2.y) * f4.y);
        dihedral_virial[4]
            = Scalar(0.25) * (vb1.z * f1.y + vb2.z * f3.y + (vb3.z + vb2.z) * f4.y);
        dihedral_virial[5]
            = Scalar(0.25) * (vb1.z * f1.z + vb2.z * f3.z + (vb3.z + vb2.z) * f4.z);

        if (cur_ABCD == 0)
            {
            force_idx.x += f1.x;
            force_idx.y += f1.y;
            force_idx.z += f1.z;
            }
        else if (cur_ABCD == 1)
            {
            force_idx.x += f2.x;
            force_idx.y += f2.y;
            force_idx.z += f2.z;
            }
        else if (cur_ABCD == 2)
            {
            force_idx.x += f3.x;
            force_idx.y += f3.y;
            force_idx.z += f3.z;
            }
        else
            {
            force_idx.x += f4.x;
            force_idx.y += f4.y;
            force_idx.z += f4.z;
            }

        force_idx.w += Scalar(0.25) * energy;
        for (unsigned int k = 0; k < 6; k++)
            {
            virial_idx[k] += dihedral_virial[k];
            }
        }

    d_force[idx] = force_idx;
    for (unsigned int k = 0; k < 6; k++)
        d_virial[k * virial_pitch + idx] = virial_idx[k];
    }

hipError_t gpu_cbt_table_dihedral_get_max_block_size(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock((const void*)gpu_compute_cbt_table_dihedral_forces_kernel,
                                       max_threads_cache,
                                       cache_mutex,
                                       max_block_size);
    }

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
                                                 const unsigned int block_size)
    {
    assert(d_tables);
    assert(d_angle_exponents);
    assert(table_width > 1);

    if (N == 0)
        {
        return hipSuccess;
        }

    unsigned int max_block_size = 0;
    hipError_t err = gpu_cbt_table_dihedral_get_max_block_size(max_block_size);
    if (err != hipSuccess)
        {
        return err;
        }

    const unsigned int run_block_size = std::min(block_size, max_block_size);
    if (run_block_size == 0)
        {
        return hipErrorInvalidConfiguration;
        }

    dim3 grid((N + run_block_size - 1) / run_block_size, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_compute_cbt_table_dihedral_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       box,
                       tlist,
                       dihedral_ABCD,
                       pitch,
                       n_dihedrals_list,
                       d_tables,
                       table_width,
                       table_value,
                       d_angle_exponents);

    return hipGetLastError();
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
