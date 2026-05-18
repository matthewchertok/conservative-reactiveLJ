// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "LiuOConnorTersoffForceComputeGPU.cuh"

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

DEVICE inline Scalar maxZero(Scalar value)
    {
    return (value > Scalar(0.0)) ? value : Scalar(0.0);
    }

DEVICE inline void computeCutoff(Scalar distance,
                                 const liu_o_connor_tersoff_params& params,
                                 Scalar& fc,
                                 Scalar& dfc)
    {
    if (distance >= params.r_cut)
        {
        fc = Scalar(0.0);
        dfc = Scalar(0.0);
        return;
        }
    if (distance <= params.r_inner)
        {
        fc = Scalar(1.0);
        dfc = Scalar(0.0);
        return;
        }

    const Scalar x = (distance - params.r_inner) * params.inv_cutoff_thickness;
    const Scalar x2 = x * x;
    const Scalar x3 = x2 * x;
    const Scalar inv_denom = Scalar(1.0) / (x3 - Scalar(1.0));

    fc = fast::exp(params.cutoff_alpha * x3 * inv_denom);
    dfc = Scalar(-3.0) * params.cutoff_alpha * x2 * inv_denom * inv_denom
          * params.inv_cutoff_thickness * fc;
    }

DEVICE inline Scalar computeBondOrder(Scalar zeta, const liu_o_connor_tersoff_params& params)
    {
    if (params.gamma_n == Scalar(0.0) || zeta <= Scalar(0.0))
        {
        return Scalar(1.0);
        }

    const Scalar zeta_n = slow::pow(zeta, params.n);
    return slow::pow(Scalar(1.0) + params.gamma_n * zeta_n, params.b_exponent);
    }

DEVICE inline Scalar computeBondOrderDerivative(Scalar zeta,
                                                const liu_o_connor_tersoff_params& params)
    {
    if (params.gamma_n == Scalar(0.0) || zeta <= Scalar(0.0))
        {
        if (zeta == Scalar(0.0) && params.n == Scalar(1.0))
            {
            return Scalar(-0.5) * params.gamma_n;
            }
        return Scalar(0.0);
        }

    const Scalar zeta_n = slow::pow(zeta, params.n);
    const Scalar sum = Scalar(1.0) + params.gamma_n * zeta_n;
    return Scalar(-0.5) * params.gamma_n * slow::pow(zeta, params.n - Scalar(1.0))
           * slow::pow(sum, params.b_exponent - Scalar(1.0));
    }

DEVICE inline bool computeDirectedTerms(Scalar distance,
                                        Scalar fc,
                                        const liu_o_connor_tersoff_params& params,
                                        Scalar& fR,
                                        Scalar& fA,
                                        Scalar& dfR,
                                        Scalar& dfA)
    {
    if (fc == Scalar(0.0) || (params.A1 == Scalar(0.0) && params.A2 == Scalar(0.0)))
        {
        fR = fA = dfR = dfA = Scalar(0.0);
        return false;
        }

    fR = params.A1 * fast::exp(params.lambda1 * (params.dimer_r - distance));
    fA = params.A2 * fast::exp(params.lambda2 * (params.dimer_r - distance));
    dfR = -params.lambda1 * fR;
    dfA = -params.lambda2 * fA;
    return true;
    }

template<int tpp>
__global__ void gpu_liu_o_connor_tersoff_compute_coordination_kernel(
    const unsigned int N,
    const Scalar4* d_pos,
    const unsigned int* d_n_neigh,
    const unsigned int* d_nlist,
    const size_t* d_head_list,
    Scalar* d_coord,
    BoxDim box,
    liu_o_connor_tersoff_params params)
    {
    const unsigned int lane = threadIdx.x % tpp;
    const unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != params.sticky_typeid)
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
    Scalar zeta = Scalar(0.0);

    for (unsigned int k = lane; k < n_neigh; k += tpp)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        if (__scalar_as_int(pos_j.w) != params.sticky_typeid)
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

        Scalar fc = Scalar(0.0);
        Scalar dfc = Scalar(0.0);
        computeCutoff(r, params, fc, dfc);
        zeta += fc;
        }

    zeta = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(zeta);
    if (lane == 0)
        {
        d_coord[idx] = zeta;
        }
    }

template<int tpp>
__global__ void gpu_liu_o_connor_tersoff_compute_lambda_kernel(
    const unsigned int N,
    const Scalar4* d_pos,
    const unsigned int* d_n_neigh,
    const unsigned int* d_nlist,
    const size_t* d_head_list,
    const Scalar* d_coord,
    Scalar* d_lambda,
    BoxDim box,
    liu_o_connor_tersoff_params params)
    {
    const unsigned int lane = threadIdx.x % tpp;
    const unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != params.sticky_typeid)
        {
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];
    const Scalar zeta_i = __ldg(d_coord + idx);
    Scalar lambda_i = Scalar(0.0);

    for (unsigned int k = lane; k < n_neigh; k += tpp)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        if (__scalar_as_int(pos_j.w) != params.sticky_typeid)
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

        Scalar fc = Scalar(0.0);
        Scalar dfc = Scalar(0.0);
        computeCutoff(r, params, fc, dfc);

        Scalar fR = Scalar(0.0);
        Scalar fA = Scalar(0.0);
        Scalar dfR = Scalar(0.0);
        Scalar dfA = Scalar(0.0);
        if (!computeDirectedTerms(r, fc, params, fR, fA, dfR, dfA))
            {
            continue;
            }

        const Scalar zeta_excluding_pair = maxZero(zeta_i - fc);
        const Scalar db_dzeta = computeBondOrderDerivative(zeta_excluding_pair, params);
        lambda_i += Scalar(0.5) * fc * fA * db_dzeta;
        }

    lambda_i = hoomd::detail::WarpReduce<Scalar, tpp>().Sum(lambda_i);
    if (lane == 0)
        {
        d_lambda[idx] = lambda_i;
        }
    }

template<bool compute_virial, int tpp>
__global__ void gpu_liu_o_connor_tersoff_compute_forces_kernel(
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
    liu_o_connor_tersoff_params params)
    {
    const unsigned int lane = threadIdx.x % tpp;
    const unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != params.sticky_typeid)
        {
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];
    const Scalar zeta_i = __ldg(d_coord + idx);
    const Scalar lambda_i = __ldg(d_lambda + idx);

    Scalar3 fi = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar pei = Scalar(0.0);
    Scalar virial_xx = Scalar(0.0);
    Scalar virial_xy = Scalar(0.0);
    Scalar virial_xz = Scalar(0.0);
    Scalar virial_yy = Scalar(0.0);
    Scalar virial_yz = Scalar(0.0);
    Scalar virial_zz = Scalar(0.0);

    for (unsigned int k = lane; k < n_neigh; k += tpp)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        if (__scalar_as_int(pos_j.w) != params.sticky_typeid)
            {
            continue;
            }

        Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);
        if (rsq >= params.r_cut_sq || rsq <= Scalar(0.0))
            {
            continue;
            }

        const Scalar inv_r = fast::rsqrt(rsq);
        const Scalar r = rsq * inv_r;

        Scalar fc = Scalar(0.0);
        Scalar dfc = Scalar(0.0);
        computeCutoff(r, params, fc, dfc);

        Scalar fR = Scalar(0.0);
        Scalar fA = Scalar(0.0);
        Scalar dfR = Scalar(0.0);
        Scalar dfA = Scalar(0.0);
        if (!computeDirectedTerms(r, fc, params, fR, fA, dfR, dfA))
            {
            continue;
            }

        const Scalar zeta_excluding_pair = maxZero(zeta_i - fc);
        const Scalar bij = computeBondOrder(zeta_excluding_pair, params);
        const Scalar db_dzeta = computeBondOrderDerivative(zeta_excluding_pair, params);
        const Scalar lambda_pair = Scalar(0.5) * fc * fA * db_dzeta;

        const Scalar directed_energy = Scalar(0.5) * fc * (fR - bij * fA);
        const Scalar pair_force_divr
            = Scalar(-0.5) * (dfc * (fR - bij * fA) + fc * (dfR - bij * dfA)) * inv_r;
        const Scalar zeta_force_divr = (lambda_i - lambda_pair) * dfc * inv_r;
        const Scalar total_force_divr = pair_force_divr + zeta_force_divr;

        const Scalar3 force = total_force_divr * dx;
        fi += force;
        atomicAddScalar(&d_force[j].x, -force.x);
        atomicAddScalar(&d_force[j].y, -force.y);
        atomicAddScalar(&d_force[j].z, -force.z);

        const Scalar half_energy = Scalar(0.5) * directed_energy;
        pei += half_energy;
        atomicAddScalar(&d_force[j].w, half_energy);

        if constexpr (compute_virial)
            {
            const Scalar force_div2r = Scalar(0.5) * total_force_divr;
            const Scalar vxx = force_div2r * dx.x * dx.x;
            const Scalar vxy = force_div2r * dx.x * dx.y;
            const Scalar vxz = force_div2r * dx.x * dx.z;
            const Scalar vyy = force_div2r * dx.y * dx.y;
            const Scalar vyz = force_div2r * dx.y * dx.z;
            const Scalar vzz = force_div2r * dx.z * dx.z;

            virial_xx += vxx;
            virial_xy += vxy;
            virial_xz += vxz;
            virial_yy += vyy;
            virial_yz += vyz;
            virial_zz += vzz;

            atomicAddScalar(d_virial + 0 * virial_pitch + j, vxx);
            atomicAddScalar(d_virial + 1 * virial_pitch + j, vxy);
            atomicAddScalar(d_virial + 2 * virial_pitch + j, vxz);
            atomicAddScalar(d_virial + 3 * virial_pitch + j, vyy);
            atomicAddScalar(d_virial + 4 * virial_pitch + j, vyz);
            atomicAddScalar(d_virial + 5 * virial_pitch + j, vzz);
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
        atomicAddScalar(&d_force[idx].x, fi.x);
        atomicAddScalar(&d_force[idx].y, fi.y);
        atomicAddScalar(&d_force[idx].z, fi.z);
        atomicAddScalar(&d_force[idx].w, pei);

        if constexpr (compute_virial)
            {
            atomicAddScalar(d_virial + 0 * virial_pitch + idx, virial_xx);
            atomicAddScalar(d_virial + 1 * virial_pitch + idx, virial_xy);
            atomicAddScalar(d_virial + 2 * virial_pitch + idx, virial_xz);
            atomicAddScalar(d_virial + 3 * virial_pitch + idx, virial_yy);
            atomicAddScalar(d_virial + 4 * virial_pitch + idx, virial_yz);
            atomicAddScalar(d_virial + 5 * virial_pitch + idx, virial_zz);
            }
        }
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
    return getCachedMaxThreadsPerBlock(
        (const void*)gpu_liu_o_connor_tersoff_compute_coordination_kernel<tpp>,
        max_threads_cache,
        cache_mutex,
        max_block_size);
    }

template<int tpp> hipError_t getLambdaMaxBlockSize(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock(
        (const void*)gpu_liu_o_connor_tersoff_compute_lambda_kernel<tpp>,
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
        (const void*)gpu_liu_o_connor_tersoff_compute_forces_kernel<compute_virial, tpp>,
        max_threads_cache,
        cache_mutex,
        max_block_size);
    }

template<int tpp> struct CoordinationLauncher
    {
    static hipError_t launch(const unsigned int N,
                             const Scalar4* d_pos,
                             const unsigned int* d_n_neigh,
                             const unsigned int* d_nlist,
                             const size_t* d_head_list,
                             Scalar* d_coord,
                             BoxDim box,
                             liu_o_connor_tersoff_params params,
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
            hipLaunchKernelGGL((gpu_liu_o_connor_tersoff_compute_coordination_kernel<tpp>),
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
                             BoxDim,
                             liu_o_connor_tersoff_params,
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
                             BoxDim box,
                             liu_o_connor_tersoff_params params,
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
            hipLaunchKernelGGL((gpu_liu_o_connor_tersoff_compute_lambda_kernel<tpp>),
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
                             BoxDim,
                             liu_o_connor_tersoff_params,
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
                             BoxDim box,
                             liu_o_connor_tersoff_params params,
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
            hipLaunchKernelGGL((gpu_liu_o_connor_tersoff_compute_forces_kernel<compute_virial,
                                                                                tpp>),
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
                             BoxDim,
                             liu_o_connor_tersoff_params,
                             unsigned int,
                             unsigned int)
        {
        return hipErrorInvalidConfiguration;
        }
    };
    } // namespace

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
    unsigned int threads_per_particle)
    {
    return CoordinationLauncher<gpu_liu_o_connor_tersoff_max_tpp>::launch(
        N,
        d_pos,
        d_n_neigh,
        d_nlist,
        d_head_list,
        d_coord,
        box,
        params,
        block_size,
        threads_per_particle);
    }

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
    unsigned int threads_per_particle)
    {
    return LambdaLauncher<gpu_liu_o_connor_tersoff_max_tpp>::launch(N,
                                                                    d_pos,
                                                                    d_n_neigh,
                                                                    d_nlist,
                                                                    d_head_list,
                                                                    d_coord,
                                                                    d_lambda,
                                                                    box,
                                                                    params,
                                                                    block_size,
                                                                    threads_per_particle);
    }

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
    unsigned int threads_per_particle)
    {
    if (compute_virial != 0)
        {
        return ForceLauncher<true, gpu_liu_o_connor_tersoff_max_tpp>::launch(
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
            box,
            params,
            block_size,
            threads_per_particle);
        }

    return ForceLauncher<false, gpu_liu_o_connor_tersoff_max_tpp>::launch(
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
        box,
        params,
        block_size,
        threads_per_particle);
    }

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
