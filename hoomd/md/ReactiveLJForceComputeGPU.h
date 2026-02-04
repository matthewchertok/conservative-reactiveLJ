// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __REACTIVE_LJ_FORCE_COMPUTE_GPU_H__
#define __REACTIVE_LJ_FORCE_COMPUTE_GPU_H__

#ifdef ENABLE_HIP

#include "ReactiveLJForceCompute.h"
#include "ReactiveLJForceComputeGPU.cuh"

#include "hoomd/Autotuner.h"

#include <memory>

namespace hoomd
    {
namespace md
    {

//! ReactiveLJ force compute on the GPU.
class PYBIND11_EXPORT ReactiveLJForceComputeGPU : public ReactiveLJForceCompute
    {
    public:
    ReactiveLJForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<NeighborList> nlist,
                              const std::string& reactive_type,
                              Scalar sigma,
                              Scalar epsilon,
                              Scalar r_cut,
                              Scalar weakening_inner,
                              Scalar weakening_outer,
                              Scalar weakening_exponent,
                              bool smooth_elbow = true,
                              Scalar smooth_kappa = Scalar(0.05),
                              Scalar smooth_beta = Scalar(1.0));

    ~ReactiveLJForceComputeGPU() override = default;

    protected:
    void computeForces(uint64_t timestep) override;

    private:
    // We keep three autotuners because the implementation is split into
    // three kernels (coordination, lambda, and force accumulation), and each
    // kernel has a different occupancy / memory profile.
    std::shared_ptr<Autotuner<1>> m_tuner_coord;
    std::shared_ptr<Autotuner<1>> m_tuner_lambda;
    std::shared_ptr<Autotuner<1>> m_tuner_force;
    // Directed per-neighbor contribution gate*dU/dCexc from pass 2.
    // Pass 3 reuses this cache to avoid recomputing expensive terms.
    GPUArray<Scalar> m_pair_contrib;
    // Device-side error code used by smoothing-bound checks.
    GPUArray<unsigned int> m_error_flag;
    };

namespace detail
    {
void export_ReactiveLJForceComputeGPU(pybind11::module& m);
    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP

#endif // __REACTIVE_LJ_FORCE_COMPUTE_GPU_H__
