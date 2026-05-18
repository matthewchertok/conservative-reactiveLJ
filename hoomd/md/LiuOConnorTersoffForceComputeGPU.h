// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_GPU_H__
#define __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_GPU_H__

#ifdef ENABLE_HIP

#include "LiuOConnorTersoffForceCompute.h"
#include "LiuOConnorTersoffForceComputeGPU.cuh"

#include "hoomd/Autotuner.h"

#include <memory>

namespace hoomd
    {
namespace md
    {

//! Optimized Liu-O'Connor isotropic Tersoff force compute on the GPU.
class PYBIND11_EXPORT LiuOConnorTersoffForceComputeGPU
    : public LiuOConnorTersoffForceCompute
    {
    public:
    LiuOConnorTersoffForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<NeighborList> nlist,
                                     const std::string& sticky_type,
                                     Scalar A1,
                                     Scalar A2,
                                     Scalar lambda1,
                                     Scalar lambda2,
                                     Scalar dimer_r,
                                     Scalar cutoff_thickness,
                                     Scalar r_cut,
                                     Scalar alpha,
                                     Scalar n,
                                     Scalar gamma);

    ~LiuOConnorTersoffForceComputeGPU() override = default;

    protected:
    void computeForces(uint64_t timestep) override;

    private:
    std::shared_ptr<Autotuner<2>> m_tuner_coord;
    std::shared_ptr<Autotuner<2>> m_tuner_lambda;
    std::shared_ptr<Autotuner<2>> m_tuner_force;
    };

namespace detail
    {
void export_LiuOConnorTersoffForceComputeGPU(pybind11::module& m);
    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP

#endif // __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_GPU_H__
