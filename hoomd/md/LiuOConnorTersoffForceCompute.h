// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_H__
#define __LIU_O_CONNOR_TERSOFF_FORCE_COMPUTE_H__

#include "hoomd/BoxDim.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/NeighborList.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

namespace hoomd
    {
namespace md
    {

//! Optimized Liu-O'Connor isotropic Tersoff force for one sticky particle type.
class PYBIND11_EXPORT LiuOConnorTersoffForceCompute : public ForceCompute
    {
    public:
    LiuOConnorTersoffForceCompute(std::shared_ptr<SystemDefinition> sysdef,
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

    ~LiuOConnorTersoffForceCompute() override;

    void setParameters(Scalar A1,
                       Scalar A2,
                       Scalar lambda1,
                       Scalar lambda2,
                       Scalar dimer_r,
                       Scalar cutoff_thickness,
                       Scalar r_cut,
                       Scalar alpha,
                       Scalar n,
                       Scalar gamma);

    pybind11::dict getParameters() const;

    std::shared_ptr<NeighborList> getNeighborList() const;

    const std::string& getStickyType() const;

    /// Start autotuning this force and its neighbor list.
    void startAutotuning() override
        {
        ForceCompute::startAutotuning();
        m_nlist->startAutotuning();
        }

    /// Check if autotuning is complete for this force and its neighbor list.
    bool isAutotuningComplete() override
        {
        bool result = ForceCompute::isAutotuningComplete();
        return result && m_nlist->isAutotuningComplete();
        }

    protected:
    void computeForces(uint64_t timestep) override;
    void ensureScratchSize();

    std::shared_ptr<NeighborList> m_nlist;
    Index2D m_typpair_idx;
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_matrix;
    GPUArray<Scalar> m_coordination;
    GPUArray<Scalar> m_lambda;
    bool m_matrix_registered;

    std::string m_sticky_type_name;
    unsigned int m_sticky_typeid;

    Scalar m_A1;
    Scalar m_A2;
    Scalar m_lambda1;
    Scalar m_lambda2;
    Scalar m_dimer_r;
    Scalar m_cutoff_thickness;
    Scalar m_r_cut;
    Scalar m_r_cut_sq;
    Scalar m_r_inner;
    Scalar m_inv_cutoff_thickness;
    Scalar m_alpha_input;
    Scalar m_cutoff_alpha;
    Scalar m_n;
    Scalar m_gamma;
    Scalar m_gamma_n;
    Scalar m_b_exponent;

    private:
    void setStickyType(const std::string& sticky_type);
    void updateNeighborListCutoffs();
    inline void computeCutoff(Scalar distance, Scalar& fc, Scalar& dfc) const;
    inline Scalar computeBondOrder(Scalar zeta) const;
    inline Scalar computeBondOrderDerivative(Scalar zeta) const;
    inline bool computeDirectedTerms(Scalar distance,
                                     Scalar fc,
                                     Scalar& fR,
                                     Scalar& fA,
                                     Scalar& dfR,
                                     Scalar& dfA) const;
    };

namespace detail
    {
void export_LiuOConnorTersoffForceCompute(pybind11::module& m);
    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif
