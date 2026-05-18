// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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

#ifndef __REACTIVE_LJ_FORCE_COMPUTE_H__
#define __REACTIVE_LJ_FORCE_COMPUTE_H__

namespace hoomd
    {
namespace md
    {

//! Reactive Lennard-Jones force compute with coordination-based weakening.
class PYBIND11_EXPORT ReactiveLJForceCompute : public ForceCompute
    {
    public:
    ReactiveLJForceCompute(std::shared_ptr<SystemDefinition> sysdef,
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

    ~ReactiveLJForceCompute() override;

    void setParameters(Scalar sigma,
                       Scalar epsilon,
                       Scalar r_cut,
                       Scalar weakening_inner,
                       Scalar weakening_outer,
                       Scalar weakening_exponent,
                       bool smooth_elbow = true,
                       Scalar smooth_kappa = Scalar(0.05),
                       Scalar smooth_beta = Scalar(1.0));

    pybind11::dict getParameters() const;

    std::shared_ptr<NeighborList> getNeighborList() const;

    const std::string& getReactiveType() const;

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

    std::string m_reactive_type_name;
    unsigned int m_reactive_typeid;

    Scalar m_sigma;
    Scalar m_epsilon;
    Scalar m_r_cut;
    Scalar m_r_cut_sq;
    Scalar m_neighbor_rcut;
    Scalar m_neighbor_rcut_sq;
    Scalar m_energy_shift;
    Scalar m_weakening_inner;
    Scalar m_weakening_outer;
    Scalar m_weakening_exponent;
    unsigned int m_weakening_exponent_int;
    Scalar m_inverse_span;
    Scalar m_pi_inverse_span;
    Scalar m_r_elbow;
    bool m_smooth_elbow;
    Scalar m_smooth_kappa;
    Scalar m_smooth_beta;
    bool m_smooth_beta_is_one;
    Scalar m_smooth_kappa_sigma;
    Scalar m_smooth_delta_tol;
    Scalar m_smooth_r_min;
    Scalar m_lj_energy12;
    Scalar m_lj_energy6;
    Scalar m_lj_force12;
    Scalar m_lj_force6;
    Scalar m_lj_force_deriv12;
    Scalar m_lj_force_deriv6;

    private:
    void setReactiveType(const std::string& reactive_type);
    void updateNeighborListCutoffs();
    void validateSmoothingBounds() const;
    Scalar computeWeight(Scalar distance) const;
    void computeWeightAndDerivative(Scalar distance, Scalar& weight, Scalar& dw_dr) const;
    Scalar evaluateEnergy(Scalar distance) const;
    Scalar evaluateForceMagnitude(Scalar distance) const;
    inline void evaluateCexcSmooth(Scalar raw, Scalar& C_exc, Scalar& gate) const;
    inline Scalar evaluateWeakening(Scalar C_exc) const;
    inline Scalar evaluateSmoothDelta(Scalar weakening) const;
    inline Scalar evaluateSmoothDeltaDerivative(Scalar weakening) const;
    inline Scalar evaluateForceDerivative(Scalar distance) const;
    Scalar evaluate_dU_dCexc(Scalar r_in, Scalar C_exc) const;
    Scalar evaluateWeightDerivative(Scalar distance) const;
    };

namespace detail
    {
void export_ReactiveLJForceCompute(pybind11::module& m);
    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif
