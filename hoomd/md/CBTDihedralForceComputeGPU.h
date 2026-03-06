// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CBTDihedralForceCompute.h"
#include "CBTDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file CBTDihedralForceComputeGPU.h
    \brief Declares the CBTDihedralForceComputeGPU class.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CBTDIHEDRALFORCECOMPUTEGPU_H__
#define __CBTDIHEDRALFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Computes CBT dihedral potentials on the GPU.
class PYBIND11_EXPORT CBTDihedralForceComputeGPU : public CBTDihedralForceCompute
    {
    public:
    //! Constructs the compute
    CBTDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~CBTDihedralForceComputeGPU() { }

    private:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
