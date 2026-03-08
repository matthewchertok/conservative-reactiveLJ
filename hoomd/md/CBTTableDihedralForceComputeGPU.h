// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CBTTableDihedralForceCompute.h"
#include "CBTTableDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file CBTTableDihedralForceComputeGPU.h
    \brief Declares the CBTTableDihedralForceComputeGPU class.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CBTTABLEDIHEDRALFORCECOMPUTEGPU_H__
#define __CBTTABLEDIHEDRALFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
class PYBIND11_EXPORT CBTTableDihedralForceComputeGPU : public CBTTableDihedralForceCompute
    {
    public:
    CBTTableDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                    unsigned int table_width);

    virtual ~CBTTableDihedralForceComputeGPU() { }

    private:
    std::shared_ptr<Autotuner<1>> m_tuner;

    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
