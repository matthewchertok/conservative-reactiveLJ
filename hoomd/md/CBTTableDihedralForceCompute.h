// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Index1D.h"

#include <memory>

/*! \file CBTTableDihedralForceCompute.h
    \brief Declares a class for computing combined bending-torsion (CBT)
           dihedrals with a tabulated torsional profile.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CBTTABLEDIHEDRALFORCECOMPUTE_H__
#define __CBTTABLEDIHEDRALFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
class PYBIND11_EXPORT CBTTableDihedralForceCompute : public ForceCompute
    {
    public:
    CBTTableDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 unsigned int table_width);

    virtual ~CBTTableDihedralForceCompute();

    virtual void setParams(unsigned int type,
                           const std::vector<Scalar>& U,
                           const std::vector<Scalar>& tau,
                           Scalar angle_exponent);

    void setParamsPython(std::string type, pybind11::dict params);

    pybind11::dict getParams(std::string type);

    unsigned int getWidth()
        {
        return m_table_width;
        }

#ifdef ENABLE_MPI
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    std::shared_ptr<DihedralData> m_dihedral_data;
    unsigned int m_table_width;
    GPUArray<Scalar2> m_tables;
    Index2D m_table_value;
    GPUArray<Scalar> m_angle_exponents;

    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
