// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"
#include "CBTDihedralForceTypes.h"

#include <memory>

/*! \file CBTDihedralForceCompute.h
    \brief Declares a class for computing combined bending-torsion (CBT) dihedrals.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CBTDIHEDRALFORCECOMPUTE_H__
#define __CBTDIHEDRALFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Computes combined bending-torsion (CBT) dihedral forces on each particle.
class PYBIND11_EXPORT CBTDihedralForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    CBTDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~CBTDihedralForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type,
                           Scalar k1,
                           Scalar k2,
                           Scalar k3,
                           Scalar k4,
                           Scalar angle_exponent);

    //! Set parameters for a named dihedral type from Python.
    virtual void setParamsPython(std::string type, pybind11::dict params);

    //! Get parameters for a named dihedral type.
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this force.
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    GPUArray<dihedral_cbt_params> m_params;

    //!< Dihedral data to use in computing dihedrals
    std::shared_ptr<DihedralData> m_dihedral_data;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
