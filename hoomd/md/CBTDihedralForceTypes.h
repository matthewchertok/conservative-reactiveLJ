// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
struct dihedral_cbt_params
    {
    Scalar k1;
    Scalar k2;
    Scalar k3;
    Scalar k4;
    Scalar angle_exponent;

#ifndef __HIPCC__
    dihedral_cbt_params() : k1(0.), k2(0.), k3(0.), k4(0.), angle_exponent(0.) { }

    dihedral_cbt_params(pybind11::dict v)
        : k1(v["k1"].cast<Scalar>()), k2(v["k2"].cast<Scalar>()), k3(v["k3"].cast<Scalar>()),
          k4(v["k4"].cast<Scalar>()), angle_exponent(v["angle_exponent"].cast<Scalar>())
        {
        if (angle_exponent < Scalar(0.0))
            {
            throw std::runtime_error("CBT dihedral angle_exponent must be non-negative.");
            }
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k1"] = k1;
        v["k2"] = k2;
        v["k3"] = k3;
        v["k4"] = k4;
        v["angle_exponent"] = angle_exponent;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

    } // namespace md
    } // namespace hoomd
