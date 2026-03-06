// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CBTDihedralForceCompute.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file CBTDihedralForceCompute.cc
    \brief Contains code for the CBTDihedralForceCompute class.
*/

namespace hoomd
    {
namespace md
    {
namespace
    {
inline Scalar3 cross3(const Scalar3& a, const Scalar3& b)
    {
    return make_scalar3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

struct angle_terms
    {
    bool valid;
    Scalar sin_theta;
    Scalar cos_theta;
    Scalar3 dtheta_du;
    Scalar3 dtheta_dv;
    };

inline angle_terms compute_angle_terms(const Scalar3& u, const Scalar3& v, Scalar eps)
    {
    angle_terms terms;
    terms.valid = false;
    terms.sin_theta = Scalar(0.0);
    terms.cos_theta = Scalar(1.0);
    terms.dtheta_du = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    terms.dtheta_dv = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    const Scalar u_norm_sq = dot(u, u);
    const Scalar v_norm_sq = dot(v, v);
    const Scalar u_norm = std::sqrt(u_norm_sq);
    const Scalar v_norm = std::sqrt(v_norm_sq);

    if (u_norm <= eps || v_norm <= eps)
        {
        return terms;
        }

    const Scalar inv_uv = Scalar(1.0) / (u_norm * v_norm);
    Scalar cos_theta = dot(u, v) * inv_uv;
    if (cos_theta > Scalar(1.0))
        {
        cos_theta = Scalar(1.0);
        }
    else if (cos_theta < Scalar(-1.0))
        {
        cos_theta = Scalar(-1.0);
        }

    const Scalar3 uxv = cross3(u, v);
    Scalar sin_theta = std::sqrt(dot(uxv, uxv)) * inv_uv;
    if (sin_theta < Scalar(0.0))
        {
        sin_theta = Scalar(0.0);
        }
    else if (sin_theta > Scalar(1.0))
        {
        sin_theta = Scalar(1.0);
        }

    if (sin_theta <= eps)
        {
        return terms;
        }

    const Scalar3 term_u = v * inv_uv - (cos_theta / u_norm_sq) * u;
    const Scalar3 term_v = u * inv_uv - (cos_theta / v_norm_sq) * v;

    terms.valid = true;
    terms.sin_theta = sin_theta;
    terms.cos_theta = cos_theta;
    terms.dtheta_du = -(Scalar(1.0) / sin_theta) * term_u;
    terms.dtheta_dv = -(Scalar(1.0) / sin_theta) * term_v;
    return terms;
    }

    } // end namespace

/*! \param sysdef System to compute forces on.
    \post Memory is allocated, and forces are zeroed.
*/
CBTDihedralForceCompute::CBTDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing CBTDihedralForceCompute" << endl;

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();

    // check for some silly errors a user could make
    if (m_dihedral_data->getNTypes() == 0)
        {
        throw runtime_error("No dihedral types in the system.");
        }

    // allocate the parameters
    GPUArray<dihedral_cbt_params> params(m_dihedral_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

CBTDihedralForceCompute::~CBTDihedralForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CBTDihedralForceCompute" << endl;
    }

/*! \param type Type of the dihedral to set parameters for.
    \param k1 Force parameter in OPLS-style dihedral.
    \param k2 Force parameter in OPLS-style dihedral.
    \param k3 Force parameter in OPLS-style dihedral.
    \param k4 Force parameter in OPLS-style dihedral.
    \param angle_exponent Exponent in [sin(theta1) sin(theta2)]^m.
*/
void CBTDihedralForceCompute::setParams(unsigned int type,
                                        Scalar k1,
                                        Scalar k2,
                                        Scalar k3,
                                        Scalar k4,
                                        Scalar angle_exponent)
    {
    // make sure the type is valid
    if (type >= m_dihedral_data->getNTypes())
        {
        throw runtime_error("Invalid dihedral type.");
        }

    if (angle_exponent < Scalar(0.0))
        {
        throw runtime_error("CBT dihedral angle_exponent must be non-negative.");
        }

    // set parameters in m_params
    ArrayHandle<dihedral_cbt_params> h_params(m_params,
                                              access_location::host,
                                              access_mode::readwrite);
    h_params.data[type].k1 = k1;
    h_params.data[type].k2 = k2;
    h_params.data[type].k3 = k3;
    h_params.data[type].k4 = k4;
    h_params.data[type].angle_exponent = angle_exponent;
    }

void CBTDihedralForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_dihedral_data->getTypeByName(type);
    dihedral_cbt_params _params(params);
    setParams(typ, _params.k1, _params.k2, _params.k3, _params.k4, _params.angle_exponent);
    }

pybind11::dict CBTDihedralForceCompute::getParams(std::string type)
    {
    auto typ = m_dihedral_data->getTypeByName(type);
    // make sure the type is valid
    if (typ >= m_dihedral_data->getNTypes())
        {
        throw runtime_error("Invalid dihedral type.");
        }
    ArrayHandle<dihedral_cbt_params> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[typ].asDict();
    }

/*! Actually perform the force computation.
    \param timestep Current time step.
 */
void CBTDihedralForceCompute::computeForces(uint64_t timestep)
    {
    (void)timestep;

    assert(m_pdata);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // access the force and virial tensor arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // access parameter data
    ArrayHandle<dihedral_cbt_params> h_params(m_params, access_location::host, access_mode::read);

    // zero data for force calculation before computation
    m_force.zeroFill();
    m_virial.zeroFill();

    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    const size_t virial_pitch = m_virial.getPitch();
    const Scalar eps = Scalar(1e-12);

    // get a local copy of the simulation box
    const BoxDim& box = m_pdata->getBox();

    // iterate through each dihedral
    const unsigned int numDihedrals = (unsigned int)m_dihedral_data->getN();
    for (unsigned int n = 0; n < numDihedrals; n++)
        {
        // lookup the tag of each of the particles participating in the dihedral
        const auto& dihedral = m_dihedral_data->getMembersByIndex(n);
        assert(dihedral.tag[0] < m_pdata->getNGlobal());
        assert(dihedral.tag[1] < m_pdata->getNGlobal());
        assert(dihedral.tag[2] < m_pdata->getNGlobal());
        assert(dihedral.tag[3] < m_pdata->getNGlobal());

        // i1 to i4 are local particle indices
        const unsigned int i1 = h_rtag.data[dihedral.tag[0]];
        const unsigned int i2 = h_rtag.data[dihedral.tag[1]];
        const unsigned int i3 = h_rtag.data[dihedral.tag[2]];
        const unsigned int i4 = h_rtag.data[dihedral.tag[3]];

        // throw an error if this dihedral is incomplete
        if (i1 == NOT_LOCAL || i2 == NOT_LOCAL || i3 == NOT_LOCAL || i4 == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "dihedral.cbt: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1] << " "
                << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }

        assert(i1 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i2 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i3 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i4 < m_pdata->getN() + m_pdata->getNGhosts());

        const Scalar3 r1 = make_scalar3(h_pos.data[i1].x, h_pos.data[i1].y, h_pos.data[i1].z);
        const Scalar3 r2 = make_scalar3(h_pos.data[i2].x, h_pos.data[i2].y, h_pos.data[i2].z);
        const Scalar3 r3 = make_scalar3(h_pos.data[i3].x, h_pos.data[i3].y, h_pos.data[i3].z);
        const Scalar3 r4 = make_scalar3(h_pos.data[i4].x, h_pos.data[i4].y, h_pos.data[i4].z);

        const Scalar3 b1 = box.minImage(r2 - r1);
        const Scalar3 b2 = box.minImage(r3 - r2);
        const Scalar3 b3 = box.minImage(r4 - r3);

        const Scalar3 n1 = cross3(b1, b2);
        const Scalar3 n2 = cross3(b2, b3);

        const Scalar n1_sq = dot(n1, n1);
        const Scalar n2_sq = dot(n2, n2);
        const Scalar b2_sq = dot(b2, b2);

        if (n1_sq <= eps || n2_sq <= eps || b2_sq <= eps)
            {
            continue;
            }

        const Scalar b2_norm = std::sqrt(b2_sq);
        const Scalar3 b2_hat = b2 / b2_norm;

        const Scalar x = dot(n1, n2);
        const Scalar y = dot(cross3(n1, n2), b2_hat);
        const Scalar phi = std::atan2(y, x);

        const Scalar3 dphi_dr1 = -(b2_norm / n1_sq) * n1;
        const Scalar3 dphi_dr4 = +(b2_norm / n2_sq) * n2;
        const Scalar3 dphi_db2 = (dot(b1, b2) / b2_sq) * dphi_dr1 - (dot(b3, b2) / b2_sq) * dphi_dr4;
        const Scalar3 dphi_dr2 = -dphi_dr1 - dphi_db2;
        const Scalar3 dphi_dr3 = dphi_db2 - dphi_dr4;

        const angle_terms angle1 = compute_angle_terms(-b1, b2, eps);
        const angle_terms angle2 = compute_angle_terms(-b2, b3, eps);
        if (!angle1.valid || !angle2.valid)
            {
            continue;
            }

        const unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(n);
        const auto& params = h_params.data[dihedral_type];

        const Scalar m = params.angle_exponent;

        const Scalar cos_phi = std::cos(phi);
        const Scalar cos_2phi = std::cos(Scalar(2.0) * phi);
        const Scalar cos_3phi = std::cos(Scalar(3.0) * phi);
        const Scalar cos_4phi = std::cos(Scalar(4.0) * phi);

        const Scalar opls_energy = Scalar(0.5) * params.k1 * (Scalar(1.0) + cos_phi)
                                   + Scalar(0.5) * params.k2 * (Scalar(1.0) - cos_2phi)
                                   + Scalar(0.5) * params.k3 * (Scalar(1.0) + cos_3phi)
                                   + Scalar(0.5) * params.k4 * (Scalar(1.0) - cos_4phi);

        const Scalar d_opls_dphi = -Scalar(0.5) * params.k1 * std::sin(phi)
                                   + params.k2 * std::sin(Scalar(2.0) * phi)
                                   - Scalar(1.5) * params.k3 * std::sin(Scalar(3.0) * phi)
                                   + Scalar(2.0) * params.k4 * std::sin(Scalar(4.0) * phi);

        const Scalar sin1_pow_m = std::pow(angle1.sin_theta, m);
        const Scalar sin2_pow_m = std::pow(angle2.sin_theta, m);

        const Scalar gate = sin1_pow_m * sin2_pow_m;
        const Scalar energy = gate * opls_energy;

        const Scalar dU_dphi = gate * d_opls_dphi;

        Scalar dU_dtheta1 = Scalar(0.0);
        Scalar dU_dtheta2 = Scalar(0.0);
        if (m != Scalar(0.0))
            {
            dU_dtheta1 = m * std::pow(angle1.sin_theta, m - Scalar(1.0)) * sin2_pow_m
                         * angle1.cos_theta * opls_energy;
            dU_dtheta2 = m * sin1_pow_m * std::pow(angle2.sin_theta, m - Scalar(1.0))
                         * angle2.cos_theta * opls_energy;
            }

        const Scalar3 dtheta1_dr1 = angle1.dtheta_du;
        const Scalar3 dtheta1_dr2 = -angle1.dtheta_du - angle1.dtheta_dv;
        const Scalar3 dtheta1_dr3 = angle1.dtheta_dv;

        const Scalar3 dtheta2_dr2 = angle2.dtheta_du;
        const Scalar3 dtheta2_dr3 = -angle2.dtheta_du - angle2.dtheta_dv;
        const Scalar3 dtheta2_dr4 = angle2.dtheta_dv;

        const Scalar3 dU_dr1 = dU_dphi * dphi_dr1 + dU_dtheta1 * dtheta1_dr1;
        const Scalar3 dU_dr2 = dU_dphi * dphi_dr2 + dU_dtheta1 * dtheta1_dr2 + dU_dtheta2 * dtheta2_dr2;
        const Scalar3 dU_dr3 = dU_dphi * dphi_dr3 + dU_dtheta1 * dtheta1_dr3 + dU_dtheta2 * dtheta2_dr3;
        const Scalar3 dU_dr4 = dU_dphi * dphi_dr4 + dU_dtheta2 * dtheta2_dr4;

        const Scalar3 f1 = -dU_dr1;
        const Scalar3 f2 = -dU_dr2;
        const Scalar3 f3 = -dU_dr3;
        const Scalar3 f4 = -dU_dr4;

        const Scalar e_dihedral = Scalar(0.25) * energy;

        // apply force to each of the 4 atoms
        h_force.data[i1].x += f1.x;
        h_force.data[i1].y += f1.y;
        h_force.data[i1].z += f1.z;
        h_force.data[i1].w += e_dihedral;

        h_force.data[i2].x += f2.x;
        h_force.data[i2].y += f2.y;
        h_force.data[i2].z += f2.z;
        h_force.data[i2].w += e_dihedral;

        h_force.data[i3].x += f3.x;
        h_force.data[i3].y += f3.y;
        h_force.data[i3].z += f3.z;
        h_force.data[i3].w += e_dihedral;

        h_force.data[i4].x += f4.x;
        h_force.data[i4].y += f4.y;
        h_force.data[i4].z += f4.z;
        h_force.data[i4].w += e_dihedral;

        // compute 1/4 of the virial, 1/4 for each atom in the dihedral.
        // Use the same decomposition as other dihedral force computes.
        const Scalar3 vb1 = -b1;
        const Scalar3 vb2 = b2;
        const Scalar3 vb3 = b3;

        Scalar dihedral_virial[6];
        dihedral_virial[0]
            = Scalar(0.25) * (vb1.x * f1.x + vb2.x * f3.x + (vb3.x + vb2.x) * f4.x);
        dihedral_virial[1]
            = Scalar(0.25) * (vb1.y * f1.x + vb2.y * f3.x + (vb3.y + vb2.y) * f4.x);
        dihedral_virial[2]
            = Scalar(0.25) * (vb1.z * f1.x + vb2.z * f3.x + (vb3.z + vb2.z) * f4.x);
        dihedral_virial[3]
            = Scalar(0.25) * (vb1.y * f1.y + vb2.y * f3.y + (vb3.y + vb2.y) * f4.y);
        dihedral_virial[4]
            = Scalar(0.25) * (vb1.z * f1.y + vb2.z * f3.y + (vb3.z + vb2.z) * f4.y);
        dihedral_virial[5]
            = Scalar(0.25) * (vb1.z * f1.z + vb2.z * f3.z + (vb3.z + vb2.z) * f4.z);

        for (unsigned int k = 0; k < 6; k++)
            {
            h_virial.data[virial_pitch * k + i1] += dihedral_virial[k];
            h_virial.data[virial_pitch * k + i2] += dihedral_virial[k];
            h_virial.data[virial_pitch * k + i3] += dihedral_virial[k];
            h_virial.data[virial_pitch * k + i4] += dihedral_virial[k];
            }
        }
    }

namespace detail
    {
void export_CBTDihedralForceCompute(pybind11::module& m)
    {
    pybind11::class_<CBTDihedralForceCompute,
                     ForceCompute,
                     std::shared_ptr<CBTDihedralForceCompute>>(m, "CBTDihedralForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &CBTDihedralForceCompute::setParamsPython)
        .def("getParams", &CBTDihedralForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
