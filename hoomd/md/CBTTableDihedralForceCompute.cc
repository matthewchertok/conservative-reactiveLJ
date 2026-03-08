// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CBTTableDihedralForceCompute.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <pybind11/numpy.h>

using namespace std;

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

inline void interpolate_table(const Scalar2* table,
                              const Index2D& table_value,
                              unsigned int table_width,
                              unsigned int type,
                              Scalar phi,
                              Scalar& U,
                              Scalar& tau)
    {
    const Scalar delta_phi = Scalar(2.0 * M_PI) / Scalar(table_width - 1);
    Scalar value_f = (Scalar(M_PI) + phi) / delta_phi;

    unsigned int value_i;
    Scalar f;
    if (value_f <= Scalar(0.0))
        {
        value_i = 0;
        f = Scalar(0.0);
        }
    else if (value_f >= Scalar(table_width - 1))
        {
        value_i = table_width - 2;
        f = Scalar(1.0);
        }
    else
        {
        value_i = static_cast<unsigned int>(value_f);
        f = value_f - Scalar(value_i);
        }

    const Scalar2 UT0 = table[table_value(value_i, type)];
    const Scalar2 UT1 = table[table_value(value_i + 1, type)];
    U = UT0.x + f * (UT1.x - UT0.x);
    tau = UT0.y + f * (UT1.y - UT0.y);
    }

    } // end namespace

CBTTableDihedralForceCompute::CBTTableDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                           unsigned int table_width)
    : ForceCompute(sysdef), m_table_width(table_width)
    {
    m_exec_conf->msg->notice(5) << "Constructing CBTTableDihedralForceCompute" << endl;

    m_dihedral_data = m_sysdef->getDihedralData();

    if (m_dihedral_data->getNTypes() == 0)
        {
        throw runtime_error("No dihedral types in the system.");
        }

    if (table_width < 2)
        {
        throw runtime_error("CBT table width must be at least 2.");
        }

    GPUArray<Scalar2> tables(m_table_width, m_dihedral_data->getNTypes(), m_exec_conf);
    m_tables.swap(tables);

    Index2D table_value((unsigned int)m_tables.getPitch(),
                        (unsigned int)m_dihedral_data->getNTypes());
    m_table_value = table_value;

    GPUArray<Scalar> angle_exponents(m_dihedral_data->getNTypes(), m_exec_conf);
    m_angle_exponents.swap(angle_exponents);
    }

CBTTableDihedralForceCompute::~CBTTableDihedralForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CBTTableDihedralForceCompute" << endl;
    }

void CBTTableDihedralForceCompute::setParams(unsigned int type,
                                             const std::vector<Scalar>& U,
                                             const std::vector<Scalar>& tau,
                                             Scalar angle_exponent)
    {
    if (type >= m_dihedral_data->getNTypes())
        {
        throw runtime_error("Invalid dihedral type.");
        }

    if (angle_exponent < Scalar(0.0))
        {
        throw runtime_error("CBT dihedral angle_exponent must be non-negative.");
        }

    if (U.size() != m_table_width || tau.size() != m_table_width)
        {
        m_exec_conf->msg->error()
            << "dihedral.cbt_table: table provided to setParams is not of the correct size"
            << endl;
        throw runtime_error("Error initializing CBTTableDihedralForceCompute");
        }

    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_angle_exponents(m_angle_exponents,
                                          access_location::host,
                                          access_mode::readwrite);

    for (unsigned int i = 0; i < m_table_width; ++i)
        {
        h_tables.data[m_table_value(i, type)].x = U[i];
        h_tables.data[m_table_value(i, type)].y = tau[i];
        }
    h_angle_exponents.data[type] = angle_exponent;
    }

void CBTTableDihedralForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto type_id = m_dihedral_data->getTypeByName(type);

    const auto U_py = params["U"].cast<pybind11::array_t<Scalar>>().unchecked<1>();
    const auto tau_py = params["tau"].cast<pybind11::array_t<Scalar>>().unchecked<1>();
    const Scalar angle_exponent = params["angle_exponent"].cast<Scalar>();

    std::vector<Scalar> U(U_py.size());
    std::vector<Scalar> tau(tau_py.size());
    std::copy(U_py.data(0), U_py.data(0) + U_py.size(), U.data());
    std::copy(tau_py.data(0), tau_py.data(0) + tau_py.size(), tau.data());

    setParams(type_id, U, tau, angle_exponent);
    }

pybind11::dict CBTTableDihedralForceCompute::getParams(std::string type)
    {
    auto type_id = m_dihedral_data->getTypeByName(type);

    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_angle_exponents(m_angle_exponents,
                                          access_location::host,
                                          access_mode::read);

    pybind11::dict params;
    auto U = pybind11::array_t<Scalar>(m_table_width);
    auto tau = pybind11::array_t<Scalar>(m_table_width);
    auto U_view = U.mutable_unchecked<1>();
    auto tau_view = tau.mutable_unchecked<1>();

    for (unsigned int i = 0; i < m_table_width; ++i)
        {
        const Scalar2 entry = h_tables.data[m_table_value(i, type_id)];
        U_view(i) = entry.x;
        tau_view(i) = entry.y;
        }

    params["U"] = U;
    params["tau"] = tau;
    params["angle_exponent"] = h_angle_exponents.data[type_id];
    return params;
    }

void CBTTableDihedralForceCompute::computeForces(uint64_t timestep)
    {
    (void)timestep;

    assert(m_pdata);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_angle_exponents(m_angle_exponents,
                                          access_location::host,
                                          access_mode::read);

    m_force.zeroFill();
    m_virial.zeroFill();

    const size_t virial_pitch = m_virial.getPitch();
    const Scalar eps = Scalar(1e-12);
    const BoxDim& box = m_pdata->getBox();

    const unsigned int numDihedrals = (unsigned int)m_dihedral_data->getN();
    for (unsigned int n = 0; n < numDihedrals; n++)
        {
        const auto& dihedral = m_dihedral_data->getMembersByIndex(n);

        const unsigned int i1 = h_rtag.data[dihedral.tag[0]];
        const unsigned int i2 = h_rtag.data[dihedral.tag[1]];
        const unsigned int i3 = h_rtag.data[dihedral.tag[2]];
        const unsigned int i4 = h_rtag.data[dihedral.tag[3]];

        if (i1 == NOT_LOCAL || i2 == NOT_LOCAL || i3 == NOT_LOCAL || i4 == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "dihedral.cbt_table: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1]
                << " " << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete."
                << endl
                << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }

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
        const Scalar m = h_angle_exponents.data[dihedral_type];

        Scalar torsion_energy;
        Scalar torsion_tau;
        interpolate_table(h_tables.data,
                          m_table_value,
                          m_table_width,
                          dihedral_type,
                          phi,
                          torsion_energy,
                          torsion_tau);

        const Scalar sin1_pow_m = std::pow(angle1.sin_theta, m);
        const Scalar sin2_pow_m = std::pow(angle2.sin_theta, m);

        const Scalar gate = sin1_pow_m * sin2_pow_m;
        const Scalar energy = gate * torsion_energy;
        const Scalar dU_dphi = gate * (-torsion_tau);

        Scalar dU_dtheta1 = Scalar(0.0);
        Scalar dU_dtheta2 = Scalar(0.0);
        if (m != Scalar(0.0))
            {
            dU_dtheta1 = m * std::pow(angle1.sin_theta, m - Scalar(1.0)) * sin2_pow_m
                         * angle1.cos_theta * torsion_energy;
            dU_dtheta2 = m * sin1_pow_m * std::pow(angle2.sin_theta, m - Scalar(1.0))
                         * angle2.cos_theta * torsion_energy;
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
void export_CBTTableDihedralForceCompute(pybind11::module& m)
    {
    pybind11::class_<CBTTableDihedralForceCompute,
                     ForceCompute,
                     std::shared_ptr<CBTTableDihedralForceCompute>>(m,
                                                                    "CBTTableDihedralForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int>())
        .def_property_readonly("width", &CBTTableDihedralForceCompute::getWidth)
        .def("setParams", &CBTTableDihedralForceCompute::setParamsPython)
        .def("getParams", &CBTTableDihedralForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
