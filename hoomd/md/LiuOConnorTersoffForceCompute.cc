// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "LiuOConnorTersoffForceCompute.h"

#include "hoomd/ParticleData.h"

#include <algorithm>
#include <stdexcept>

namespace hoomd
    {
namespace md
    {

LiuOConnorTersoffForceCompute::LiuOConnorTersoffForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
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
    Scalar gamma)
    : ForceCompute(sysdef), m_nlist(std::move(nlist)), m_typpair_idx(m_pdata->getNTypes()),
      m_matrix_registered(false), m_sticky_typeid(0), m_A1(Scalar(0.0)), m_A2(Scalar(0.0)),
      m_lambda1(Scalar(0.0)), m_lambda2(Scalar(0.0)), m_dimer_r(Scalar(1.0)),
      m_cutoff_thickness(Scalar(0.2)), m_r_cut(Scalar(1.5)), m_r_cut_sq(Scalar(2.25)),
      m_r_inner(Scalar(1.3)), m_inv_cutoff_thickness(Scalar(5.0)),
      m_alpha_input(Scalar(-3.0)), m_cutoff_alpha(Scalar(3.0)), m_n(Scalar(1.0)),
      m_gamma(Scalar(1.0)), m_gamma_n(Scalar(1.0)), m_b_exponent(Scalar(-0.5))
    {
    if (!m_nlist)
        {
        throw std::invalid_argument("LiuOConnorTersoffForceCompute requires a valid neighbor list.");
        }

    m_exec_conf->msg->notice(6) << "Constructing LiuOConnorTersoffForceCompute" << std::endl;

    if (m_sysdef->isDomainDecomposed())
        {
        throw std::runtime_error(
            "LiuOConnorTersoffForceCompute currently supports only single-rank simulations.");
        }

    m_nlist->setStorageMode(NeighborList::full);

    GPUArray<Scalar> coordination(m_pdata->getMaxN(), m_exec_conf);
    m_coordination.swap(coordination);

    GPUArray<Scalar> lambda(m_pdata->getMaxN(), m_exec_conf);
    m_lambda.swap(lambda);

    m_r_cut_matrix = std::make_shared<GPUArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
        {
        ArrayHandle<Scalar> h_matrix(*m_r_cut_matrix,
                                     access_location::host,
                                     access_mode::overwrite);
        std::fill_n(h_matrix.data, m_typpair_idx.getNumElements(), Scalar(0.0));
        }
    m_nlist->addRCutMatrix(m_r_cut_matrix);
    m_matrix_registered = true;

    setStickyType(sticky_type);
    setParameters(A1,
                  A2,
                  lambda1,
                  lambda2,
                  dimer_r,
                  cutoff_thickness,
                  r_cut,
                  alpha,
                  n,
                  gamma);
    }

LiuOConnorTersoffForceCompute::~LiuOConnorTersoffForceCompute()
    {
    if (m_matrix_registered && m_nlist)
        {
        m_nlist->removeRCutMatrix(m_r_cut_matrix);
        m_matrix_registered = false;
        }
    }

void LiuOConnorTersoffForceCompute::setParameters(Scalar A1,
                                                  Scalar A2,
                                                  Scalar lambda1,
                                                  Scalar lambda2,
                                                  Scalar dimer_r,
                                                  Scalar cutoff_thickness,
                                                  Scalar r_cut,
                                                  Scalar alpha,
                                                  Scalar n,
                                                  Scalar gamma)
    {
    if (A1 < Scalar(0.0) || A2 < Scalar(0.0))
        throw std::invalid_argument("magnitudes must be non-negative");
    if (r_cut <= Scalar(0.0))
        throw std::invalid_argument("r_cut must be positive");
    if (cutoff_thickness <= Scalar(0.0) || cutoff_thickness >= r_cut)
        throw std::invalid_argument("cutoff_thickness must be positive and less than r_cut");
    if (n <= Scalar(0.0))
        throw std::invalid_argument("n must be positive");
    if (gamma < Scalar(0.0))
        throw std::invalid_argument("gamma must be non-negative");

    m_A1 = A1;
    m_A2 = A2;
    m_lambda1 = lambda1;
    m_lambda2 = lambda2;
    m_dimer_r = dimer_r;
    m_cutoff_thickness = cutoff_thickness;
    m_r_cut = r_cut;
    m_r_cut_sq = r_cut * r_cut;
    m_r_inner = r_cut - cutoff_thickness;
    m_inv_cutoff_thickness = Scalar(1.0) / cutoff_thickness;
    m_alpha_input = alpha;
    // Match EvaluatorTersoff's Python parameter convention. The fitted
    // Liu-O'Connor comparison rows use alpha=-3, which gives a positive
    // smoothing coefficient in the exponential cutoff.
    m_cutoff_alpha = -alpha;
    m_n = n;
    m_gamma = gamma;
    m_gamma_n = slow::pow(gamma, n);
    m_b_exponent = Scalar(-0.5) / n;

    updateNeighborListCutoffs();
    }

pybind11::dict LiuOConnorTersoffForceCompute::getParameters() const
    {
    pybind11::dict params;
    params["sticky_type"] = m_sticky_type_name;
    params["magnitudes"] = pybind11::make_tuple(m_A1, m_A2);
    params["exp_factors"] = pybind11::make_tuple(m_lambda1, m_lambda2);
    params["dimer_r"] = m_dimer_r;
    params["cutoff_thickness"] = m_cutoff_thickness;
    params["r_cut"] = m_r_cut;
    params["alpha"] = m_alpha_input;
    params["n"] = m_n;
    params["gamma"] = m_gamma;
    return params;
    }

std::shared_ptr<NeighborList> LiuOConnorTersoffForceCompute::getNeighborList() const
    {
    return m_nlist;
    }

const std::string& LiuOConnorTersoffForceCompute::getStickyType() const
    {
    return m_sticky_type_name;
    }

void LiuOConnorTersoffForceCompute::setStickyType(const std::string& sticky_type)
    {
    m_sticky_type_name = sticky_type;
    m_sticky_typeid = m_pdata->getTypeByName(sticky_type);
    }

void LiuOConnorTersoffForceCompute::updateNeighborListCutoffs()
    {
    ArrayHandle<Scalar> h_matrix(*m_r_cut_matrix, access_location::host, access_mode::overwrite);
    std::fill_n(h_matrix.data, m_typpair_idx.getNumElements(), Scalar(0.0));
    h_matrix.data[m_typpair_idx(m_sticky_typeid, m_sticky_typeid)] = m_r_cut;
    m_nlist->notifyRCutMatrixChange();
    }

void LiuOConnorTersoffForceCompute::ensureScratchSize()
    {
    const unsigned int max_n = m_pdata->getMaxN();
    if (m_coordination.getNumElements() < max_n)
        {
        GPUArray<Scalar> coordination(max_n, m_exec_conf);
        m_coordination.swap(coordination);
        }
    if (m_lambda.getNumElements() < max_n)
        {
        GPUArray<Scalar> lambda(max_n, m_exec_conf);
        m_lambda.swap(lambda);
        }
    }

inline void LiuOConnorTersoffForceCompute::computeCutoff(Scalar distance,
                                                         Scalar& fc,
                                                         Scalar& dfc) const
    {
    if (distance >= m_r_cut)
        {
        fc = Scalar(0.0);
        dfc = Scalar(0.0);
        return;
        }
    if (distance <= m_r_inner)
        {
        fc = Scalar(1.0);
        dfc = Scalar(0.0);
        return;
        }

    const Scalar x = (distance - m_r_inner) * m_inv_cutoff_thickness;
    const Scalar x2 = x * x;
    const Scalar x3 = x2 * x;
    const Scalar inv_denom = Scalar(1.0) / (x3 - Scalar(1.0));

    fc = fast::exp(m_cutoff_alpha * x3 * inv_denom);
    dfc = Scalar(-3.0) * m_cutoff_alpha * x2 * inv_denom * inv_denom
          * m_inv_cutoff_thickness * fc;
    }

inline Scalar LiuOConnorTersoffForceCompute::computeBondOrder(Scalar zeta) const
    {
    if (m_gamma_n == Scalar(0.0) || zeta <= Scalar(0.0))
        {
        return Scalar(1.0);
        }

    const Scalar zetan = slow::pow(zeta, m_n);
    return slow::pow(Scalar(1.0) + m_gamma_n * zetan, m_b_exponent);
    }

inline Scalar LiuOConnorTersoffForceCompute::computeBondOrderDerivative(Scalar zeta) const
    {
    if (m_gamma_n == Scalar(0.0) || zeta <= Scalar(0.0))
        {
        if (zeta == Scalar(0.0) && m_n == Scalar(1.0))
            {
            return Scalar(-0.5) * m_gamma_n;
            }
        return Scalar(0.0);
        }

    const Scalar zeta_n = slow::pow(zeta, m_n);
    const Scalar sum = Scalar(1.0) + m_gamma_n * zeta_n;
    return Scalar(-0.5) * m_gamma_n * slow::pow(zeta, m_n - Scalar(1.0))
           * slow::pow(sum, m_b_exponent - Scalar(1.0));
    }

inline bool LiuOConnorTersoffForceCompute::computeDirectedTerms(Scalar distance,
                                                                Scalar fc,
                                                                Scalar& fR,
                                                                Scalar& fA,
                                                                Scalar& dfR,
                                                                Scalar& dfA) const
    {
    if (fc == Scalar(0.0) || (m_A1 == Scalar(0.0) && m_A2 == Scalar(0.0)))
        {
        fR = fA = dfR = dfA = Scalar(0.0);
        return false;
        }

    fR = m_A1 * fast::exp(m_lambda1 * (m_dimer_r - distance));
    fA = m_A2 * fast::exp(m_lambda2 * (m_dimer_r - distance));
    dfR = -m_lambda1 * fR;
    dfA = -m_lambda2 * fA;
    return true;
    }

void LiuOConnorTersoffForceCompute::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);
    ensureScratchSize();

    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error()
            << "LiuOConnorTersoffForceCompute cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in LiuOConnorTersoffForceCompute");
        }

    const unsigned int N = m_pdata->getN();
    const BoxDim box = m_pdata->getBox();
    const bool compute_virial = m_pdata->getFlags()[pdata_flag::pressure_tensor];

    m_force.zeroFill();
    m_virial.zeroFill();
    m_coordination.zeroFill();
    m_lambda.zeroFill();

    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_coord(m_coordination, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_lambda(m_lambda, access_location::host, access_mode::readwrite);

    // Pass 1: zeta_i = sum_k f_C(r_ik) for sticky centers.
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar4 pos_i = h_pos.data[i];
        const unsigned int type_i = __scalar_as_int(pos_i.w);
        if (type_i != m_sticky_typeid)
            {
            continue;
            }

        const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const size_t head = h_head_list.data[i];
        const unsigned int n_neigh = h_n_neigh.data[i];
        Scalar zeta = Scalar(0.0);

        for (unsigned int k = 0; k < n_neigh; ++k)
            {
            const unsigned int j = h_nlist.data[head + k];
            const Scalar4 pos_j = h_pos.data[j];
            const unsigned int type_j = __scalar_as_int(pos_j.w);
            if (type_j != m_sticky_typeid)
                {
                continue;
                }

            Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
            dx = box.minImage(dx);
            const Scalar rsq = dot(dx, dx);
            if (rsq >= m_r_cut_sq)
                {
                continue;
                }

            const Scalar r = fast::sqrt(rsq);
            Scalar fc = Scalar(0.0);
            Scalar dfc = Scalar(0.0);
            computeCutoff(r, fc, dfc);
            zeta += fc;
            }

        h_coord.data[i] = zeta;
        }

    // Pass 2: lambda_i = sum_j 0.5 f_C(r_ij) f_A(r_ij) db_ij/dzeta.
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar4 pos_i = h_pos.data[i];
        const unsigned int type_i = __scalar_as_int(pos_i.w);
        if (type_i != m_sticky_typeid)
            {
            continue;
            }

        const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const size_t head = h_head_list.data[i];
        const unsigned int n_neigh = h_n_neigh.data[i];
        const Scalar zeta_i = h_coord.data[i];
        Scalar lambda_i = Scalar(0.0);

        for (unsigned int k = 0; k < n_neigh; ++k)
            {
            const unsigned int j = h_nlist.data[head + k];
            const Scalar4 pos_j = h_pos.data[j];
            const unsigned int type_j = __scalar_as_int(pos_j.w);
            if (type_j != m_sticky_typeid)
                {
                continue;
                }

            Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
            dx = box.minImage(dx);
            const Scalar rsq = dot(dx, dx);
            if (rsq >= m_r_cut_sq)
                {
                continue;
                }

            const Scalar r = fast::sqrt(rsq);
            Scalar fc = Scalar(0.0);
            Scalar dfc = Scalar(0.0);
            computeCutoff(r, fc, dfc);

            Scalar fR = Scalar(0.0);
            Scalar fA = Scalar(0.0);
            Scalar dfR = Scalar(0.0);
            Scalar dfA = Scalar(0.0);
            if (!computeDirectedTerms(r, fc, fR, fA, dfR, dfA))
                {
                continue;
                }

            const Scalar zeta_excluding_pair = std::max(Scalar(0.0), zeta_i - fc);
            const Scalar db_dzeta = computeBondOrderDerivative(zeta_excluding_pair);
            lambda_i += Scalar(0.5) * fc * fA * db_dzeta;
            }

        h_lambda.data[i] = lambda_i;
        }

    // Pass 3: directed pair forces plus the simplified bond-order zeta force.
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar4 pos_i = h_pos.data[i];
        const unsigned int type_i = __scalar_as_int(pos_i.w);
        if (type_i != m_sticky_typeid)
            {
            continue;
            }

        const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const size_t head = h_head_list.data[i];
        const unsigned int n_neigh = h_n_neigh.data[i];
        const Scalar zeta_i = h_coord.data[i];
        const Scalar lambda_i = h_lambda.data[i];

        Scalar3 fi = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        Scalar pei = Scalar(0.0);
        Scalar virial_i[6] = {Scalar(0.0), Scalar(0.0), Scalar(0.0),
                              Scalar(0.0), Scalar(0.0), Scalar(0.0)};

        for (unsigned int k = 0; k < n_neigh; ++k)
            {
            const unsigned int j = h_nlist.data[head + k];
            const Scalar4 pos_j = h_pos.data[j];
            const unsigned int type_j = __scalar_as_int(pos_j.w);
            if (type_j != m_sticky_typeid)
                {
                continue;
                }

            Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
            dx = box.minImage(dx);
            const Scalar rsq = dot(dx, dx);
            if (rsq >= m_r_cut_sq)
                {
                continue;
                }

            Scalar r = fast::sqrt(rsq);
            if (r <= Scalar(0.0))
                {
                continue;
                }

            Scalar fc = Scalar(0.0);
            Scalar dfc = Scalar(0.0);
            computeCutoff(r, fc, dfc);

            Scalar fR = Scalar(0.0);
            Scalar fA = Scalar(0.0);
            Scalar dfR = Scalar(0.0);
            Scalar dfA = Scalar(0.0);
            if (!computeDirectedTerms(r, fc, fR, fA, dfR, dfA))
                {
                continue;
                }

            const Scalar zeta_excluding_pair = std::max(Scalar(0.0), zeta_i - fc);
            const Scalar bij = computeBondOrder(zeta_excluding_pair);
            const Scalar db_dzeta = computeBondOrderDerivative(zeta_excluding_pair);
            const Scalar lambda_pair = Scalar(0.5) * fc * fA * db_dzeta;

            const Scalar directed_energy = Scalar(0.5) * fc * (fR - bij * fA);
            const Scalar pair_force_divr
                = Scalar(-0.5) * (dfc * (fR - bij * fA) + fc * (dfR - bij * dfA)) / r;

            const Scalar zeta_force_divr = (lambda_i - lambda_pair) * dfc / r;
            const Scalar total_force_divr = pair_force_divr + zeta_force_divr;

            fi += total_force_divr * dx;
            h_force.data[j].x -= total_force_divr * dx.x;
            h_force.data[j].y -= total_force_divr * dx.y;
            h_force.data[j].z -= total_force_divr * dx.z;

            const Scalar half_energy = Scalar(0.5) * directed_energy;
            pei += half_energy;
            h_force.data[j].w += half_energy;

            if (compute_virial)
                {
                const Scalar force_div2r = Scalar(0.5) * total_force_divr;
                const Scalar vxx = force_div2r * dx.x * dx.x;
                const Scalar vxy = force_div2r * dx.x * dx.y;
                const Scalar vxz = force_div2r * dx.x * dx.z;
                const Scalar vyy = force_div2r * dx.y * dx.y;
                const Scalar vyz = force_div2r * dx.y * dx.z;
                const Scalar vzz = force_div2r * dx.z * dx.z;

                virial_i[0] += vxx;
                virial_i[1] += vxy;
                virial_i[2] += vxz;
                virial_i[3] += vyy;
                virial_i[4] += vyz;
                virial_i[5] += vzz;

                h_virial.data[0 * m_virial_pitch + j] += vxx;
                h_virial.data[1 * m_virial_pitch + j] += vxy;
                h_virial.data[2 * m_virial_pitch + j] += vxz;
                h_virial.data[3 * m_virial_pitch + j] += vyy;
                h_virial.data[4 * m_virial_pitch + j] += vyz;
                h_virial.data[5 * m_virial_pitch + j] += vzz;
                }
            }

        h_force.data[i].x += fi.x;
        h_force.data[i].y += fi.y;
        h_force.data[i].z += fi.z;
        h_force.data[i].w += pei;

        if (compute_virial)
            {
            for (unsigned int row = 0; row < 6; ++row)
                {
                h_virial.data[row * m_virial_pitch + i] += virial_i[row];
                }
            }
        }
    }

namespace detail
    {
void export_LiuOConnorTersoffForceCompute(pybind11::module& m)
    {
    pybind11::class_<LiuOConnorTersoffForceCompute,
                     ForceCompute,
                     std::shared_ptr<LiuOConnorTersoffForceCompute>>(
        m,
        "LiuOConnorTersoffForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            const std::string&,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>(),
             pybind11::arg("sysdef"),
             pybind11::arg("nlist"),
             pybind11::arg("sticky_type"),
             pybind11::arg("A1"),
             pybind11::arg("A2"),
             pybind11::arg("lambda1"),
             pybind11::arg("lambda2"),
             pybind11::arg("dimer_r"),
             pybind11::arg("cutoff_thickness"),
             pybind11::arg("r_cut"),
             pybind11::arg("alpha"),
             pybind11::arg("n"),
             pybind11::arg("gamma"))
        .def("setParameters", &LiuOConnorTersoffForceCompute::setParameters)
        .def("getParameters", &LiuOConnorTersoffForceCompute::getParameters)
        .def("getNeighborList", &LiuOConnorTersoffForceCompute::getNeighborList)
        .def("getStickyType", &LiuOConnorTersoffForceCompute::getStickyType);
    }

    } // namespace detail
    } // namespace md
    } // namespace hoomd
