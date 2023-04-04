#pragma once

#include <cmath>
#include <string>

using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double>>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecS = ROOT::VecOps::RVec<size_t>;
using RVecUC = ROOT::VecOps::RVec<UChar_t>;
using RVecF = ROOT::VecOps::RVec<float>;
using RVecB = ROOT::VecOps::RVec<bool>;
using RVecVecI = ROOT::VecOps::RVec<RVecI>;
using RVecLV = ROOT::VecOps::RVec<LorentzVectorM>;
using RVecSetInt = ROOT::VecOps::RVec<std::set<int>>;

template<typename T1, typename T2>
auto DeltaPhi(T1 phi1, T2 phi2) -> decltype(phi2 - phi1) { return ROOT::Math::VectorUtil::Phi_mpi_pi(phi2 - phi1); }
template<typename T1, typename T2>
auto DeltaEta(T1 eta1, T2 eta2) -> decltype(eta2 - eta1) { return eta2 - eta1; }
template<typename T1, typename T2, typename T3, typename T4>
auto DeltaR(T1 eta1, T2 phi1, T3 eta2, T4 phi2) -> decltype(eta1 + phi1 + eta2 + phi2)
{
  const auto dphi = DeltaPhi(phi1, phi2);
  const auto deta = DeltaEta(eta1, eta2);
  return std::hypot(dphi, deta);
}

inline RVecS CreateIndexes(size_t vecSize)
{
  RVecS i(vecSize);
  std::iota(i.begin(), i.end(), 0);
  return i;
}

template<typename V, typename Cmp=std::greater<typename V::value_type>>
RVecI ReorderObjects(const V& varToOrder, const RVecI& indices, size_t nMax=std::numeric_limits<size_t>::max(),
                     const Cmp& cmp=Cmp())
{
  RVecI ordered_indices = indices;
  std::sort(ordered_indices.begin(), ordered_indices.end(), [&](int a, int b) {
    return cmp(varToOrder.at(a), varToOrder.at(b));
  });
  const size_t n = std::min(ordered_indices.size(), nMax);
  ordered_indices.resize(n);
  return ordered_indices;
}

template<typename T, int n_binary_places=std::numeric_limits<T>::digits>
std::string GetBinaryString(T x)
{
  std::bitset<n_binary_places> bs(x);
  std::ostringstream ss;
  ss << bs;
  return ss.str();
}

template<typename V1, typename V2, typename V3, typename V4>
RVecLV GetP4(const V1& pt, const V2& eta, const V3& phi, const V4& mass, const RVecS& indices)
{
  const size_t N = pt.size();
  assert(N == eta.size() && N == phi.size() && N == mass.size());
  RVecLV p4;
  p4.reserve(indices.size());
  for(auto idx : indices) {
    assert(idx < N);
    p4.emplace_back(pt[idx], eta[idx], phi[idx], mass[idx]);
  }
  return p4;
}

template<typename V1, typename V2, typename V3, typename V4>
RVecLV GetP4(const V1& pt, const V2& eta, const V3& phi, const V4& mass)
{
  return GetP4(pt, eta, phi, mass, CreateIndexes(pt.size()));
}

inline RVecB RemoveOverlaps(const RVecLV& obj_p4, const RVecB& pre_sel, const std::vector<RVecLV>& other_objects,
                            size_t min_number_of_non_overlaps, double min_deltaR)
{
  RVecB result(pre_sel);
  const double min_deltaR2 = std::pow(min_deltaR, 2);

  const auto hasMinNumberOfNonOverlaps = [&](const LorentzVectorM& p4) {
    size_t cnt = 0;
    for(const auto& other_obj_col : other_objects) {
      for(const auto& other_obj_p4 : other_obj_col) {
        const double dR2 = ROOT::Math::VectorUtil::DeltaR2(p4, other_obj_p4);
        if(dR2 > min_deltaR2) {
          ++cnt;
          if(cnt >= min_number_of_non_overlaps)
            return true;
        }
      }
    }
    return false;
  };

  for(size_t obj_idx = 0; obj_idx < obj_p4.size(); ++obj_idx) {
    result[obj_idx] = pre_sel[obj_idx] && hasMinNumberOfNonOverlaps(obj_p4.at(obj_idx));
  }
  return result;
}

inline std::pair<int, double> FindMatching(const LorentzVectorM& target_p4, const RVecLV& ref_p4, double deltaR_thr)
{
  double deltaR_min = deltaR_thr;
  int current_idx = -1;
  for(int refIdx = 0; refIdx < ref_p4.size(); ++refIdx) {
    const auto dR_targetRef = ROOT::Math::VectorUtil::DeltaR(target_p4, ref_p4.at(refIdx));
    if(dR_targetRef < deltaR_min) {
      deltaR_min = dR_targetRef;
      current_idx = refIdx;
    }
  }
  return std::make_pair(current_idx, deltaR_min);
}

inline RVecI FindBestMatching(const RVecLV& target_p4, const RVecLV& ref_p4, double deltaR_thr)
{
  RVecI targetIndices(target_p4.size());
  for(size_t targetIdx = 0; targetIdx < target_p4.size(); ++targetIdx) {
    const auto match = FindMatching(target_p4[targetIdx], ref_p4, deltaR_thr);
    targetIndices[targetIdx] = match.first;
  }
  return targetIndices;
}

inline RVecI FindUniqueMatching(const RVecLV& target_p4, const RVecLV& ref_p4, double deltaR_thr)
{
  RVecI targetIndices(target_p4.size(), -1);
  const auto default_matching = std::make_pair(-1, deltaR_thr);
  std::vector<std::pair<int, double>> matchings(ref_p4.size(), default_matching);
  auto refIndices = CreateIndexes(ref_p4.size());
  std::sort(refIndices.begin(), refIndices.end(), [&](int a, int b) {
    if(ref_p4[a].pt() != ref_p4[b].pt()) return ref_p4[a].pt() > ref_p4[b].pt();
    if(ref_p4[a].mass() != ref_p4[b].mass()) return ref_p4[a].mass() > ref_p4[b].mass();
    const auto a_eta = std::abs(ref_p4[a].eta());
    const auto b_eta = std::abs(ref_p4[b].eta());
    if(a_eta != b_eta) return a_eta < b_eta;
    const auto a_dir = std::make_pair(ref_p4[a].eta(), ref_p4[a].phi());
    const auto b_dir = std::make_pair(ref_p4[b].eta(), ref_p4[b].phi());
    return a_dir < b_dir;
  });
  for(int refIdx : refIndices) {
    matchings[refIdx] = FindMatching(ref_p4[refIdx], target_p4, deltaR_thr);
    for(int prevIdx : refIndices) {
      if(prevIdx == refIdx) break;
      if(matchings[prevIdx].first == matchings[refIdx].first) {
        matchings[refIdx] = default_matching;
        break;
      }
    }
  }
  for(int refIdx = 0; refIdx < ref_p4.size(); ++refIdx) {
    if(matchings[refIdx].first >= 0)
      targetIndices[matchings[refIdx].first] = refIdx;
  }
  return targetIndices;
}

inline RVecSetInt FindMatchingSet(const RVecLV& target_p4, const RVecLV& ref_p4, double dR_thr,
                                  const RVecB& pre_sel_target, const RVecB& pre_sel_ref)
{
  assert(target_p4.size() == pre_sel_target.size());
  assert(ref_p4.size() == pre_sel_ref.size());
  RVecSetInt findMatching(pre_sel_target.size());
  for(size_t ref_idx = 0; ref_idx < pre_sel_ref.size(); ++ref_idx) {
    if(!pre_sel_ref[ref_idx]) continue;
    for(size_t target_idx = 0; target_idx < pre_sel_target.size(); ++target_idx) {
      if(!pre_sel_target[target_idx]) continue;
      const auto dR = ROOT::Math::VectorUtil::DeltaR(target_p4[target_idx], ref_p4[ref_idx]);
      if(dR < dR_thr)
        findMatching[target_idx].insert(ref_idx);
    }
  }
  return findMatching;
}

inline RVecSetInt FindMatchingSet(const RVecLV& target_p4, const RVecLV& ref_p4, double dR_thr)
{
  return FindMatchingSet(target_p4, ref_p4, dR_thr, RVecB(target_p4.size(), true), RVecB(ref_p4.size(), true));
}

namespace v_ops{
  template<typename LV>
  RVecF pt(const LV& p4) { return ROOT::VecOps::Map(p4, [](const auto& p4) -> float { return p4.pt(); }); }
  template<typename LV>
  RVecF eta(const LV& p4) { return ROOT::VecOps::Map(p4, [](const auto& p4) -> float { return p4.eta(); }); }
  template<typename LV>
  RVecF phi(const LV& p4) { return ROOT::VecOps::Map(p4, [](const auto& p4) -> float { return p4.phi(); }); }
  template<typename LV>
  RVecF mass(const LV& p4) { return ROOT::VecOps::Map(p4, [](const auto& p4) -> float { return p4.mass(); }); }
}