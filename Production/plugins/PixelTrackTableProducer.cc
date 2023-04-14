#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

class PixelTrackTableProducer : public edm::global::EDProducer<> {
public:
  using TrackSoAHost = pixelTrack::TrackSoAHostPhase1;
  using TrackSoAConstView = TrackSoAHost::ConstView;
  using TrackHelpers = TracksUtilities<pixelTopology::Phase1>;

  PixelTrackTableProducer(const edm::ParameterSet& cfg) :
      tracksToken_(consumes(cfg.getParameter<edm::InputTag>("tracks"))),
      verticesToken_(consumes(cfg.getParameter<edm::InputTag>("vertices"))),
      beamSpotToken_(consumes(cfg.getParameter<edm::InputTag>("beamSpot"))),
      bFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      precision_(cfg.getParameter<int>("precision"))
  {
    produces<nanoaod::FlatTable>("PixelTrack");
    produces<nanoaod::FlatTable>("PixelVertex");
  }

private:
  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override
  {
    const auto& tracks = event.get(tracksToken_);
    const auto& vertices = *event.get(verticesToken_);
    const auto& beamSpot = event.get(beamSpotToken_);
    const auto& bField = setup.getData(bFieldToken_);

    const auto [trkGood, vtxGood] = selectGoodTracksAndVertices(tracks.const_view(), vertices);

    FillTracks(event, tracks.const_view(), vertices, beamSpot, bField, trkGood, vtxGood);
    FillVertices(event, vertices, vtxGood);
  }

  void FillTracks(edm::Event& event, const TrackSoAConstView& tracks, const ZVertexSoA& vertices,
                  const reco::BeamSpot& beamSpot, const MagneticField& bField,
                  const std::vector<int>& trkGood, const std::vector<int>& vtxGood) const
  {
    static const std::string name = "PixelTrack";
    const size_t nTrk = trkGood.size();
    std::vector<float> pt(nTrk), eta(nTrk), phi(nTrk), tip(nTrk), zip(nTrk), chi2(nTrk), charge(nTrk),
                       dxy(nTrk), dz(nTrk);
    std::vector<int> quality(nTrk), nLayers(nTrk), nHits(nTrk), vtxIdx(nTrk);

    for(size_t trk_outIdx = 0; trk_outIdx < nTrk; ++trk_outIdx) {
      const int trk_idx = trkGood[trk_outIdx];
      pt[trk_outIdx] = tracks.pt(trk_idx);
      eta[trk_outIdx] = tracks.eta(trk_idx);
      phi[trk_outIdx] = TrackHelpers::phi(tracks, trk_idx);
      tip[trk_outIdx] = TrackHelpers::tip(tracks, trk_idx);
      zip[trk_outIdx] = TrackHelpers::zip(tracks, trk_idx);
      chi2[trk_outIdx] = tracks[trk_idx].chi2();
      charge[trk_outIdx] = TrackHelpers::charge(tracks, trk_idx);
      quality[trk_outIdx] = static_cast<int>(tracks.quality(trk_idx));
      nLayers[trk_outIdx] = tracks.nLayers(trk_idx);
      nHits[trk_outIdx] = TrackHelpers::nHits(tracks, trk_idx);

      const auto [trk_dxy, trk_dz] = computeImpactParameters(tracks, trk_idx, beamSpot, bField);
      dxy[trk_outIdx] = trk_dxy;
      dz[trk_outIdx] = trk_dz;

      const int vtx_idx = vertices.idv[trk_idx];
      auto iter = std::find(vtxGood.begin(), vtxGood.end(), vtx_idx);
      int vtx_outIdx = -1;
      if(iter != vtxGood.end())
        vtx_outIdx = std::distance(vtxGood.begin(), iter);
      vtxIdx[trk_outIdx] = vtx_outIdx;
    }

    auto table = std::make_unique<nanoaod::FlatTable>(nTrk, name, false, false);
    table->addColumn<float>("pt", pt, "pt", precision_);
    table->addColumn<float>("eta", eta, "eta", precision_);
    table->addColumn<float>("phi", phi, "phi", precision_);
    table->addColumn<float>("tip", tip, "tip", precision_);
    table->addColumn<float>("zip", zip, "zip", precision_);
    table->addColumn<float>("chi2", chi2, "chi2", precision_);
    table->addColumn<float>("charge", charge, "charge", precision_);
    table->addColumn<int>("quality", quality,
      "track quality: bad = 0, edup = 1, dup = 2, loose = 3, strict = 4, tight = 5, highPurity = 6");
    table->addColumn<int>("nLayers", nLayers, "number of layers with hits");
    table->addColumn<int>("nHits", nHits, "number of hits");
    table->addColumn<float>("dxy", dxy, "transverse IP wrt the beam spot", precision_);
    table->addColumn<float>("dz", dz, "longitudinal IP wrt the beam spot", precision_);
    table->addColumn<int>("vtxIdx", vtxIdx, "index of the associated vertex (-1 if none)");

    event.put(std::move(table), name);
  }

  void FillVertices(edm::Event& event, const ZVertexSoA& vertices, const std::vector<int>& vtxGood) const
  {
    static const std::string name = "PixelVertex";
    const size_t nVtx = vtxGood.size();

    std::vector<float> zv(nVtx), wv(nVtx), chi2(nVtx), ptv2(nVtx);
    std::vector<int> ndof(nVtx);

    for(size_t vtx_outIdx = 0; vtx_outIdx < nVtx; ++vtx_outIdx) {
      const int vtx_idx = vtxGood[vtx_outIdx];
      zv[vtx_outIdx] = vertices.zv[vtx_idx];
      wv[vtx_outIdx] = vertices.wv[vtx_idx];
      chi2[vtx_outIdx] = vertices.chi2[vtx_idx];
      ptv2[vtx_outIdx] = vertices.ptv2[vtx_idx];
      ndof[vtx_outIdx] = vertices.ndof[vtx_idx];
    }

    auto table = std::make_unique<nanoaod::FlatTable>(nVtx, name, false, false);
    table->addColumn<float>("z", zv, "z-position", precision_);
    table->addColumn<float>("weight", wv, "weight (1/error^2)", precision_);
    table->addColumn<float>("chi2", chi2, "chi2", precision_);
    table->addColumn<float>("ptv2", ptv2, "pt^2", precision_);
    table->addColumn<int>("ndof", ndof, "number of degrees of freedom");

    event.put(std::move(table), name);
  }

  std::pair<std::vector<int>, std::vector<int>> selectGoodTracksAndVertices(const TrackSoAConstView& tracks,
                                                                            const ZVertexSoA& vertices) const
  {
    const uint32_t nv = vertices.nvFinal;
    std::vector<int> trkGood, vtxGood;
    const auto max_tracks = tracks.metadata().size();

    std::vector<size_t> nTrkAssociated(nv, 0);
    for(int32_t trk_idx = 0; trk_idx < max_tracks; ++trk_idx) {
      const auto nHits = TrackHelpers::nHits(tracks, trk_idx);
      if(nHits == 0) break;
      if(nHits > 0 && tracks.quality(trk_idx) >= pixelTrack::Quality::loose) {
        trkGood.push_back(trk_idx);
        const int16_t vtx_idx = vertices.idv[trk_idx];
        if(vtx_idx >= 0 && vtx_idx < static_cast<int16_t>(nv))
          ++nTrkAssociated[vtx_idx];
      }
    }
    for (int j = nv - 1; j >= 0; --j) {
      const uint16_t vtx_idx = vertices.sortInd[j];
      assert(vtx_idx < nv);
      if (nTrkAssociated[vtx_idx] >= 2) {
        vtxGood.push_back(vtx_idx);
      }
    }
    return std::make_pair(std::move(trkGood), std::move(vtxGood));
  }

  std::pair<float, float> computeImpactParameters(const TrackSoAConstView& tracks, int trk_idx,
                                                  const reco::BeamSpot& beamspot, const MagneticField& magfi) const
  {
    riemannFit::Vector5d ipar, opar;
    riemannFit::Matrix5d icov, ocov;
    TrackHelpers::copyToDense(tracks, ipar, icov, trk_idx);
    riemannFit::transformToPerigeePlane(ipar, icov, opar, ocov);
    const LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);
    const float sp = std::sin(TrackHelpers::phi(tracks, trk_idx));
    const float cp = std::cos(TrackHelpers::phi(tracks, trk_idx));
    const Surface::RotationType rotation(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);
    const GlobalPoint beamSpotPoint(beamspot.x0(), beamspot.y0(), beamspot.z0());
    const Plane impPointPlane(beamSpotPoint, rotation);
    const GlobalTrajectoryParameters gp(impPointPlane.toGlobal(lpar.position()),
                                        impPointPlane.toGlobal(lpar.momentum()), lpar.charge(), &magfi);
    const GlobalPoint vv = gp.position();
    const math::XYZPoint pos(vv.x(), vv.y(), vv.z());
    const GlobalVector pp = gp.momentum();
    const math::XYZVector mom(pp.x(), pp.y(), pp.z());
    const auto lambda = M_PI_2 - pp.theta();
    const auto phi = pp.phi();
    const float dxy = -vv.x() * std::sin(phi) + vv.y() * std::cos(phi);
    const float dz =
        (vv.z() * std::cos(lambda) - (vv.x() * std::cos(phi) + vv.y() * std::sin(phi)) * std::sin(lambda)) /
        std::cos(lambda);
    return std::make_pair(dxy, dz);
  }

private:
  const edm::EDGetTokenT<TrackSoAHost> tracksToken_;
  const edm::EDGetTokenT<ZVertexHeterogeneous> verticesToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;
  const unsigned int precision_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelTrackTableProducer);
