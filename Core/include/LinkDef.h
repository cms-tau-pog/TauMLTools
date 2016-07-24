/*! Definition of pre-processor directives to generate ROOT dictionaries.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#ifdef __MAKECINT__
#pragma link C++ class std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double>>>;
#pragma link C++ class std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>>;
#pragma link C++ class std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>>;
#pragma link C++ class std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double>>>;
#endif
