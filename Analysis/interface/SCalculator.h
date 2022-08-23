// -*- C++ -*-
//
//
/**\class SCalculator.h SCalculator.cc
 Description:
*/
// https://github.com/TauPolSoftware/TauDecaysInterface
// Original Author:  Vladimir Cherepanov
//         Created:  Mon Sep 4 13:49:02 CET 2017
//
//

#pragma once

#include <vector>
#include "TLorentzVector.h"
#include "TComplex.h"
#include "TMatrixT.h"
#include "TVectorT.h"
#include "TMatrixTSym.h"
#include <string.h>
#include "PolarimetricA1.h"

using namespace std;

class SCalculator {

 public:
  SCalculator(){}
  SCalculator(string type){
    type_=type;
  }
  ~SCalculator(){}
  void Configure(vector<TLorentzVector> TauAndProd, TLorentzVector Frame, int charge=1){
    for(unsigned int i=0; i<TauAndProd.size(); i++){
      TauAndProd_HRF.push_back(Boost(TauAndProd.at(i), Frame));

    }
    charge_=charge;
  }
  bool isConfigured(){
    if(TauAndProd_LF.size()!=2){ std::cout<<"Error:   SCalculator is not Configured! Check  the size of input vector!  Size =  "<< TauAndProd_LF.size() <<std::endl; return false;} return true;
  }
  std::vector<TLorentzVector> getBoosted(){return TauAndProd_LF;}
  TLorentzVector Boost(TLorentzVector pB, TLorentzVector frame){
    TMatrixT<double> transform(4,4);
    TMatrixT<double> result(4,1);
    TVectorT<double> vec(4);
    TVector3 b;

    if(frame.Vect().Mag()==0){ std::cout<<"RH Boost is not set, perfrom calculation in the Lab Frame   "<<std::endl; return pB;}
    if(frame.E()==0){ std::cout<<" Caution: Please check that you perform boost correctly!  " <<std::endl; return pB;}
    else   b=frame.Vect()*(1/frame.E());


    vec(0)  = pB.E();    vec(1)  = pB.Px();
    vec(2)  = pB.Py();  vec(3)  = pB.Pz();

    double gamma  = 1/sqrt( 1 - b.Mag2());

    transform(0,0)=gamma; transform(0,1) =- gamma*b.X() ;  transform(0,2) =  - gamma*b.Y();  transform(0,3) = - gamma*b.Z();

    transform(1,0)=-gamma*b.X(); transform(1,1) =(1+ (gamma-1)*b.X()*b.X()/b.Mag2()) ;  transform(1,2) = ((gamma-1)*b.X()*b.Y()/b.Mag2());  transform(1,3) = ((gamma-1)*b.X()*b.Z()/b.Mag2());
    transform(2,0)=-gamma*b.Y(); transform(2,1) = ((gamma-1)*b.Y()*b.X()/b.Mag2());  transform(2,2) = (1 + (gamma-1)*b.Y()*b.Y()/b.Mag2());  transform(2,3) =  ((gamma-1)*b.Y()*b.Z()/b.Mag2());
    transform(3,0)=-gamma*b.Z(); transform(3,1) =((gamma-1)*b.Z()*b.X()/b.Mag2()) ;  transform(3,2) = ((gamma-1)*b.Z()*b.Y()/b.Mag2());  transform(3,3) = (1 + (gamma-1)*b.Z()*b.Z()/b.Mag2());
    result=transform*convertToMatrix(vec);
    return TLorentzVector(result(1,0), result(2,0) ,result(3,0), result(0,0));
  }
  TVector3 Rotate(TVector3 LVec, TVector3 Rot){
    TVector3 vec = LVec;
    vec.RotateZ(0.5*TMath::Pi() - Rot.Phi());  // not 0.5, to avoid warnings about 0 pT
    vec.RotateX(Rot.Theta());
    return vec;
  }
  TVector3 pv(){
    TVector3 out(0,0,0);
    if(type_=="pion") out = -TauAndProd_HRF.at(1).Vect();
    if(type_=="rho"){
      TLorentzVector pi  = TauAndProd_HRF.at(1);
      TLorentzVector pi0 = TauAndProd_HRF.at(2);
      TLorentzVector Tau = TauAndProd_HRF.at(0);
      TLorentzVector q= pi  - pi0;
      TLorentzVector P= Tau;
      TLorentzVector N= Tau - pi - pi0;
      out = P.M()*(2*(q*N)*q.Vect() - q.Mag2()*N.Vect()) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)));
    }
    if(type_=="a1"){
      PolarimetricA1  a1pol;
      a1pol.Configure(TauAndProd_HRF,charge_);
      out = -a1pol.PVC().Vect();
    }
    return out;
  }
  void SortPions(std::vector<TLorentzVector >& pionsvec, std::vector<int>& charges){

    int npim(0),npip(0), npin(0);
    int OSMCPionIndex(0);
    int SSMCPion1Index(0);
    int SSMCPion2Index(0);
    int OSCharge(0);
    int SS1Charge(0);
    int SS2Charge(0);

    TLorentzVector os;
    TLorentzVector ss1;
    TLorentzVector ss2;

    int MCNeutralPionIndex(0);
    int MCChargedPionIndex(0);
    int NeutralPionCharge(0);
    int ChargedPionCharge(0);

    TLorentzVector ChargedPion;
    TLorentzVector NeutralPion;
    if(charges.size()==3) //A1
      {
        for(unsigned int i=0; i<charges.size(); i++){
  	if( charges.at(i)== 1) npip++;
  	if( charges.at(i)==-1) npim++;
        }
        if(npip == 1 && npim == 2){
  	int nss=0;
  	for(unsigned int i=0; i<charges.size(); i++){
  	  if(charges.at(i)== 1){
  	    OSCharge=1;
  	    OSMCPionIndex=i;
  	  }
  	  if(charges.at(i)== -1 && nss == 0){
  	    nss++;
  	    SS1Charge=-1;
  	    SSMCPion1Index=i;
  	  }
  	  if(charges.at(i)== -1 && nss == 1){
  	    SS2Charge=-1;
  	    SSMCPion2Index=i;
  	  }
  	}
        }
        if( npip== 2 && npim==1){
  	int nss=0;
  	for(unsigned int i=0; i<charges.size(); i++){
  	  if(charges.at(i) == -1){
  	    OSCharge=-1;
  	    OSMCPionIndex=i;
  	  }
  	  if(charges.at(i) == 1 && nss ==0){
  	    nss++;
  	    SS1Charge=1;
  	    SSMCPion1Index=i;
  	  }
  	  if(charges.at(i) == 1 && nss == 1){
  	    SS2Charge=1;
  	    SSMCPion2Index=i;
  	  }
  	}
        }
        os=pionsvec.at(OSMCPionIndex);
        ss1=pionsvec.at(SSMCPion1Index);
        ss2=pionsvec.at(SSMCPion2Index);

        charges.clear();
        charges.push_back(OSCharge);
        charges.push_back(SS1Charge);
        charges.push_back(SS2Charge);

        pionsvec.clear();
        pionsvec.push_back(os);
        pionsvec.push_back(ss1);
        pionsvec.push_back(ss2);
      }

    if(charges.size()==2) //Rho
      {
        for(unsigned int i=0; i<charges.size(); i++){
  	if( charges.at(i)== 1) npip++;
  	if( charges.at(i)==-1) npim++;
  	if( charges.at(i)==0) npin++;
        }
        if(npip == 1 && npin == 1){
  	for(unsigned int i=0; i<charges.size(); i++){
  	  if(charges.at(i)== 1){
  	    ChargedPionCharge=1;
  	    MCChargedPionIndex=i;
  	  }
  	  if(charges.at(i)== 0){
  	    NeutralPionCharge=0;
  	    MCNeutralPionIndex=i;
  	  }
  	}
        }
        if( npim== 1 && npin==1){
  	for(unsigned int i=0; i<charges.size(); i++){
  	  if(charges.at(i) == -1){
  	    ChargedPionCharge=-1;
  	    MCChargedPionIndex=i;
  	  }
  	  if(charges.at(i) == 0){
  	    NeutralPionCharge=0;
  	    MCNeutralPionIndex=i;
  	  }
  	}
        }

        ChargedPion=pionsvec.at(MCChargedPionIndex);
        NeutralPion=pionsvec.at(MCNeutralPionIndex);

        charges.clear();
        charges.push_back(ChargedPionCharge);
        charges.push_back(NeutralPionCharge);

        pionsvec.clear();
        pionsvec.push_back(ChargedPion);
        pionsvec.push_back(NeutralPion);
      }
  }
  static bool isOk(TVector3 h1, TVector3 h2, TLorentzVector Tau1, TLorentzVector Tau2, std::vector<TLorentzVector> Pions1, std::vector<TLorentzVector> Pions2) {
    
    bool ok = true;
    TLorentzVector zeroLV(0,0,0,0);
    //
    for(auto ipi1 : Pions1) {
      if(ipi1 == zeroLV) ok = false;
    }
    for(auto ipi2 : Pions2) {
      if(ipi2 == zeroLV) ok = false;
    }
    if(std::isnan(h1.Mag()) || std::isnan(h2.Mag()) || Tau1==zeroLV || Tau2==zeroLV || Tau1==Tau2 || Pions1==Pions2) ok = false;
    //
    return ok;
  }
  static float AcopAngle(std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> tauandprod1, std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> tauandprod2){

    TLorentzVector Tau1P4 = std::get<0>(tauandprod1);
    TLorentzVector Tau2P4 = std::get<0>(tauandprod2);
    //
    std::vector<TLorentzVector> Pions1P4 = std::get<1>(tauandprod1);
    std::vector<TLorentzVector> Pions2P4 = std::get<1>(tauandprod2);
    //
    std::vector<int> Pions1Q = std::get<2>(tauandprod1);
    std::vector<int> Pions2Q = std::get<2>(tauandprod2);
    //
    int tau1Q = 0;
    int tau2Q = 0;
    for(auto iq1 : Pions1Q) tau1Q += iq1;
    for(auto iq2 : Pions2Q) tau2Q += iq2;
    //
    std::string tau1decay = "";
    std::string tau2decay = "";
    if(Pions1P4.size() == 1) tau1decay = "pion";
    else if(Pions1P4.size() == 2) tau1decay = "rho";
    else if(Pions1P4.size() == 3) tau1decay = "a1";
    if(Pions2P4.size() == 1) tau2decay = "pion";
    else if(Pions2P4.size() == 2) tau2decay = "rho";
    else if(Pions2P4.size() == 3) tau2decay = "a1";
    //
    SCalculator Scalc1(tau1decay);
    SCalculator Scalc2(tau2decay);
    if(tau1decay!="pion") Scalc1.SortPions(Pions1P4, Pions1Q);
    if(tau2decay!="pion") Scalc2.SortPions(Pions2P4, Pions2Q);
    //
    std::vector<TLorentzVector> tauandprod1vec;
    std::vector<TLorentzVector> tauandprod2vec;
    tauandprod1vec.push_back(Tau1P4);
    for(auto ipi1 : Pions1P4) tauandprod1vec.push_back(ipi1);
    tauandprod2vec.push_back(Tau2P4);
    for(auto ipi2 : Pions2P4) tauandprod2vec.push_back(ipi2);
    //
    TLorentzVector frame = tauandprod1vec[0] + tauandprod2vec[0];
    //
    Scalc1.Configure(tauandprod1vec,frame,tau1Q);
    TVector3 h1=Scalc1.pv();
    Scalc2.Configure(tauandprod2vec,frame,tau2Q);
    TVector3 h2=Scalc2.pv();
    //
    if(!isOk(h1,h2,Tau1P4,Tau2P4,Pions1P4,Pions2P4)) return -99;
    //
    TLorentzVector tau1_HRF = Scalc1.Boost(tauandprod1vec.at(0),frame);
    TLorentzVector tau2_HRF = Scalc2.Boost(tauandprod2vec.at(0),frame);
    //
    double h1Norm=1./h1.Mag();
    double h2Norm=1./h2.Mag();
    h1=h1*h1Norm;
    h2=h2*h2Norm;
    double k1Norm=1./((h1.Cross(tau1_HRF.Vect().Unit())).Mag());
    double k2Norm=1./((h2.Cross(tau2_HRF.Vect().Unit())).Mag());
    TVector3 k1 = (h1.Cross(tau1_HRF.Vect().Unit()))*k1Norm;
    TVector3 k2 = (h2.Cross(tau2_HRF.Vect().Unit()))*k2Norm;
    //
    if(((h1.Cross(h2))*(tau1_HRF.Vect().Unit()))<=0) {
      return TMath::ATan2((k1.Cross(k2)).Mag(),k1*k2);
    }
    else {
      return (2.*TMath::Pi()-TMath::ATan2((k1.Cross(k2)).Mag(),k1*k2));
    }
  }

  static double M(TLorentzVector LV){
    Double_t mm = LV.T()*LV.T()-LV.X()*LV.X()-LV.Y()*LV.Y()-LV.Z()*LV.Z();
    return mm < 0.0 ? -TMath::Sqrt(-mm) : TMath::Sqrt(mm);
  }

 private:
  string type_;
  int charge_;
  TVector3 polvec();
  vector<TLorentzVector> TauAndProd_LF;
  vector<TLorentzVector> TauAndProd_HRF;
  TLorentzVector TauLV;
  bool debug;
  TMatrixT<double> convertToMatrix(TVectorT<double> V){
    TMatrixT<double> M(V.GetNrows(),1);
    for(int i=0; i < M.GetNrows(); i++){
      M(i,0)=V(i);
    } return M;
  }

};
