// -*- C++ -*-
//
//
/**\class PolarimetricA1.h PolarimetricA1.cc
 Description:
*/
// https://github.com/TauPolSoftware/TauDecaysInterface
// Original Author:  Vladimir Cherepanov
//         Created:  Mon May 1 13:49:02 CET 2017
//
//
#pragma once

#include <vector>
#include "TLorentzVector.h"
#include "TComplex.h"
#include "TMatrixT.h"
#include "TVectorT.h"
#include "TMatrixTSym.h"
#include "TString.h"
#include <string.h>
using namespace std;


class PolarimetricA1 {

 public:
  PolarimetricA1(){};
  PolarimetricA1(vector<TLorentzVector> TauA1andProd){
    if(TauA1andProd.size()!=4){
      std::cout<<" Warning!! Size of a1 input vector != 4 !! "<<std::endl;
    }
    TLorentzVector boost = TauA1andProd.at(0);
    int fakecharge = 1;
    Setup(TauA1andProd,boost,fakecharge);
  }

  PolarimetricA1(vector<TLorentzVector> TauA1andProd, TLorentzVector RefernceFrame){
    if(TauA1andProd.size()!=4){
      std::cout<<" Warning!! Size of a1 input vector != 4 !! "<<std::endl;
    }
    int fakecharge = 1;
    Setup(TauA1andProd,RefernceFrame,fakecharge);
  }

  ~PolarimetricA1(){};

  void Configure(vector<TLorentzVector> TauA1andProd, int taucharge){

    if(TauA1andProd.size()!=4){
      std::cout<<" Warning!! Size of input vector != 4 !! "<<std::endl;
    }
    TLorentzVector boost = TauA1andProd.at(0);
    Setup(TauA1andProd,boost,taucharge);

  }

  void Configure(vector<TLorentzVector> TauA1andProd, TLorentzVector RefernceFrame, int taucharge){
    if(TauA1andProd.size()!=4){
      std::cout<<" a1 helper:  Warning!! Size of input vector != 4!   Size = "<< TauA1andProd.size()<<std::endl;
    }
    Setup(TauA1andProd,RefernceFrame, taucharge);

  }

  bool isConfigured(){
    if(TauA1andProd_RF.size()!=4){ std::cout<<"Error:   PolarimetricA1 is not Configured! Check  the size of input vector!  Size =  "<< TauA1andProd_RF.size() <<std::endl; return false;} return true;
  }

  void Setup(vector<TLorentzVector> TauA1andProd, TLorentzVector ReferenceFrame, int taucharge ){
   mpi  = 0.13957018; // GeV
   mpi0 = 0.1349766;   // GeV
   mtau = 1.776; // GeV
   coscab = 0.975;
   mrho = 0.773; // GeV
   mrhoprime = 1.370; // GeV
   ma1 = 1.251; // GeV
   mpiprime = 1.300; // GeV
   Gamma0rho  =0.145; // GeV
   Gamma0rhoprime = 0.510; // GeV
   Gamma0a1 = 0.599; // GeV
   Gamma0piprime = 0.3; // GeV
   fpi= 0.093; // GeV
   fpiprime = 0.08; //GeV
   gpiprimerhopi = 5.8; //GeV
   grhopipi = 6.08;  //GeV
   beta = -0.145;
   COEF1 =2.0*sqrt(2.)/3.0;
   COEF2 =-2.0*sqrt(2.)/3.0;
   COEF3 = 2.0*sqrt(2.)/3.0; //C AJW 2/98: Add in the D-wave and I=0 3pi substructure:
   SIGN = -taucharge;
   doSystematic = false;
   systType="UP";

   debug  = false;

   TVector3 RotVector = TauA1andProd.at(0).Vect();

   ReferenceFrame.SetVect(Rotate(ReferenceFrame.Vect(),RotVector));
   _tauAlongZLabFrame = TauA1andProd.at(0);
   _tauAlongZLabFrame.SetVect(Rotate(_tauAlongZLabFrame.Vect(),RotVector));

   for(unsigned int i=0; i<TauA1andProd.size(); i++){
     TLorentzVector Rotated = TauA1andProd.at(i);
     Rotated.SetVect(Rotate(Rotated.Vect(),RotVector));
     //     TauA1andProd_RF.push_back(Boost(Rotated,ReferenceFrame));
     TauA1andProd_RF.push_back(TauA1andProd.at(i));
     // TauA1andProd_RF.push_back(Rotated);
   }

   LFosPionLV  = TauA1andProd.at(1);
   LFss1pionLV = TauA1andProd.at(2);
   LFss2pionLV = TauA1andProd.at(3);
   LFa1LV = LFosPionLV+LFss1pionLV+LFss2pionLV;
   LFtauLV = TauA1andProd.at(0);
   LFQ= LFa1LV.M();

   _osPionLV   = TauA1andProd_RF.at(1);
   _ss1pionLV  = TauA1andProd_RF.at(2);
   _ss2pionLV  = TauA1andProd_RF.at(3);
   _a1LV       = _osPionLV+_ss1pionLV+_ss2pionLV;
   _tauLV      = TauA1andProd_RF.at(0);
   _nuLV      = _tauLV - _a1LV;
   _s12 = _ss1pionLV  + _ss2pionLV;
   _s13 = _ss1pionLV  + _osPionLV;
   _s23 = _ss2pionLV  + _osPionLV;
   _s1  =  _s23.M2();
   _s2  =  _s13.M2();
   _s3  =  _s12.M2();
   _Q = _a1LV.M();
   // std::cout << "PolarimetricA1 Tau LV: ";
   // _tauLV.Print();
}

  void subSetup(double s1, double s2, double s3, double Q){
     _s1  =   s1;
     _s2  =   s2;
     _s3  =   s3;
     _Q = Q;
  }

  std::vector<TLorentzVector> getBoosted(){return TauA1andProd_RF;}

  TLorentzVector Boost(TLorentzVector pB, TLorentzVector frame){
     TMatrixT<double> transform(4,4);
     TMatrixT<double> result(4,1);
     TVectorT<double> vec(4);
     TVector3 b;
     if(frame.Vect().Mag()==0){ std::cout<<"PolarimetricA1  Boost is not set, perfrom calculation in the Lab Frame   "<<std::endl; return pB;}
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


  //====================
  double costhetaLF(){
    double QQ = LFQ*LFQ;
    double x = LFa1LV.E()/LFtauLV.E();
    double s = 4*LFtauLV.E()*LFtauLV.E();
    if( 1 - 4*mtau*mtau/s  <= 0 ){if(debug){std::cout<<"Warning! In PolarimetricA1::costheta root square <=0! return 0"<<std::endl;} return 0;}
    return (2*x*mtau*mtau - mtau*mtau - QQ)/( (mtau*mtau - QQ)*sqrt(1 - 4*mtau*mtau/s) );
  }
  double sinthetaLF(){
    if( costhetaLF()*costhetaLF() > 1 ) {if(debug){std::cout<<"Warning! In PolarimetricA1::sin theta root square <=0! return nan;   costheta = "<< costhetaLF()<<std::endl; }}
    return sqrt(1- costhetaLF()*costhetaLF());
  }

  double cosbetaLF(){
    double QQ = LFQ*LFQ;
    double B1 = (pow(LFss1pionLV.E()*LFa1LV.E()   - Scalar(LFss1pionLV,LFa1LV),2 ) - QQ*mpi*mpi)/QQ;
    double B2 = (pow(LFss2pionLV.E()*LFa1LV.E()   - Scalar(LFss2pionLV,LFa1LV),2 ) - QQ*mpi*mpi)/QQ;
    double B3 = (pow(LFosPionLV.E()*LFa1LV.E()   -   Scalar(LFosPionLV,LFa1LV),2 ) - QQ*mpi*mpi)/QQ;

    TVector3 ss1pionVect = LFss1pionLV.Vect();
    TVector3 ss2pionVect = LFss2pionLV.Vect();
    TVector3 ospionVect = LFosPionLV.Vect();
    double T = 0.5*sqrt(-lambda(B1,B2,B3));
    if(T==0 || LFa1LV.P()==0){if(debug){std::cout<<" Warning!  Can not compute cosbetaLF, denominator =0; return 0; "<<std::endl;} return 0;}
    return ospionVect.Dot(ss1pionVect.Cross(ss2pionVect)) /LFa1LV.P()/T;
  }
  double cospsiLF(){
    double QQ = LFQ*LFQ;
    double s = 4*LFtauLV.E()*LFtauLV.E();
    double x = 2*LFa1LV.E()/sqrt(s);
    if(x*x  - 4*QQ/s <= 0 ){if(debug){std::cout<<"Warning! In PolarimetricA1::cospsi root square <=0! return 0"<<std::endl;} return 0;}
    return    ( x*(mtau*mtau + QQ)  - 2*QQ  )   /   ( mtau*mtau  - QQ   ) / sqrt(x*x  - 4*QQ/s);
  }
  double sinpsiLF(){
    if(cospsiLF()*cospsiLF() > 1  ){if(debug){std::cout<<"Warning! In PolarimetricA1::sinpsi root square <=0! return nan"<<std::endl;}}
    return    sqrt(1 - cospsiLF()*cospsiLF());
  }
  double ultrarel_cospsiLF(){
    double QQ = LFQ*LFQ;
    double cos = (costhetaLF()*(mtau*mtau  + QQ)   + (mtau*mtau  - QQ))/(costhetaLF()*(mtau*mtau  - QQ)   + (mtau*mtau  + QQ));
    return cos;
  }
  double cosgammaLF(){
    double QQ=LFQ*LFQ;
    double B3 = (pow(LFosPionLV.E()*LFtauLV.E()   - LFosPionLV.Vect().Dot(LFa1LV.Vect()),2 ) - QQ*mpi*mpi)/QQ;
    double A3=(LFa1LV.E() * LFa1LV.Vect().Dot(LFosPionLV.Vect()) - LFosPionLV.E()*LFa1LV.P()*LFa1LV.P()) / LFQ;

    if(B3<=0 || cosbetaLF() >=1){std::cout<<"Warning! In PolarimetricA1::cosgamma square root <=0! return 0"<<std::endl; return 0;}
    return A3/LFa1LV.P()/sqrt(B3)/sqrt(1 - cosbetaLF()*cosbetaLF());
  }
  double singammaLF(){
    double QQ=LFQ*LFQ;
     double B1 = (pow(LFss1pionLV.E()*LFa1LV.E()   - LFss1pionLV.Vect().Dot(LFa1LV.Vect()),2 ) - QQ*mpi*mpi)/QQ;
    double B2 = (pow(LFss2pionLV.E()*LFa1LV.E()   - LFss2pionLV.Vect().Dot(LFa1LV.Vect()),2 ) - QQ*mpi*mpi)/QQ;
     double B3 = (pow(LFosPionLV.E()*LFa1LV.E()   - LFosPionLV.Vect().Dot(LFa1LV.Vect()),2 ) - QQ*mpi*mpi)/QQ;

    double T = 0.5*sqrt(-lambda(B1,B2,B3));

    double A1=(LFa1LV.E()*LFa1LV.Vect().Dot(LFss1pionLV.Vect()) - LFss1pionLV.E()*LFa1LV.P()*LFa1LV.P())/QQ;

    double A3=(LFa1LV.E()*LFa1LV.Vect().Dot(LFosPionLV.Vect()) - LFosPionLV.E()*LFa1LV.P()*LFa1LV.P())/QQ;

    if(A3==0 || T==0){std::cout<<"Warning! In PolarimetricA1::singamma denominator ==0! return 0"<<std::endl; return 0;}
    double scale = -(B3*A1/A3 - 0.5*(B2 - B1 - B3))/T;

    return cosgammaLF()*scale;
  }
  double cosalpha(){
     TVector3 nLCrossnT  = nL().Cross(nT());
     TVector3 nLCrossnPerp  = nL().Cross(nPerp());

     if(nLCrossnPerp.Mag() ==0 || nLCrossnT.Mag() ==0){if(debug){std::cout<<" Can not compute cos alpha, one denominator is 0, return cos alpha =0  "<< std::endl;} return 0;}
    return nLCrossnT.Dot(nLCrossnPerp)/nLCrossnT.Mag()/nLCrossnPerp.Mag();
  }
  double sinalpha(){
    TVector3 nLCrossnT  = nL().Cross(nT());
    TVector3 nLCrossnPerp  = nL().Cross(nPerp());
    if(nLCrossnPerp.Mag() ==0 || nLCrossnT.Mag() ==0){if(debug){std::cout<<" Can not compute sin alpha, one denominator is 0, return sin alpha =0  "<< std::endl; }return 0;}
    return -nT().Dot(nLCrossnPerp)/nLCrossnT.Mag()/nLCrossnPerp.Mag();
  }
  double cos2gamma(){
     return singamma()*singamma()   -     cosgamma()*cosgamma();
  }
  double sin2gamma(){
    return 2*singamma()*cosgamma();
  }
  double singamma(){
    TVector3 nLCrossnPerp  = nL().Cross(nPerp());
    TVector3 qvect = _osPionLV.Vect()*(1/_osPionLV.Vect().Mag());

    if(nLCrossnPerp.Mag()==0) { if(debug){std::cout<<"Warning! Can not compute cos gamma, denominator =0, return 0  "<< std::endl;} return 0; }
    return qvect*nLCrossnPerp/nLCrossnPerp.Mag();
  }
  double cosgamma(){
    TVector3 nLCrossnPerp  = nL().Cross(nPerp());

    TVector3 qvect = _osPionLV.Vect()*(1/_osPionLV.Vect().Mag());
    //  qvect.Print();
    if(nLCrossnPerp.Mag()==0) { if(debug){std::cout<<"Warning! Can not compute cos gamma, denominator =0, return 0  "<< std::endl; }return 0; }
    return -nL()*qvect/nLCrossnPerp.Mag();
  }
  double cosbeta(){
    return nL().Dot(nPerp());
  }
  double sinbeta(){
    if(cosbeta()*cosbeta() > 1 ){if(debug){std::cout<<"Warning! Can not compute sin beta! return 0"<<std::endl;} return 0;}
    return sqrt(1 - cosbeta()*cosbeta());
  }
  //====================
  double getg(){
    double QQ=_Q*_Q;
    double l0= 0.5*(mtau*mtau - QQ)/sqrt(QQ);
    double line1 =   -2*l0   * costhetaLF()*  ( 2*WA()/3   + 0.5*(3*cospsiLF()*cospsiLF()   -1)  *  ( WA()*(3*cosbeta()*cosbeta() -1 )/6    -  0.5*WC()*sinbeta()*sinbeta()*cos2gamma()   + 0.5*WD()* sinbeta()*sinbeta()* sin2gamma() )   )/sqrt(QQ);
    double line2 = mtau*mtau*WA()*costhetaLF()/QQ  +    sqrt(mtau*mtau/QQ )  * sinthetaLF()* ( 0.5*WA()*2* sinbeta()*cosbeta()* cosalpha() -
                                               WC()*sinbeta()* (sinalpha()* sin2gamma() + cos2gamma()* cosalpha()*    cosbeta() )    -    WD()*sinbeta()*( sinalpha()*cos2gamma() + sin2gamma()* cosalpha()*cosbeta()  )- 2*cospsiLF()*sinpsiLF()  *
  				            (WA()*(3*cosbeta() *cosbeta() -1 )/6   -    0.5*WC()*sinbeta()* sinbeta()* cos2gamma()+ 0.5*WD()*sinbeta()* sinbeta()* cos2gamma() + WD()*sinbeta()* sinbeta()* sin2gamma())/3   );

    double line3  =  sqrt(mtau*mtau/QQ ) *sinthetaLF()* (WE()*(cosbeta()*sinpsiLF() + sinbeta()*cosalpha()) +cosbeta()*sinalpha()*(WSC()*cosgamma() - WSE()*singamma()) + cosalpha()*(WSC()*singamma() + WSE()*cosgamma()));
    double line4  =  -WE()*costhetaLF()*cosbeta()*cospsiLF() + mtau*mtau*costhetaLF()*(WSA() + cospsiLF()*sinbeta()  * (WSB()*cosgamma() - WSD()* singamma()  ) )/QQ;
    double line5  =  sqrt(mtau*mtau/QQ)*sinthetaLF() *  ( sinpsiLF()*sinbeta()*( WSB()* cosgamma() - WSD()* singamma()) + cosbeta()*cosalpha()*(WSD()*singamma() - WSB()*cosgamma()  ) +
  							sinalpha()*(WSD()*cosgamma() + WSB()*singamma())          );
    double res = line1+ line2 + line3 + line4 + line5;
    return res;
  }
  double getf(){
    double QQ=_Q*_Q;
    double l  = 0.5*(mtau*mtau + QQ)/sqrt(QQ);
    double line1 =   -2*l   *   ( 2*WA()/3   + 0.5*(3*cospsiLF()*cospsiLF()   -1)  *  ( WA()*(3*cosbeta()*cosbeta() -1 )/6    - 0.5*WC()*sinbeta()*sinbeta()*cos2gamma()   + 0.5*WD()* sinbeta()*sinbeta()* sin2gamma() )   )/sqrt(QQ);
    double line2 = mtau*mtau*WA()/QQ + mtau*mtau  *  (  WSA() +  cospsiLF()*sinbeta()*(   WSB() *cosgamma()      - WSD() * singamma())     )/QQ + WE()*cosbeta()*cospsiLF();
    double res = line1+ line2;

    return res;
  }

 double vgetg(TString type=""){
   double QQ=_Q*_Q;
   double RR  = mtau*mtau/QQ; double R = sqrt(RR);
   double V = 0.5*(3*cospsiLF()*cospsiLF() - 1)*(1 + RR)*costhetaLF() + 0.5*3*2*cospsiLF()* sinpsiLF()*sinthetaLF()*R;
   double B = 0.5*(3*cosbeta()*cosbeta() - 1);
   double fact =0;
   if(type == "bar") fact =1;
   double gA =  WA()*(costhetaLF()*(RR - 2)   - B*V)/3                                             +     fact*WA()*0.5*R*sinthetaLF()*cosalpha()*2*sinbeta()*cosbeta();
   double gC =  WC()*0.5*V*sinbeta()*sinbeta()* cos2gamma()                                        -     fact*WC()*R*sinthetaLF()*sinbeta()*(sinalpha()*sin2gamma()  -  cos2gamma()*cosalpha()*cosbeta() ) ;
   double gD = -WD()*0.5*V*sinbeta()*sinbeta()* sin2gamma()                                        -     fact*WD()*R*sinthetaLF()*sinbeta()*(sinalpha()*cos2gamma() + sin2gamma()* cosalpha()*cosbeta()  );
   double gE = - WE()*cosbeta()*( costhetaLF()*cospsiLF() + R*sinthetaLF()*sinpsiLF())             +     fact*WE()*R*sinthetaLF()*sinbeta()*cosalpha();


   double res = gA+gC+gD+gE;
   return res;
 }
 double vgetf(TString type=""){
   double QQ=_Q*_Q;
   double RR  = mtau*mtau/QQ;
   double U = 0.5*(3*cospsiLF()*cospsiLF() - 1)*(1 - RR);
   double B = 0.5*(3*cosbeta()*cosbeta() - 1);

   double fA =  WA()*(2+RR + B*U)/3;
   double fC = -WC()*0.5*U*sinbeta()*sinbeta()* cos2gamma();
   double fD = WD()*0.5*U*sinbeta()*sinbeta()* sin2gamma();
   double fE =  WE()*cospsiLF()*cosbeta();

   double res = fA+fC+fD+fE;

   return res;
 }
 double vgetgscalar(TString type=""){
   double QQ=_Q*_Q;
   double RR  = mtau*mtau/QQ; double R = sqrt(RR);
   double V = 0.5*(3*cospsiLF()*cospsiLF() - 1)*(1 + RR)*costhetaLF() + 0.5*3*2*cospsiLF()* sinpsiLF()*sinthetaLF()*R;
   double B = 0.5*(3*cosbeta()*cosbeta() - 1);
   double fact =0;
   if(type == "bar") fact =1;


   double gA =  WA()*(costhetaLF()*(RR - 2)   - B*V)/3                                             +     fact*WA()*0.5*R*sinthetaLF()*cosalpha()*2*sinbeta()*cosbeta();
   double gC =  WC()*0.5*V*sinbeta()*sinbeta()* cos2gamma()                                        -     fact*WC()*R*sinthetaLF()*sinbeta()*(sinalpha()*sin2gamma()  -  cos2gamma()*cosalpha()*cosbeta() ) ;
   double gD = -WD()*0.5*V*sinbeta()*sinbeta()* sin2gamma()                                        -     fact*WD()*R*sinthetaLF()*sinbeta()*(sinalpha()*cos2gamma() + sin2gamma()* cosalpha()*cosbeta()  );
   double gE = - WE()*cosbeta()*( costhetaLF()*cospsiLF() + R*sinthetaLF()*sinpsiLF())             +     fact*WE()*R*sinthetaLF()*sinbeta()*cosalpha();
   double res = gA+gC+gD+gE;

   return res;
 }
 double vgetfscalar(TString type=""){
   double QQ=_Q*_Q;
   double RR  = mtau*mtau/QQ;
   double U = 0.5*(3*cospsiLF()*cospsiLF() - 1)*(1 - RR);
   double B = 0.5*(3*cosbeta()*cosbeta() - 1);

   double fA =  WA()*(2+RR + B*U)/3;
   double fC = -WC()*0.5*U*sinbeta()*sinbeta()* cos2gamma();
   double fD = WD()*0.5*U*sinbeta()*sinbeta()* sin2gamma();
   double fE =  WE()*cospsiLF()*cosbeta();
   double fSA = WSA()*RR;
   double fSB = WSB()*RR*cospsiLF()*sinbeta()*cosgamma();
   double fSC = 0;
   double fSD = -WSD()*RR*cospsiLF()*sinbeta()*singamma();
   double fSE =0;

   double res = fA+fC+fD+fE  + fSA + fSB + fSC  + fSD + fSE;

   return res;
 }

 double vgetA1omega(TString type=""){
   if(vgetf(type)==0){ if(debug){std::cout<<"Warning!  Can not return vomega; f(0)=0; return -5;  "<<std::endl; }return -5;}
   return vgetg(type)/vgetf(type);
 }
 double vgetA1omegascalar(TString type=""){
   if(vgetfscalar(type)==0){ if(debug){std::cout<<"Warning!  Can not return vomegascalar; f(0)=0; return -5;  "<<std::endl;} return -5;}
   return vgetgscalar(type)/vgetfscalar(type);
 }

 double getOmegaA1(){
   double QQ=_Q*_Q;
   double RR  = mtau*mtau/QQ;
   double U = 0.5*(3*cospsiLF()*cospsiLF() - 1)*(1 - RR);
   double V = 0.5*(3*cospsiLF()*cospsiLF() - 1)*(1 + RR)*costhetaLF() + 0.5*3*2*cospsiLF()* sinpsiLF()*sinthetaLF()*sqrt(RR);

   double fa1 = (2  + RR + 0.5*(3*cosbeta()*cosbeta()- 1)*U)*WA()/3 - 0.5*sinbeta()*sinbeta()*cos2gamma()*U*WC() + 0.5*sinbeta()*sinbeta()*sin2gamma()*U*WD() + cospsiLF()*cosbeta()*WE();
   double ga1 = (costhetaLF()*(RR -2) - 0.5*(3*cosbeta()*cosbeta() - 1)*V)*WA()/3 + 0.5*sinbeta()*sinbeta()*cos2gamma()*V*WC() - 0.5*sinbeta()*sinbeta()*sin2gamma()*V*WD() -cosbeta()*(costhetaLF()*cospsiLF() + sinthetaLF()*sinpsiLF()*sqrt(RR))*WE();

   double omega = ga1/fa1;
   if(std::isinf(std::fabs(omega)) || std::isnan(std::fabs(omega))) omega  = -999.;
   return omega;
 }
 double getOmegaA1Bar(){
   return -nTZLFr()*PVC().Vect();
 }

//========== TRF  =======
 double TRF_vgetf(TString type=""){
   double QQ=_Q*_Q;
   double RR  = mtau*mtau/QQ;

   double cb = TRF_cosbeta();     double cg = TRF_cosgamma();
   double sb = TRF_sinbeta();     double sg = TRF_singamma();
   double s2g  = 2*sg*cg; double c2g = cg*cg - sg*sg;
   double Bb = 0.5*(cb*cb + 1);
   double fact=0;
   if(type=="scalar") fact=1;

   double fA =  WA()*(Bb*(1 - RR) + RR);
   double fC = -WC()*0.5*sb*sb*c2g*(1- RR);
   double fD = WD()*0.5*(1-RR)*sb*sb*s2g;
   double fE =  WE()*cb;
   double fSA = WSA()*RR;
   double fSB = WSB()*RR*sb*cg;
   double fSC = 0;
   double fSD = -WSD()*RR*sb*sg;
   double fSE = 0;
   double res = fA+fC+fD+fE+fact*(fSA+fSB+ fSC+fSD+fSE);

   return res;
 }
 double TRF_vgetg(TString type=""){
   double QQ=_Q*_Q;
   double RR  = mtau*mtau/QQ; double R = sqrt(RR);

   double cb = TRF_cosbeta();     double ct = costhetaLF();    double ca = TRF_cosalpha();   double cg = TRF_cosgamma();
   double sb = TRF_sinbeta();     double st = sinthetaLF();    double sa = TRF_sinalpha();   double sg = TRF_singamma();
   double s2g  = 2*sg*cg; double c2g = cg*cg - sg*sg;
   double s2b  = 2*sb*cb;
   double Bb = 0.5*(cb*cb + 1);
   double fact=0;
   if(type=="scalar") fact=1;

   double gA =  WA()*(RR*ct - Bb*ct*(1+RR)) + WA()*0.5*R*st*s2b*ca ;
   double gC =  WC()*(0.5*ct*(1+RR)  *sb*sb*c2g)  - WC()*R*st*sb*( sa*s2g - c2g*ca*cb );
   double gD = -WD()*(0.5*ct*(1+RR)*sb*sb*s2g)  - WD()*R*st*sb*( sa*c2g + s2g*ca*cb  );
   double gE =  -WE()*ct*cb + WE()*R*st*sb*ca ;

   double gSA = WSA()*RR*ct;
   double gSB = WSB()*(RR*ct*sb*cg - R*st*(cb*ca*cg - sa*sg  ));
   double gSC = WSC()*R*st*(cb*sa*cg + ca*sg);
   double gSD = WSD()*(R*st*(cb*ca*sg + sa*cg) - RR*ct*sb*sg);
   double gSE = -WSE()*R*st*(cb*sa*sg - ca*cg);
   double res = gA+gC+gD+gE+  fact*(gSA + gSB + gSC+ gSD + gSE);
   return res;
 }
 double TRF_vgetA1omega(TString type=""){
   if(TRF_vgetf(type)==0){ if(debug){std::cout<<"Warning!  Can not return vomegascalar; f(0)=0; return -50;  "<<std::endl;} return -50;}
   return TRF_vgetg(type)/TRF_vgetf(type);
 }
 double TRF_cosbeta(){
   return nT().Dot(nPerp());
 }
 double  TRF_cosalpha(){
    TVector3 nTCrossns  = nT().Cross(ns());
    TVector3 nTCrossnPerp  = nT().Cross(nPerp());

    if(nTCrossns.Mag() ==0 || nTCrossnPerp.Mag() ==0){if(debug){std::cout<<" Can not compute cos alpha, one denominator is 0, return TRF cos alpha =0  "<< std::endl; }return 0;}
   return nTCrossns.Dot(nTCrossnPerp)/nTCrossns.Mag()/nTCrossnPerp.Mag();
 }
 double  TRF_cosgamma(){
   TVector3 nTCrossnPerp  = nT().Cross(nPerp());

   TVector3 qvect = _osPionLV.Vect()*(1/_osPionLV.Vect().Mag());
   if(nTCrossnPerp.Mag()==0) { if(debug){std::cout<<"Warning! Can not compute TRF cos gamma, denominator =0, return 0  "<< std::endl;} return 0; }
   return -nT()*qvect/nTCrossnPerp.Mag();
 }
 double TRF_sinbeta(){
   if(TRF_cosbeta()*TRF_cosbeta() > 1 ){if(debug){std::cout<<"Warning! Can not compute TRF sin beta! return 0"<<std::endl;} return 0;}
   return sqrt(1 - TRF_cosbeta()*TRF_cosbeta());
 }
 double TRF_sinalpha(){
    TVector3 nTCrossns  = nT().Cross(ns());
    TVector3 nTCrossnPerp  = nT().Cross(nPerp());
    if(nTCrossns.Mag() ==0 || nTCrossnPerp.Mag() ==0){if(debug){std::cout<<" Can not compute sin alpha, one denominator is 0, return TRF sin alpha =0  "<< std::endl; }return 0;}
   return -ns().Dot(nTCrossnPerp)/nTCrossns.Mag()/nTCrossnPerp.Mag();
 }
 double  TRF_singamma(){
   TVector3 nTCrossnPerp  = nT().Cross(nPerp());
   TVector3 qvect = _osPionLV.Vect()*(1/_osPionLV.Vect().Mag());

   if(nTCrossnPerp.Mag()==0) { if(debug){std::cout<<"Warning! Can not compute TRF  sin gamma, denominator =0, return 0  "<< std::endl;} return 0; }
   return qvect*nTCrossnPerp/nTCrossnPerp.Mag();
 }
//========== TRF  =======
  double lambda(double x, double y, double z){
      return x*x +y*y +z*z - 2*x*y - 2*x*z - 2*z*y;
  }
  double Scalar(TLorentzVector p1, TLorentzVector p2){
      return p1.Vect()*p2.Vect();
  }

  double MomentSFunction(double s,string type="WA"){
    int cells(20);
    //  double s = Q*Q;
    double intx(0);
    double m1 = mpi;
    double m2 = mpi;
    double m3 = mpi;

    double m13(0);
    double integral(0);

    double da1(0), db1(0);
    vector<double> set;
    set.push_back(_s1);
    set.push_back(_s2);
    set.push_back(_s3);
    set.push_back(_Q);
    double  stepx  = (pow(sqrt(s)-m2,2) - pow( m1+m3,2) ) / cells;
    for(int i=1;i<cells + 1;i++){
      da1 = pow(m1+m3,2) + stepx*(i-1);
      db1 = pow(m1+m3,2) + stepx*i;
      m13 = 0.5*(da1 + db1);
      double  E3s = (m13 - m1*m1 + m3*m3)/(2*sqrt(m13));
      double  E2s = (s   - m13  -m2*m2)/(2*sqrt(m13));
      double  m23max =pow (E2s+E3s,2) - pow( sqrt(E2s*E2s - m2*m2) - sqrt(E3s*E3s - m3*m3),2);
      double  m23min =  pow(E2s+E3s,2) - pow( sqrt(E2s*E2s - m2*m2) + sqrt(E3s*E3s - m3*m3),2);
      double  stepy = (m23max - m23min)/cells;
      double da2(0), db2(0);
      double inty(0);
      double m23(0);
      double m12(0);
      for(int j=1;j<cells + 1;j++){
        da2 = m23min + stepy*(j-1);
        db2 = m23min + stepy*j;
        m23 = 0.5*(da2 + db2);
        m12 = s +m1*m1 + m2*m2 + m3*m3 - m13 - m23;
        subSetup(m23,m13,m12,sqrt(s));
        double SFunction(0);
        if(type=="WA")SFunction=WA();
        if(type=="WC")SFunction=WC();
        if(type=="WSA")SFunction=WSA();
        if(type=="WSB")SFunction=WSB();
        if(type=="WD"  ){
  	if(m23 > m13)SFunction=WD();
  	else SFunction=-WD();
        }
        if(type=="WE"){
  	if(m23 > m13)SFunction=WE();
  	else SFunction=-WE();
        }
        if(type=="WSD"){
  	if(m23 > m13)SFunction=WSD();
  	else SFunction=-WSD();
        }
        inty+=stepx*stepy*SFunction;
      }
      intx+=inty;
    }
    integral = intx;
    subSetup(set.at(0),set.at(1),set.at(2),set.at(3));

    return integral;
  }
  double K1(double ct, double QQ, int hel){
    if(debug){if(std::fabs(ct) > 1) std::cout<<"Warning! K1: |ct| > 1 "<<std::endl;}
    return   1 - hel*ct - mtau*mtau*(1+hel*ct)/QQ/QQ;
  }
  double K1bar(double ct, double QQ, int hel){
    if(debug){if(std::fabs(ct) > 1) std::cout<<"Warning! K1bar: |ct| > 1 "<<std::endl;}
    double cpsi = (ct*(mtau*mtau  + QQ)   + (mtau*mtau  - QQ))/(ct*(mtau*mtau  - QQ)   + (mtau*mtau  + QQ));
    if(debug){if(std::fabs(cpsi) > 1) std::cout<<"Warning! K1bar: |cpsi| > 1 "<<std::endl;}
    return  K1(ct,QQ,hel)*0.5*(3*cpsi*cpsi - 1) - 3*sqrt(mtau*mtau/QQ/QQ)*cpsi*sqrt(1-cpsi*cpsi)*sqrt(1-ct*ct)*hel;
  }
  double K2(double ct, double QQ, int hel){
    if(debug){if(std::fabs(ct) > 1) std::cout<<"Warning! K1: |ct| > 1 "<<std::endl;}
    return   mtau*mtau*(1+hel*ct)/QQ/QQ;
  }
  double K2bar(double ct, double QQ, int hel){
    if(debug){if(std::fabs(ct) > 1) std::cout<<"Warning! K1bar: |ct| > 1 "<<std::endl;}
    double cpsi = (ct*(mtau*mtau  + QQ)   + (mtau*mtau  - QQ))/(ct*(mtau*mtau  - QQ)   + (mtau*mtau  + QQ));
    if(debug){if(std::fabs(cpsi) > 1) std::cout<<"Warning! K1bar: |cpsi| > 1 "<<std::endl;}
    return  K2(ct,QQ,hel)*cpsi  + sqrt(mtau*mtau/QQ/QQ)*sqrt(1-cpsi*cpsi)*sqrt(1-ct*ct)*hel;

  }
  double K3(double ct, double QQ, int hel){
    if(debug){if(std::fabs(ct) > 1) std::cout<<"Warning! K1: |ct| > 1 "<<std::endl;}
    return   1 - hel*ct;
  }
  double K3bar(double ct, double QQ, int hel){
   if(debug){if(std::fabs(ct) > 1) std::cout<<"Warning! K1bar: |ct| > 1 "<<std::endl;}
   double cpsi = (ct*(mtau*mtau  + QQ)   + (mtau*mtau  - QQ))/(ct*(mtau*mtau  - QQ)   + (mtau*mtau  + QQ));
   if(debug){if(std::fabs(cpsi) > 1) std::cout<<"Warning! K1bar: |cpsi| > 1 "<<std::endl;}
   return  K3(ct,QQ,hel)*cpsi  - sqrt(mtau*mtau/QQ/QQ)*sqrt(1-cpsi*cpsi)*sqrt(1-ct*ct)*hel;
 }
  double getMoment(double ct, string type, int hel){
    int cells(20);
    double qqmin  = 0.4;
    double qqmax = 3.0;
    vector<double> set;
    set.push_back(_s1);
    set.push_back(_s2);
    set.push_back(_s3);
    set.push_back(_Q);
    double  stepqq  = ( qqmax - qqmin) / cells;
    double kern(0);
    double atQQa(0);
    double atQQb(0);
    double atQQ(0);
    double integral(0);
    for(int i=1;i<cells + 1;i++){
      atQQa = qqmin + stepqq*(i-1);
      atQQb = qqmin + stepqq*i;
      atQQ = 0.5*(atQQa + atQQb);
      if(type=="one") kern = (2*K1(ct,atQQ,hel) + 3*K2(ct,atQQ,hel))*MomentSFunction(atQQ,"WA");
      if(type=="beta") kern = 0.2*K1bar(ct,atQQ,hel)*MomentSFunction(atQQ,"WA");
      if(type=="c2g") kern = -0.5*K1bar(ct,atQQ,hel)*MomentSFunction(atQQ,"WC");
      if(type=="s2g") kern = 0.5*K1bar(ct,atQQ,hel)*MomentSFunction(atQQ,"WD");
      if(type=="cb") kern = K3bar(ct,atQQ,hel)*MomentSFunction(atQQ,"WE");
      integral += kern*stepqq;
    }
    //  subSetup(set.at(0),set.at(1),set.at(2),set.at(3));
    return integral;
  }
  //--------------------------- Hadronic current ---------------------
  //  only 9 structure fucbntions are non-zero in 3pions case

  double WA(){
     return  VV1()*F1().Rho2() + VV2()*F2().Rho2()  + 2*V1V2()*( F1()*Conjugate(F2()) ).Re();
   }
  double WC(){
     return  -(-VV1() + 2*h() )*F1().Rho2() - (-VV2() + 2*h())*F2().Rho2()   -   (-2*V1V2() - 4*h())*( F1()*Conjugate(F2()) ).Re();
   }
  double WD(){
    double QQ = _Q*_Q;
    double undersqrt1 = VV1()  -h();
    double undersqrt2 = VV2()  -h();
    return  -sqrt(h()) * ( 2 * sqrt(undersqrt1) * F1().Rho2() - 2*sqrt(undersqrt2)*F2().Rho2()
 			  + (QQ - mpi*mpi + _s3)*(_s1 - _s2 )*( F1()*Conjugate(F2()) ).Re()/QQ/sqrt(h0() ) );
  }
  double WE(){
   return  3*sqrt(h()*h0())*( F1()*Conjugate(F2()) ).Im();
  }
  double WSA(){
   double QQ = _Q*_Q;
   return  QQ*F4().Rho2();
  }
  double WSB(){
   double undersqrt1 = VV1()  -h();
    double undersqrt2 = VV2()  -h();
    return  -2*_Q* (sqrt(undersqrt1) * (F1()*Conjugate(F4())).Re() +   sqrt(undersqrt2)*(F2()*Conjugate(F4())).Re()  );
  }
  double WSD(){
   double QQ = _Q*_Q;
   return  2*sqrt(QQ*h())* ( (F1()*Conjugate(F4())).Re() - (F2()*Conjugate(F4())).Re()   );
  }
  double WSC(){
   double undersqrt1 = VV1()  -h();
    double undersqrt2 = VV2()  -h();
    return  2*_Q* (sqrt(undersqrt1) * (F1()*Conjugate(F4())).Im() +   sqrt(undersqrt2)*(F2()*Conjugate(F4())).Im()  );
  }
  double WSE(){
   double QQ = _Q*_Q;
    return  -2*sqrt(QQ*h())* ( (F1()*Conjugate(F4())).Im() - (F2()*Conjugate(F4())).Im()   );
  }

  double V1(){
    double QQ = _Q*_Q;
    return  4*mpi*mpi -_s2  - pow(_s3  - _s1,2)/4/QQ;
  }
  double V2(){
    double QQ = _Q*_Q;
    return  4*mpi*mpi -_s1  - pow(_s3  - _s2,2)/4/QQ;
  }
  double VV12(){
    double QQ = _Q*_Q;
    return  -(QQ/2 - _s3 - 0.5*mpi*mpi) - (_s3 - _s1)*(_s3 - _s2)/4/QQ;
  }
  double JJ(){
    double QQ = _Q*_Q;
    return  (V1()*BreitWigner(sqrt(_s2),"rho").Rho2() + V2()*BreitWigner(sqrt(_s1),"rho").Rho2()  + VV12()*( BreitWigner(sqrt(_s1),"rho")*Conjugate(BreitWigner(sqrt(_s2),"rho")) + BreitWigner(sqrt(_s2),"rho")*Conjugate(BreitWigner(sqrt(_s1),"rho"))  ))*f3(sqrt(QQ)).Rho2();
  }
  //double JN();
  TComplex f3(double Q){
    return  (coscab*2*sqrt(2)/3/fpi)*BreitWigner(Q,"a1");
  }
  TLorentzVector JRe(TLorentzVector q1, TLorentzVector q2, TLorentzVector q3, TLorentzVector a1){

    double s1 = (q2+q3).M2();
    double s2 = (q1+q3).M2();
    //  double s3 = (q2+q3).M2();

    TLorentzVector vec1 = q1 - q3 -  a1* (a1*(q1-q3)/a1.M2());
    TLorentzVector vec2 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());

    TComplex BWProd1 = f3(a1.M())*BreitWigner(sqrt(s2),"rho");
    TComplex BWProd2 = f3(a1.M())*BreitWigner(sqrt(s1),"rho");

    TLorentzVector out = vec1*BWProd1.Re() + vec2*BWProd2.Re();
    return out;
  }
  TLorentzVector JIm(TLorentzVector q1, TLorentzVector q2, TLorentzVector q3, TLorentzVector a1){

    double s1 = (q2+q3).M2();
    double s2 = (q1+q3).M2();
    //  double s3 = (q2+q3).M2();

    TLorentzVector vec1 = q1 - q3 -  a1* (a1*(q1-q3)/a1.M2());
    TLorentzVector vec2 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());

    TComplex BWProd1 = f3(a1.M())*BreitWigner(sqrt(s2),"rho");
    TComplex BWProd2 = f3(a1.M())*BreitWigner(sqrt(s1),"rho");

    TLorentzVector out = vec1*BWProd1.Im() + vec2*BWProd2.Im();
    return out;
  }

  TLorentzVector JCRe(TLorentzVector q1, TLorentzVector q2, TLorentzVector q3, TLorentzVector a1){

    double s1 = (q2+q3).M2();
    double s2 = (q1+q3).M2();
    //  double s3 = (q2+q3).M2();

    TLorentzVector vec1 = q1 - q3 -  a1* (a1*(q1-q3)/a1.M2());
    TLorentzVector vec2 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());

    TComplex BWProd1 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s2),"rho"));
    TComplex BWProd2 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s1),"rho"));
    TLorentzVector out = vec1*BWProd1.Re() + vec2*BWProd2.Re();
    return out;
  }
  TLorentzVector JCIm(TLorentzVector q1, TLorentzVector q2, TLorentzVector q3, TLorentzVector a1){

    double s1 = (q2+q3).M2();
    double s2 = (q1+q3).M2();
    //  double s3 = (q2+q3).M2();

    TLorentzVector vec1 = q1 - q3 -  a1* (a1*(q1-q3)/a1.M2());
    TLorentzVector vec2 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());

    TComplex BWProd1 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s2),"rho"));
    TComplex BWProd2 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s1),"rho"));

    TLorentzVector out = vec1*BWProd1.Im() + vec2*BWProd2.Im();
    return out;
  }
  TComplex JN(TLorentzVector q1, TLorentzVector q2, TLorentzVector q3, TLorentzVector a1, TLorentzVector N){
    double s1 = (q2+q3).M2();
    double s2 = (q1+q3).M2();


    TLorentzVector vec1 = q1 - q3 -  a1* (a1*(q1-q3)/a1.M2());
    TLorentzVector vec2 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());

    double prod1 = vec1*N;
    double prod2 = vec2*N;

    TComplex BWProd1 = f3(a1.M())*BreitWigner(sqrt(s2),"rho");
    TComplex BWProd2 = f3(a1.M())*BreitWigner(sqrt(s1),"rho");

    TComplex out  = BWProd1*prod1 + BWProd2*prod2;
    return out;
  }
  TComplex JCN(TLorentzVector q1, TLorentzVector q2, TLorentzVector q3, TLorentzVector a1, TLorentzVector N){
    double s1 = (q2+q3).M2();
    double s2 = (q1+q3).M2();

    TLorentzVector vec1 = q1 - q3 -  a1* (a1*(q1-q3)/a1.M2());
    TLorentzVector vec2 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());

    double prod1 = vec1*N;
    double prod2 = vec2*N;
    TComplex BWProd1 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s2),"rho"));
    TComplex BWProd2 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s1),"rho"));
    TComplex out  = BWProd1*prod1 + BWProd2*prod2;
    return out;
  }

  TLorentzVector PTZ5(TLorentzVector aR, TLorentzVector aI, TLorentzVector bR, TLorentzVector bI, TLorentzVector c){
    TComplex a4(aR.E(), aI.E());  TComplex a1(aR.Px(),aI.Px());   TComplex a2(aR.Py(),aI.Py());   TComplex a3(aR.Pz(),aI.Pz());
    TComplex b4(bR.E(), bI.E());  TComplex b1(bR.Px(),bI.Px());   TComplex b2(bR.Py(),bI.Py());   TComplex b3(bR.Pz(),bI.Pz());

    double  c1 = c.Px();   double  c2 = c.Py();   double  c3 = c.Pz();   double  c4 = c.E();

    double d34 = (a3*b4 - a4*b3).Im();
    double d24 = (a2*b4 - a4*b2).Im();
    double d23 = (a2*b3 - a3*b2).Im();
    double d14 = (a1*b4 - a4*b1).Im();
    double d13 = (a1*b3 - a3*b1).Im();
    double d12 = (a1*b2 - a2*b1).Im();

    double PIAX1 = 2*( c2*d34 - c3*d24 + c4*d23);
    double PIAX2 = 2*(-c1*d34 + c3*d14 - c4*d13);
    double PIAX3 = 2*( c1*d24 - c2*d14 + c4*d12);
    double PIAX4 = 2*(-c1*d23 + c2*d13 - c3*d12);

    TLorentzVector d(PIAX1,PIAX2,PIAX3,PIAX4);
    return d;
  }
  TLorentzVector PTZ(TLorentzVector q1, TLorentzVector q2, TLorentzVector q3, TLorentzVector a1, TLorentzVector N){
    double s1 = (q2+q3).M2();
    double s2 = (q1+q3).M2();
    //  double s3 = (q2+q3).M2();

    TLorentzVector vec1 = q1 - q3 -  a1* (a1*(q1-q3)/a1.M2());
    TLorentzVector vec2 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());



    TComplex L1 = f3(a1.M())*BreitWigner(sqrt(s2),"rho");
    TComplex L2 = f3(a1.M())*BreitWigner(sqrt(s1),"rho");

    TComplex CL1 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s2),"rho"));
    TComplex CL2 = Conjugate(f3(a1.M())*BreitWigner(sqrt(s1),"rho"));



    TComplex factor1=JCN(q1,q2,q3,a1,N)*L1 + JN(q1,q2,q3,a1,N)*CL1;
    TComplex factor2=JCN(q1,q2,q3,a1,N)*L2 + JN(q1,q2,q3,a1,N)*CL2;

    TLorentzVector  Ptenz= 2*BreitWigner(sqrt(s1),"rho").Rho2()*(vec2*N)*vec2 + 2*BreitWigner(sqrt(s2),"rho").Rho2()*(vec1*N)*vec1 + 2*(BreitWigner(sqrt(s2),"rho")*Conjugate(BreitWigner(sqrt(s1),"rho"))  ).Re() *((vec1*N)*vec2 + (vec2*N)*vec1) - JJ()*N;

    TLorentzVector out  = 2*(factor1*vec1 + factor2*vec2 - JJ()*N);
    return out;
  }
  TLorentzVector PVC(){
     TLorentzVector q1= _ss1pionLV;
     TLorentzVector q2= _ss2pionLV;
     TLorentzVector q3= _osPionLV;
     TLorentzVector a1 = q1+q2+q3;
     TLorentzVector N = _nuLV;
     TLorentzVector P = _tauLV;
     double s1 = (q2+q3).M2();
     double s2 = (q1+q3).M2();
     double s3 = (q1+q2).M2();


     TLorentzVector vec1 = q2 - q3 -  a1* (a1*(q2-q3)/a1.M2());
     TLorentzVector vec2 = q3 - q1 -  a1* (a1*(q3-q1)/a1.M2());
     TLorentzVector vec3 = q1 - q2 -  a1* (a1*(q1-q2)/a1.M2());

     TComplex F1 = TComplex(COEF1)*F3PI(1,a1.M2(),s1,s2);
     TComplex F2 = TComplex(COEF2)*F3PI(2,a1.M2(),s2,s1);
     TComplex F3 = TComplex(COEF3)*F3PI(3,a1.M2(),s3,s1);


     std::vector<TComplex> HADCUR;
     std::vector<TComplex> HADCURC;

     HADCUR.push_back(TComplex(vec1.E())*F1  + TComplex(vec2.E())*F2   +   TComplex(vec3.E())*F3 ); // energy component goes first
     HADCUR.push_back(TComplex(vec1.Px())*F1 + TComplex(vec2.Px())*F2  +   TComplex(vec3.Px())*F3 );
     HADCUR.push_back(TComplex(vec1.Py())*F1 + TComplex(vec2.Py())*F2  +   TComplex(vec3.Py())*F3 );
     HADCUR.push_back(TComplex(vec1.Pz())*F1 + TComplex(vec2.Pz())*F2  +   TComplex(vec3.Pz())*F3 );


     HADCURC.push_back(Conjugate(TComplex(vec1.E())*F1  + TComplex(vec2.E())*F2   +   TComplex(vec3.E())*F3) ); // energy component goes first
     HADCURC.push_back(Conjugate(TComplex(vec1.Px())*F1 + TComplex(vec2.Px())*F2  +   TComplex(vec3.Px())*F3 ));
     HADCURC.push_back(Conjugate(TComplex(vec1.Py())*F1 + TComplex(vec2.Py())*F2  +   TComplex(vec3.Py())*F3 ) );
     HADCURC.push_back(Conjugate(TComplex(vec1.Pz())*F1 + TComplex(vec2.Pz())*F2  +   TComplex(vec3.Pz())*F3) );

     TLorentzVector CLV =  CLVEC(HADCUR,HADCURC,N );
     TLorentzVector CLA =  CLAXI(HADCUR,HADCURC,N );

     TComplex BWProd1 = f3(a1.M())*BreitWigner(sqrt(s2),"rho");
     TComplex BWProd2 = f3(a1.M())*BreitWigner(sqrt(s1),"rho");

     double omega = P*CLV - P*CLA;
     return (P.M()*P.M()*  (CLA - CLV)  -  P*(  P*CLA -  P*CLV))*(1/omega/P.M());
  }

  TLorentzVector CLVEC(std::vector<TComplex> H, std::vector<TComplex> HC, TLorentzVector N){
    TComplex HN  = H.at(0)*N.E()     - H.at(1)*N.Px()    - H.at(2)*N.Py()   - H.at(3)*N.Pz();
    TComplex HCN = HC.at(0)*N.E()    - HC.at(1)*N.Px()   - HC.at(2)*N.Py()  - HC.at(3)*N.Pz();
    double   HH  = (H.at(0)*HC.at(0) - H.at(1)*HC.at(1)  - H.at(2)*HC.at(2) - H.at(3)*HC.at(3)).Re();

    double PIVEC0 = 2*(   2*(HN*HC.at(0)).Re()  - HH*N.E()   );
    double PIVEC1 = 2*(   2*(HN*HC.at(1)).Re()  - HH*N.Px()  );
    double PIVEC2 = 2*(   2*(HN*HC.at(2)).Re()  - HH*N.Py()  );
    double PIVEC3 = 2*(   2*(HN*HC.at(3)).Re()  - HH*N.Pz()  );
    return TLorentzVector( PIVEC1, PIVEC2, PIVEC3, PIVEC0);
  }
  TLorentzVector CLAXI(std::vector<TComplex> H, std::vector<TComplex> HC, TLorentzVector N){
    TComplex a4 = HC.at(0);   TComplex a1 =  HC.at(1);   TComplex a2 =  HC.at(2);   TComplex a3 =  HC.at(3);
    TComplex b4 = H.at(0);    TComplex b1 =  H.at(1);    TComplex b2 =  H.at(2);    TComplex b3 =  H.at(3);
    double   c4 = N.E();      double   c1 =  N.Px();     double   c2 =  N.Py();     double   c3 = N.Pz();
    double d34 = (a3*b4 - a4*b3).Im();
    double d24 = (a2*b4 - a4*b2).Im();
    double d23 = (a2*b3 - a3*b2).Im();
    double d14 = (a1*b4 - a4*b1).Im();
    double d13 = (a1*b3 - a3*b1).Im();
    double d12 = (a1*b2 - a2*b1).Im();

    double PIAX0 = -SIGN*2*(-c1*d23 + c2*d13 - c3*d12);
    double PIAX1 = SIGN*2*( c2*d34 - c3*d24 + c4*d23);
    double PIAX2 = SIGN*2*(-c1*d34 + c3*d14 - c4*d13);
    double PIAX3 = SIGN*2*( c1*d24 - c2*d14 + c4*d12);

    return TLorentzVector( PIAX1,PIAX2,PIAX3,PIAX0);
  }

  double VV1(){ //  this is -V1^{2}
    double QQ = _Q*_Q;
    return  _s2 - 4*mpi*mpi + pow(_s3 - _s1,2)/4/QQ;
  }
  double VV2(){ //  this is -V2^{2}
    double QQ = _Q*_Q;
    return  _s1 - 4*mpi*mpi + pow(_s3 - _s2,2)/4/QQ;
  }
  double V1V2(){  // this is -V1V2
    double QQ = _Q*_Q;
    return  (QQ/2 - _s3 - 0.5*mpi*mpi) + (_s3 - _s1)*(_s3 - _s2)/4/QQ;
  }
  double h0(){ // this is -3sqrt{h0}/2
    double QQ = _Q*_Q;
    return -4*mpi*mpi + pow(2*mpi*mpi - _s1 - _s2,2)/QQ;
  }
  double h(){
    double QQ = _Q*_Q;
    return (_s1*_s2*_s3 - mpi*mpi*pow(QQ - mpi*mpi,2))/h0()/QQ;  // this is sqrt{h}
  }

  TVector3 nL(){
    return   -LFa1LV.Vect()*(1/LFa1LV.Vect().Mag());
  }
  TVector3 nT(){
    return   _tauLV.Vect()*(1/_tauLV.Vect().Mag());
  }
  TVector3 nTZLFr(){
    return _tauAlongZLabFrame.Vect()*(1/_tauAlongZLabFrame.Vect().Mag());
  }
  TVector3 nPerp(){
    if(_ss1pionLV.Vect().Cross(_ss2pionLV.Vect()).Mag()==0){if(debug){ std::cout<<"  Can not return nPerp, same sign pions seem to be parallel in a1 rest frame, return 0,0,0  "<<std::endl;} return TVector3(0,0,0);}

    TVector3 nss1= _ss1pionLV.Vect()*(1/_ss1pionLV.Vect().Mag());
    TVector3 nss2= _ss2pionLV.Vect()*(1/_ss2pionLV.Vect().Mag());
    return   (nss1.Cross(nss2))*(1/(nss1.Cross(nss2)).Mag());
  }
  TVector3 ns(){
    return   sLV().Vect()*(1/sLV().Vect().Mag());
  }
  TLorentzVector sLV(){
    double QQ = _Q*_Q;
    double l0 = 0.5*(mtau*mtau + QQ)/sqrt(QQ);
    double l   = 0.5*(mtau*mtau  - QQ)/sqrt(QQ);
    return TLorentzVector(sinthetaLF(),0,-l0*costhetaLF()/mtau,-l*costhetaLF()/mtau);
  }



  TComplex  BreitWigner(double Q, string type="rho"){
    double QQ=Q*Q;
    double re,im;
    double m = Mass(type);
    double g  = Widths(Q,type);
    re = (m*m*(m*m - QQ))/(pow(m*m - QQ,2) + m*m*g*g);
    im = Q*g/(pow(m*m - QQ,2) + m*m*g*g);


    TComplex out(re,im);
    return out;
  }
  TComplex  BRho(double Q){
     return (BreitWigner(Q) + beta*BreitWigner(Q,"rhoprime"))/(1+beta);
  }
  TComplex F1(){
    TComplex scale(0, -2*sqrt(2)/3/fpi);
    TComplex res = scale*BreitWigner(_Q,"a1")*BRho(sqrt(_s2));
    return res;
  }
  TComplex F2(){
    TComplex scale(0, -2*sqrt(2)/3/fpi);
    TComplex res = scale*BreitWigner(_Q,"a1")*BRho(sqrt(_s1));
    return res;
  }
  TComplex F4(){
    TComplex scale(0, -gpiprimerhopi*grhopipi*fpiprime/2/pow(mrho,4)/pow(mpiprime,3));
    TComplex res = scale*BreitWigner(_Q,"piprime")*(_s1*(_s2-_s3)*BRho(sqrt(_s1)) + _s2*(_s1-_s3)*BRho(sqrt(_s2)));
    return res;
  }
  TComplex Conjugate(TComplex a){
    return TComplex(a.Re(), -a.Im());
  }
  TVector3 Rotate(TVector3 LVec, TVector3 Rot){
    TVector3 vec = LVec;
    vec.RotateZ(0.5*TMath::Pi() - Rot.Phi());
    vec.RotateX(Rot.Theta());
    return vec;
  }
  double  Widths(double Q, string type="rho"){
    double QQ = Q*Q;
    double Gamma;
    Gamma = Gamma0rho*mrho*pow( ppi(QQ)  / ppi(mrho*mrho), 3) /sqrt(QQ);
    if(type == "rhoprime"){
      Gamma=Gamma0rhoprime*QQ/mrhoprime/mrhoprime;
   }
    if(type == "a1"){
      Gamma=Gamma0a1*ga1(Q)/ga1(ma1);
      //    Gamma=Gamma0a1*ma1*ga1(Q)/ga1(ma1)/Q;
   }
    if(type == "piprime"){
      Gamma = Gamma0piprime*pow( sqrt(QQ)/mpiprime  ,5)*pow( (1-mrho*mrho/QQ)/(1-mrho*mrho/mpiprime/mpiprime) ,3);
    }
    return Gamma;
  }
  double ppi(double QQ){  if(QQ < 4*mpi*mpi) std::cout<<"Warning! Can not compute ppi(Q); root square <0 ; return nan  "; return 0.5*sqrt(QQ - 4*mpi*mpi);}
  double ga1(double  Q){
    double QQ = Q*Q;
    return (QQ > pow(mrho + mpi,2)) ?  QQ*(1.623 + 10.38/QQ - 9.32/QQ/QQ   + 0.65/QQ/QQ/QQ)  : 4.1*pow(QQ - 9*mpi*mpi,3)*(  1 - 3.3*(QQ - 9*mpi*mpi)  + 5.8*pow(QQ - 9*mpi*mpi,2)  );
  }
  double Mass(string type="rho"){
    double m = mrho;
    if(type == "rhoprime") return mrhoprime;
    if(type == "a1") return ma1;
    if(type == "piprime") return mpiprime;
    return m;
  }

  TComplex BWIGML(double S, double M,  double G, double m1, double m2, int L){
    int IPOW;
    double MP = pow(m1+m2,2);
    double MM = pow(m1-m2,2);
    double MSQ = M*M;
    double W = sqrt(S);
    double WGS =0.0;
    double QS,QM;
    if(W > m1+m2){
      QS = sqrt(std::fabs( (S  - MP)*(S  - MM)))/W;
      QM = sqrt(std::fabs( (MSQ - MP)*(MSQ - MM)))/M;
      IPOW = 2*L +1;
      WGS=G*(MSQ/W)*pow(QS/QM, IPOW);
    }

   TComplex out;
   out = TComplex(MSQ,0)/TComplex(MSQ - S, -WGS) ;
   return out;
  }
  TComplex FPIKM(double W, double XM1, double XM2){
    double ROM  = 0.773;
    double ROG  = 0.145;
    double ROM1 = 1.370;
    double ROG1 = 0.510;
    double BETA1=-0.145;

    double S=W*W;
    int L =1; // P-wave
    TComplex out = (BWIGML(S,ROM,ROG,XM1,XM2,L) + BETA1*BWIGML(S,ROM1,ROG1,XM1,XM2,L))/(1+BETA1);
    return out;
  }
  TComplex F3PI(double IFORM,double QQ,double SA,double SB){
    double MRO = 0.7743;
    double GRO = 0.1491;
    double MRP = 1.370 ;
    double GRP = 0.386 ;
    double MF2 = 1.275;
    double GF2 = 0.185;
    double MF0 = 1.186;
    double GF0 = 0.350;
    double MSG = 0.860;
    double GSG = 0.880;
    double MPIZ = mpi0;
    double MPIC = mpi;
    double M1;
    double M2;
    double M3;
    int IDK =1;  // --------- it is 3pi
    if(IDK ==1){
      M1=MPIZ;
      M2=MPIZ;
      M3=MPIC;
    }
    if(IDK==2){
      M1=MPIC;
      M2=MPIC;
      M3=MPIC;
    }


    double M1SQ = M1*M1;
    double M2SQ = M2*M2;
    double M3SQ = M3*M3;

    // parameter varioation for
    // systematics   from, https://arxiv.org/pdf/hep-ex/9902022.pdf

    double db2 = 0.094;   double dph2 = 0.253;
    double db3 = 0.094;   double dph3 = 0.104;
    double db4 = 0.296;   double dph4 = 0.170;
    double db5 = 0.167;   double dph5 = 0.104;
    double db6 = 0.284;   double dph6 = 0.036;
    double db7 = 0.148;   double dph7 = 0.063;

    double scale(0.);
    if(doSystematic)
      {
        if(systType=="UP")
  	{
  	  scale =  1;
  	}


        if(systType=="DOWN")
  	{
  	  scale = -1;
  	}
      }

    TComplex  BT1 = TComplex(1.,0.);
    TComplex  BT2 = TComplex(0.12  + scale*db2,0.)*TComplex(1, (0.99   +  scale*dph2)*TMath::Pi(), true);//  TComplex(1, 0.99*TMath::Pi(), true);   Real part must be equal to one, stupid polar implemenation in root
    TComplex  BT3 = TComplex(0.37  + scale*db3,0.)*TComplex(1, (-0.15  +  scale*dph3)*TMath::Pi(), true);
    TComplex  BT4 = TComplex(0.87  + scale*db4,0.)*TComplex(1, (0.53   +  scale*dph4)*TMath::Pi(), true);
    TComplex  BT5 = TComplex(0.71  + scale*db5,0.)*TComplex(1, (0.56   +  scale*dph5)*TMath::Pi(), true);
    TComplex  BT6 = TComplex(2.10  + scale*db6,0.)*TComplex(1, (0.23   +  scale*dph6)*TMath::Pi(), true);
    TComplex  BT7 = TComplex(0.77  + scale*db7,0.)*TComplex(1, (-0.54  +  scale*dph7)*TMath::Pi(), true);

    TComplex  F3PIFactor(0.,0.); // initialize to zero
    if(IDK == 2){
      if(IFORM == 1 || IFORM == 2 ){
        double S1 = SA;
        double S2 = SB;
        double S3 = QQ-SA-SB+M1SQ+M2SQ+M3SQ;
        //Lorentz invariants for all the contributions:
        double F134 = -(1./3.)*((S3-M3SQ)-(S1-M1SQ));
        double F15A = -(1./2.)*((S2-M2SQ)-(S3-M3SQ));
        double F15B = -(1./18.)*(QQ-M2SQ+S2)*(2.*M1SQ+2.*M3SQ-S2)/S2;
        double F167 = -(2./3.);

        // Breit Wigners for all the contributions:


        TComplex  FRO1 = BWIGML(S1,MRO,GRO,M2,M3,1);
        TComplex  FRP1 = BWIGML(S1,MRP,GRP,M2,M3,1);
        TComplex  FRO2 = BWIGML(S2,MRO,GRO,M3,M1,1);
        TComplex  FRP2 = BWIGML(S2,MRP,GRP,M3,M1,1);
        TComplex  FF21 = BWIGML(S1,MF2,GF2,M2,M3,2);
        TComplex  FF22 = BWIGML(S2,MF2,GF2,M3,M1,2);
        TComplex  FSG2 = BWIGML(S2,MSG,GSG,M3,M1,0);
        TComplex  FF02 = BWIGML(S2,MF0,GF0,M3,M1,0);


        F3PIFactor = BT1*FRO1+BT2*FRP1+
  	BT3*TComplex(F134,0.)*FRO2+BT4*TComplex(F134,0.)*FRP2
  	-BT5*TComplex(F15A,0.)*FF21-BT5*TComplex(F15B,0.)*FF22
  	-BT6*TComplex(F167,0.)*FSG2-BT7*TComplex(F167,0.)*FF02;

      } else if (IFORM == 3 ){

        double S3 = SA;
        double S1 = SB;
        double S2 = QQ-SA-SB+M1SQ+M2SQ+M3SQ;

        double F34A = (1./3.)*((S2-M2SQ)-(S3-M3SQ));
        double F34B = (1./3.)*((S3-M3SQ)-(S1-M1SQ));
        double F35A = -(1./18.)*(QQ-M1SQ+S1)*(2.*M2SQ+2.*M3SQ-S1)/S1;
        double F35B =  (1./18.)*(QQ-M2SQ+S2)*(2.*M3SQ+2.*M1SQ-S2)/S2;
        double F36A = -(2./3.);
        double F36B =  (2./3.);

        //C Breit Wigners for all the contributions:
        TComplex  FRO1 = BWIGML(S1,MRO,GRO,M2,M3,1);
        TComplex  FRP1 = BWIGML(S1,MRP,GRP,M2,M3,1);
        TComplex  FRO2 = BWIGML(S2,MRO,GRO,M3,M1,1);
        TComplex  FRP2 = BWIGML(S2,MRP,GRP,M3,M1,1);
        TComplex  FF21 = BWIGML(S1,MF2,GF2,M2,M3,2);
        TComplex  FF22 = BWIGML(S2,MF2,GF2,M3,M1,2);
        TComplex  FSG1 = BWIGML(S1,MSG,GSG,M2,M3,0);
        TComplex  FSG2 = BWIGML(S2,MSG,GSG,M3,M1,0);
        TComplex  FF01 = BWIGML(S1,MF0,GF0,M2,M3,0);
        TComplex  FF02 = BWIGML(S2,MF0,GF0,M3,M1,0);

        F3PIFactor =
  	BT3*(TComplex(F34A,0.)*FRO1+TComplex(F34B,0.)*FRO2)+
  	BT4*(TComplex(F34A,0.)*FRP1+TComplex(F34B,0.)*FRP2)
  	-BT5*(TComplex(F35A,0.)*FF21+TComplex(F35B,0.)*FF22)
  	-BT6*(TComplex(F36A,0.)*FSG1+TComplex(F36B,0.)*FSG2)
  	-BT7*(TComplex(F36A,0.)*FF01+TComplex(F36B,0.)*FF02);

        // F3PIFactor = TComplex(0.,0.);
      }
    }

    if(IDK==1){
     if(IFORM == 1 || IFORM == 2 ){
       double  S1 = SA;
       double  S2 = SB;
       double  S3 = QQ-SA-SB+M1SQ+M2SQ+M3SQ;

  // C it is 2pi0pi-
  // C Lorentz invariants for all the contributions:
      double   F134 = -(1./3.)*((S3-M3SQ)-(S1-M1SQ));
      double   F150 =  (1./18.)*(QQ-M3SQ+S3)*(2.*M1SQ+2.*M2SQ-S3)/S3;
      double   F167 =  (2./3.);

      //C Breit Wigners for all the contributions:
      TComplex FRO1 = BWIGML(S1,MRO,GRO,M2,M3,1);
      TComplex FRP1 = BWIGML(S1,MRP,GRP,M2,M3,1);
      TComplex FRO2 = BWIGML(S2,MRO,GRO,M3,M1,1);
      TComplex FRP2 = BWIGML(S2,MRP,GRP,M3,M1,1);
      TComplex FF23 = BWIGML(S3,MF2,GF2,M1,M2,2);
      TComplex FSG3 = BWIGML(S3,MSG,GSG,M1,M2,0);
      TComplex FF03 = BWIGML(S3,MF0,GF0,M1,M2,0);

      F3PIFactor = BT1*FRO1+BT2*FRP1+
        BT3*TComplex(F134,0.)*FRO2+BT4*TComplex(F134,0.)*FRP2+
        BT5*TComplex(F150,0.)*FF23+
        BT6*TComplex(F167,0.)*FSG3+BT7*TComplex(F167,0.)*FF03;
     }
     else if (IFORM == 3 ){
       double   S3 = SA;
       double   S1 = SB;
       double   S2 = QQ-SA-SB+M1SQ+M2SQ+M3SQ;

       double F34A = (1./3.)*((S2-M2SQ)-(S3-M3SQ));
       double F34B = (1./3.)*((S3-M3SQ)-(S1-M1SQ));
       double F35  =-(1./2.)*((S1-M1SQ)-(S2-M2SQ));

       //C Breit Wigners for all the contributions:
       TComplex FRO1 = BWIGML(S1,MRO,GRO,M2,M3,1);
       TComplex FRP1 = BWIGML(S1,MRP,GRP,M2,M3,1);
       TComplex FRO2 = BWIGML(S2,MRO,GRO,M3,M1,1);
       TComplex FRP2 = BWIGML(S2,MRP,GRP,M3,M1,1);
       TComplex FF23 = BWIGML(S3,MF2,GF2,M1,M2,2);

       F3PIFactor =
         BT3*(TComplex(F34A,0.)*FRO1+TComplex(F34B,0.)*FRO2)+
         BT4*(TComplex(F34A,0.)*FRP1+TComplex(F34B,0.)*FRP2)+
         BT5*TComplex(F35,0.)*FF23;

     }
    }
    TComplex FORMA1 = FA1A1P(QQ);
    return  F3PIFactor*FORMA1;
  }
  TComplex FA1A1P(double XMSQ){
    double  XM1 = 1.275000;
    double  XG1 =0.700 ;
    double  XM2 = 1.461000 ;
    double  XG2 = 0.250;
    TComplex BET = TComplex(0.00,0.);

    double GG1 = XM1*XG1/(1.3281*0.806);
    double GG2 = XM2*XG2/(1.3281*0.806);
    double XM1SQ = XM1*XM1;
    double XM2SQ = XM2*XM2;

    double GF = WGA1(XMSQ);
    double FG1 = GG1*GF;
    double FG2 = GG2*GF;
    TComplex F1 = TComplex(-XM1SQ,0.0)/TComplex(XMSQ-XM1SQ,FG1);
    TComplex F2 = TComplex(-XM2SQ,0.0)/TComplex(XMSQ-XM2SQ,FG2);
    TComplex FA1A1P = F1+BET*F2;

    return FA1A1P;
  }
  double WGA1(double QQ){
  // C mass-dependent M*Gamma of a1 through its decays to
  // C.   [(rho-pi S-wave) + (rho-pi D-wave) +
  // C.    (f2 pi D-wave) + (f0pi S-wave)]
  // C.  AND simple K*K S-wave
    double  MKST = 0.894;
    double  MK = 0.496;
    double  MK1SQ = (MKST+MK)*(MKST+MK);
    double  MK2SQ = (MKST-MK)*(MKST-MK);
    //C coupling constants squared:
    double   C3PI = 0.2384*0.2384;
    double   CKST = 4.7621*4.7621*C3PI;
  // C Parameterization of numerical integral of total width of a1 to 3pi.
  // C From M. Schmidtler, CBX-97-64-Update.
    double  S = QQ;
    double  WG3PIC = WGA1C(S);
    double  WG3PIN = WGA1N(S);

    //C Contribution to M*Gamma(m(3pi)^2) from S-wave K*K, if above threshold
    double  GKST = 0.0;
    if(S > MK1SQ) GKST = sqrt((S-MK1SQ)*(S-MK2SQ))/(2.*S);

    return C3PI*(WG3PIC+WG3PIN)+CKST*GKST;
  }
  double WGA1C(double S){
    double STH,Q0,Q1,Q2,P0,P1,P2,P3,P4,G1_IM;
    Q0 =   5.80900; Q1 =  -3.00980; Q2 =   4.57920;
    P0 = -13.91400; P1 =  27.67900; P2 = -13.39300;
    P3 =   3.19240; P4 =  -0.10487; STH=0.1753;

    if(S < STH){
      G1_IM=0.0;
    }else if(S > STH && S < 0.823){
      G1_IM = Q0*   pow(S-STH,3)   *(1. + Q1*(S-STH) + Q2*pow(S-STH,2));
    }
    else{
      G1_IM = P0 + P1*S + P2*S*S+ P3*S*S*S + P4*S*S*S*S;
    }
    return G1_IM;
  }
  double WGA1N(double S){
    double STH,Q0,Q1,Q2,P0,P1,P2,P3,P4,G1_IM;
    Q0 =   6.28450;Q1 =  -2.95950;Q2 =   4.33550;
    P0 = -15.41100;P1 =  32.08800;P2 = -17.66600;
    P3 =   4.93550;P4 =  -0.37498;STH   = 0.1676;
    if(S < STH){
      G1_IM = 0.0;
    }else if(S > STH && S < 0.823){
      G1_IM = Q0*pow(S-STH,3)*(1. + Q1*(S-STH) + Q2*pow(S-STH,2));
    }
    else{
      G1_IM = P0 + P1*S + P2*S*S+ P3*S*S*S + P4*S*S*S*S;
    }
    return G1_IM;
  }


  TComplex  BWa1(double QQ);
  TComplex  BWrho(double QQ);
  TComplex  BWrhoPrime(double QQ);
  double GammaA1(double QQ);
  double gForGammaA1(double QQ);
  double GammaRho(double QQ);
  double  GammaRhoPrime(double QQ);



  double GetOmegaA1();



 private:
  double mpi;
  double mpi0;
  double mtau;
  double coscab;
  double mrho;
  double mrhoprime;
  double ma1;
  double mpiprime;
  double Gamma0rho;
  double Gamma0rhoprime;
  double Gamma0a1;
  double Gamma0piprime;
  double fpi;
  double fpiprime;
  double gpiprimerhopi;
  double grhopipi;
  double beta;
  double COEF1;
  double COEF2;
  double COEF3;
  int SIGN;

  const TLorentzVector a1pos;
  const TLorentzVector a1pss1;
  const TLorentzVector a1pss2;

  bool debug;

  vector<TLorentzVector> TauA1andProd_RF;
  TLorentzVector _osPionLV;
  TLorentzVector _ss1pionLV;
  TLorentzVector _ss2pionLV;
  TLorentzVector _a1LV;
  TLorentzVector _tauLV;
  TLorentzVector _nuLV;
  TLorentzVector _s12;
  TLorentzVector _s13;
  TLorentzVector _s23;
  TLorentzVector _tauAlongZLabFrame;

  double _s1;
  double _s2;
  double _s3;
  double _Q;


  double LFQ;
  TLorentzVector   LFosPionLV;
  TLorentzVector   LFss1pionLV;
  TLorentzVector   LFss2pionLV;
  TLorentzVector   LFa1LV;
  TLorentzVector   LFtauLV;

  bool doSystematic;
  TString systType;
  TMatrixT<double> convertToMatrix(TVectorT<double> V){
    TMatrixT<double> M(V.GetNrows(),1);
    for(int i=0; i < M.GetNrows(); i++){
      M(i,0)=V(i);
    } return M;
  }


};
