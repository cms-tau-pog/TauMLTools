#ifndef HISTOGRAM_2D
#define HISTOGRAM_2D
/*
implementation of a 2d histogram with custom y-binning for each x bin.
*/

#include <iostream>
#include <map>
#include <utility>
#include <algorithm>
#include "TH2D.h"
#include "TH1D.h"
#include "TCanvas.h"

void load_axis_into_vector(const TAxis* axis, std::vector<double>& vector);

class Histogram_2D{
  public:
    Histogram_2D(const char* name, std::vector<double> xaxis, const double ymin, const double ymax);
    Histogram_2D(Histogram_2D& histo) = delete;
    ~Histogram_2D();

    void th2d_add (const TH2D& histo);
    void divide   (const Histogram_2D& histo);
    void reset();

    void add_y_binning_by_index(const int index, const std::vector<double> yaxis);

    bool can_be_imported(const TH2D& histo);
    void print(const char* dir);

    //use this function only for weights, not for counts
    TH2D& get_weights_th2d(const char* name, const char* title);

  private:
    int find_bin_by_value_(const double& x);

    std::string name_;

    std::vector<double>                 xaxis_;
    std::vector<std::shared_ptr<TH1D>>  xaxis_content_;

    std::shared_ptr<TH2D> histo_ = std::make_shared<TH2D>();

    double ymin_ = std::numeric_limits<float>::max();
    double ymax_ = std::numeric_limits<float>::lowest();
    
};

Histogram_2D::~Histogram_2D(){
}

Histogram_2D::Histogram_2D(const char* name, std::vector<double> xaxis, const double ymin, const double ymax){
  xaxis_ = xaxis;
  for (std::vector<double>::iterator it = xaxis.begin(); it != std::prev(xaxis.end()); it++){
    xaxis_content_.push_back(std::make_shared<TH1D>());
  }
  ymin_ = ymin;
  ymax_ = ymax;
  name_ = name;
}

void Histogram_2D::add_y_binning_by_index(const int index, const std::vector<double> yaxis){
  std::string name = name_+std::to_string(index);
  if(index >= xaxis_content_.size()){
    std::runtime_error("Index "+std::to_string(index)+" out of x-axis range");
  }
  xaxis_content_[index] = std::shared_ptr<TH1D>(new TH1D(name.c_str(), "", yaxis.size()-1, &yaxis[0]));
}

bool Histogram_2D::can_be_imported(const TH2D& histo){
  auto check_axis = [](const std::vector<double>& input_axis, const std::vector<double>& this_axis){
    if (input_axis.front() != this_axis.front() || input_axis.back() != this_axis.back()) return false;
    bool matching;
    for (auto this_low_edge  : this_axis ){
    for (auto input_low_edge : input_axis){
      matching = (fabs(this_low_edge-input_low_edge) < 2*std::numeric_limits<float>::epsilon());
      if (matching) break;
    } if (!matching) return false;
    }
    return true;
  };

  std::vector<double> input_xaxis;
  std::vector<double> input_yaxis;
  std::vector<double> this_yaxis;

  //load_axis_into_vector(histo.GetXaxis(), input_xaxis);
  //load_axis_into_vector(histo.GetYaxis(), input_yaxis);
  // WORKAROUND to switch x and y axes
  std::cout << "WARNING from can_be_imported: x and y axes are switched!" << std::endl;
  load_axis_into_vector(histo.GetYaxis(), input_xaxis);
  load_axis_into_vector(histo.GetXaxis(), input_yaxis);
  
  if (!check_axis(input_xaxis, xaxis_)){
    std::cerr << "Invalid x-axis binning found" << std::endl;
    return false;
  }

  for(int ix = 0; ix < xaxis_.size()-1; ix++){
    auto this_histo = xaxis_content_[ix];
    load_axis_into_vector(this_histo->GetXaxis(), this_yaxis);
    if(!check_axis(input_yaxis, this_yaxis)){
      std::cerr << "Invalid y axis binning found for x bin n. " << ix << std::endl;
      return false;
    }
    this_yaxis.clear();
  }
  return true;
}

int Histogram_2D::find_bin_by_value_(const double& x){
  for(int ix = 0; ix < xaxis_.size() - 1; ix++){
    if(x >= xaxis_[ix] && x < xaxis_[ix+1]) return ix;
  }
  throw std::range_error("Value "+std::to_string(x)+" is not in the range of the x axis");
}

void Histogram_2D::th2d_add (const TH2D& histo){
  if(!can_be_imported(histo)){
    throw std::invalid_argument("Given TH2D "+std::string(histo.GetName())+" can not be imported");
  }
  
  //auto input_yaxis = histo.GetYaxis();
  //auto input_xaxis = histo.GetXaxis();
  // WORKAROUND to switch x and y axes
  std::cout << "WARNING from th2d_add: x and y axes are switched!" << std::endl;
  auto input_yaxis = histo.GetXaxis();
  auto input_xaxis = histo.GetYaxis();

  for (int iy = 1; iy <= input_yaxis->GetNbins(); iy++){
  for (int ix = 1; ix <= input_xaxis->GetNbins(); ix++){
    auto bincy = input_yaxis->GetBinCenter(iy);
    auto bincx = input_xaxis->GetBinCenter(ix);

    auto yhisto = xaxis_content_[find_bin_by_value_(bincx)];
    yhisto->SetBinContent(
      yhisto->FindBin(bincy), 
      //yhisto->GetBinContent(yhisto->FindBin(bincy)) + histo.GetBinContent(ix, iy));
      // WORKAROUND to switch x and y axes
      yhisto->GetBinContent(yhisto->FindBin(bincy)) + histo.GetBinContent(iy, ix));
  }}
}

void Histogram_2D::divide(const Histogram_2D& histo){
  auto check_axis = [] (const std::vector<double>& axis1, const std::vector<double>& axis2){
    return axis1 == axis2;
  };

  if (!check_axis(xaxis_, histo.xaxis_)){
    throw std::logic_error("Invalid x binning detected on denominator for Histogram_2D "+histo.name_);
  }
  
  std::vector<double> thisyaxis;
  std::vector<double> yaxis;
  for (int ix = 0; ix < xaxis_content_.size(); ix++){
    auto thisyhisto = xaxis_content_[ix];
    auto yhisto     = histo.xaxis_content_[ix];
    load_axis_into_vector(thisyhisto->GetXaxis(), thisyaxis);
    load_axis_into_vector(yhisto->GetXaxis()    , yaxis    );

    if(!check_axis(thisyaxis, yaxis)){
      throw std::logic_error("Invalid y-axis binning found for denominator in x bin n. "+std::to_string(ix)+" for Histogram_2D "+histo.name_);
    }

    (*thisyhisto).Divide(yhisto.get());
  }
}

TH2D& Histogram_2D::get_weights_th2d(const char* name, const char* title){
  double xwidthmin = std::numeric_limits<float>::max();
  double ywidthmin = std::numeric_limits<float>::max();

  auto get_min_width = [](std::vector<double>& vector, double& min){
    for (int i = 0; i < vector.size()-1; i++) min = std::min(min, vector[i+1] - vector[i]);
  };

  get_min_width(xaxis_, xwidthmin);

  std::vector<double> yaxis;
  for (int ix = 0; ix < xaxis_content_.size(); ix++){
    auto yhisto = xaxis_content_[ix];
    load_axis_into_vector(yhisto->GetXaxis(), yaxis);
    get_min_width(yaxis, ywidthmin);
    yaxis.clear();
  }

  int nx = (xaxis_.back() - xaxis_.front()) / xwidthmin;
  int ny = (ymax_ - ymin_) / ywidthmin;

  //histo_ = std::shared_ptr<TH2D>(new TH2D(name, title, nx, xaxis_.front(), xaxis_.back(), ny, ymin_, ymax_));
  // WORKAROUND to switch x and y axes
  std::cout << "WARNING from get_weights_th2d: x and y axes are switched!" << std::endl;
  histo_ = std::shared_ptr<TH2D>(new TH2D(name, title, ny, ymin_, ymax_, nx, xaxis_.front(), xaxis_.back()));

  for(int iy = 1; iy <= ny; iy++){
  for(int ix = 1; ix <= nx; ix++){
    //double bincx = histo_->GetXaxis()->GetBinCenter(ix);
    //double bincy = histo_->GetYaxis()->GetBinCenter(iy);
    // WORKAROUND to switch x and y axess
    double bincx = histo_->GetYaxis()->GetBinCenter(ix);
    double bincy = histo_->GetXaxis()->GetBinCenter(iy);

    auto yhisto    = xaxis_content_[find_bin_by_value_(bincx)];
    double content = yhisto->GetBinContent(yhisto->FindBin(bincy));

    //histo_->SetBinContent(ix, iy, content);
    // WORKAROUND to switch x and y axess
    histo_->SetBinContent(iy, ix, content);
  }}

  return *histo_;
}

void Histogram_2D::print(const char* dir){
  TCanvas can;
  for(int ix = 0; ix < xaxis_content_.size(); ix++){
    auto yhisto = xaxis_content_[ix]; 
    std::string name = std::string(dir)+"/"+yhisto->GetName()+".pdf";
    can.SaveAs(name.c_str(), "pdf");
  }
}

void Histogram_2D::reset(){
  for (int ix = 0; ix < xaxis_content_.size(); ix++){
    auto yhisto = xaxis_content_[ix].get();
    yhisto->Reset();
  }
}

void load_axis_into_vector(const TAxis* axis, std::vector<double>& vector){
  for(int i = 1; i <= axis->GetNbins() + 1; i++) vector.push_back(axis->GetBinLowEdge(i));
}

#endif