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
    Histogram_2D(const char* name, std::vector<double> yaxis, const double xmin, const double xmax);
    Histogram_2D(Histogram_2D& histo) = delete;
    ~Histogram_2D();

    void th2d_add (const TH2D& histo);
    void divide   (const Histogram_2D& histo);
    void reset();

    void add_x_binning_by_index(const int index, const std::vector<double> xaxis);

    bool can_be_imported(const TH2D& histo);
    void print(const char* dir);

    //use this function only for weights, not for counts
    std::shared_ptr<TH2D> get_weights_th2d(const char* name, const char* title);

  private:
    int find_bin_by_value_(const double& y);

    std::string name_;

    std::vector<double>                 yaxis_;
    std::vector<std::shared_ptr<TH1D>>  yaxis_content_;
    std::vector<bool> occupancy_;


    double xmin_ = std::numeric_limits<float>::max();
    double xmax_ = std::numeric_limits<float>::lowest();
};

Histogram_2D::~Histogram_2D(){
}

Histogram_2D::Histogram_2D(const char* name, std::vector<double> yaxis, const double xmin, const double xmax){
  yaxis_ = yaxis;
  for (std::vector<double>::iterator it = yaxis.begin(); it != std::prev(yaxis.end()); it++){
    yaxis_content_.push_back(std::make_shared<TH1D>());
    occupancy_.push_back(false);
  }
  xmin_ = xmin;
  xmax_ = xmax;
  name_ = name;
}

void Histogram_2D::add_x_binning_by_index(const int index, const std::vector<double> xaxis){
  std::string name = name_+std::to_string(index);
  if(index >= yaxis_content_.size()){
    std::runtime_error("Index "+std::to_string(index)+" out of y-axis range");
  }
  if (xaxis.front() != xmax_ || xaxis.back() != xmin_){
    std::runtime_error("Input yaxis min or max values not matching specified min and max values");
  }
  yaxis_content_[index] = std::shared_ptr<TH1D>(new TH1D(name.c_str(), "", xaxis.size()-1, &xaxis[0]));
  occupancy_[index] = true;
}

bool Histogram_2D::can_be_imported(const TH2D& histo){
  auto check_axis = [](const std::vector<double>& input_axis, const std::vector<double>& this_axis){
    //if (input_axis.front() != this_axis.front() || input_axis.back() != this_axis.back()) return false;
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
  std::vector<double> this_xaxis;

  load_axis_into_vector(histo.GetXaxis(), input_xaxis);
  load_axis_into_vector(histo.GetYaxis(), input_yaxis);

  if (!check_axis(input_yaxis, yaxis_)){
    std::cerr << "Invalid y-axis binning found" << std::endl;
    return false;
  }

  for(int iy = 0; iy < yaxis_.size()-1; iy++){
    auto this_histo = yaxis_content_[iy];
    load_axis_into_vector(this_histo->GetXaxis(), this_xaxis);
    if(!check_axis(input_xaxis, this_xaxis)){
      std::cerr << "Invalid x axis binning found for y bin n. " << iy << std::endl;
      return false;
    }
    this_xaxis.clear();
  }
  return true;
}

int Histogram_2D::find_bin_by_value_(const double& y){
  for(int iy = 0; iy < yaxis_.size() - 1; iy++){
    if(y >= yaxis_[iy] && y < yaxis_[iy+1]) return iy;
  }
  throw std::range_error("Value "+std::to_string(y)+" is not in the range of the y axis");
}

void Histogram_2D::th2d_add (const TH2D& histo){
  if(!can_be_imported(histo)){
    throw std::invalid_argument("Given TH2D "+std::string(histo.GetName())+" can not be imported");
  }

  auto input_yaxis = histo.GetYaxis();
  auto input_xaxis = histo.GetXaxis();

  for (int iy = 1; iy <= input_yaxis->GetNbins(); iy++){
  for (int ix = 1; ix <= input_xaxis->GetNbins(); ix++){
    auto bincy = input_yaxis->GetBinCenter(iy);
    auto bincx = input_xaxis->GetBinCenter(ix);

    if(bincx < xmin_ || bincx >= xmax_ || bincy < yaxis_.front() || bincy >= yaxis_.back()) continue;

    auto xhisto = yaxis_content_[find_bin_by_value_(bincy)];
    xhisto->SetBinContent(
      xhisto->FindBin(bincx),
      xhisto->GetBinContent(xhisto->FindBin(bincx)) + histo.GetBinContent(ix, iy));
  }}

  for (auto xhisto : yaxis_content_){
    if (!xhisto->GetSumw2N()){
      xhisto->Sumw2();
    }
  }
}

void Histogram_2D::divide(const Histogram_2D& histo){
  auto check_axis = [] (const std::vector<double>& axis1, const std::vector<double>& axis2){
    return axis1 == axis2;
  };

  if (!std::all_of(occupancy_.begin(), occupancy_.end(), [](bool i){return i;})){
    throw std::logic_error("Not all the bins have been initialized");
  }
  if (!check_axis(yaxis_, histo.yaxis_)){
    throw std::logic_error("Invalid x binning detected on denominator for Histogram_2D "+histo.name_);
  }
  
  std::vector<double> thisxaxis;
  std::vector<double> xaxis;
  for (int iy = 0; iy < yaxis_content_.size(); iy++){
    TH1D* thisxhisto = yaxis_content_[iy].get();
    TH1D* xhisto     = histo.yaxis_content_[iy].get();
    load_axis_into_vector(thisxhisto->GetXaxis(), thisxaxis);
    load_axis_into_vector(xhisto->GetXaxis()    , xaxis    );

    if(!check_axis(thisxaxis, xaxis)){
      throw std::logic_error("Invalid x-axis binning found for denominator in y bin n. "+std::to_string(iy)+" for Histogram_2D "+histo.name_);
    }

    (*thisxhisto).Divide(xhisto);
  }
}

std::shared_ptr<TH2D> Histogram_2D::get_weights_th2d(const char* name, const char* title){
  double xwidthmin = std::numeric_limits<float>::max();
  double ywidthmin = std::numeric_limits<float>::max();

  auto get_min_width = [](std::vector<double>& vector, double& min){
    for (int i = 0; i < vector.size()-1; i++) min = std::min(min, vector[i+1] - vector[i]);
  };

  get_min_width(yaxis_, ywidthmin);

  std::vector<double> xaxis;
  for (int iy = 0; iy < yaxis_content_.size(); iy++){
    auto xhisto = yaxis_content_[iy];
    load_axis_into_vector(xhisto->GetXaxis(), xaxis);
    get_min_width(xaxis, xwidthmin);
    xaxis.clear();
  }

  int ny = (yaxis_.back() - yaxis_.front()) / ywidthmin;
  int nx = (xmax_ - xmin_) / xwidthmin;

  auto histo_ = std::make_shared<TH2D>(name, title, nx, xmin_, xmax_, ny, yaxis_.front(), yaxis_.back());

  for(int iy = 1; iy <= ny; iy++){
  for(int ix = 1; ix <= nx; ix++){
    double bincx = histo_->GetXaxis()->GetBinCenter(ix);
    double bincy = histo_->GetYaxis()->GetBinCenter(iy);

    auto xhisto    = yaxis_content_[find_bin_by_value_(bincy)];
    double content = xhisto->GetBinContent(xhisto->FindBin(bincx));
    double error   = xhisto->GetBinError(xhisto->FindBin(bincx));

    histo_->SetBinContent(ix, iy, content);
    histo_->SetBinError  (ix, iy, error  );
  }}

  return histo_;
}

void Histogram_2D::print(const char* dir){
  TCanvas can;
  for(int iy = 0; iy < yaxis_content_.size(); iy++){
    auto xhisto = yaxis_content_[iy];
    std::string name = std::string(dir)+"/"+xhisto->GetName()+".pdf";
    xhisto->Draw("HIST");
    can.SaveAs(name.c_str(), "pdf");
  }
}

void Histogram_2D::reset(){
  for (int iy = 0; iy < yaxis_content_.size(); iy++){
    auto xhisto = yaxis_content_[iy].get();
    xhisto->Reset();
  }
}

void load_axis_into_vector(const TAxis* axis, std::vector<double>& vector){
  for(int i = 1; i <= axis->GetNbins() + 1; i++) vector.push_back(axis->GetBinLowEdge(i));
}

#endif