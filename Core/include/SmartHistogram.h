/*! Definition of class SmartHistogram that allows to create ROOT-compatible histograms.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <deque>
#include <string>
#include <limits>
#include <stdexcept>
#include <memory>
#include <fstream>

#include <TObject.h>
#include <TH1.h>
#include <TH2.h>
#include <TTree.h>
#include <TGraph.h>

#include "RootExt.h"

namespace root_ext {

class AbstractHistogram {
public:
    AbstractHistogram(const std::string& _name)
        : name(_name), outputDirectory(nullptr) {}

    virtual ~AbstractHistogram() {}

    virtual void WriteRootObject() = 0;
    virtual void SetOutputDirectory(TDirectory* directory) { outputDirectory = directory; }

    TDirectory* GetOutputDirectory() const { return outputDirectory; }
    const std::string& Name() const { return name; }

private:
    std::string name;
    TDirectory* outputDirectory;
};

namespace detail {

template<typename ValueType>
class Base1DHistogram : public AbstractHistogram {
public:
    using const_iterator = typename std::deque<ValueType>::const_iterator;

    Base1DHistogram(const std::string& name) : AbstractHistogram(name) {}

    const std::deque<ValueType>& Data() const { return data; }
    const size_t size() const { return data.size(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }

    void Fill(const ValueType& value)
    {
        data.push_back(value);
    }

    virtual void WriteRootObject()
    {
        if(!GetOutputDirectory()) return;
        std::unique_ptr<TTree> rootTree(new TTree(Name().c_str(), Name().c_str()));
        rootTree->SetDirectory(GetOutputDirectory());
        ValueType branch_value;
        rootTree->Branch("values", &branch_value);
        for(const ValueType& value : data) {
            branch_value = value;
            rootTree->Fill();
        }
        root_ext::WriteObject(*rootTree);
    }

private:
    std::deque<ValueType> data;
};

template<typename NumberType>
class Base2DHistogram : public AbstractHistogram {
public:
    struct Value {
        NumberType x, y;
        Value() {}
        Value(NumberType _x, NumberType _y) : x(_x), y(_y) {}
    };

    using const_iterator = typename std::deque<Value>::const_iterator;

    Base2DHistogram(const std::string& name) : AbstractHistogram(name) {}

    const std::deque<Value>& Data() const { return data; }
    const size_t size() const { return data.size(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }

    void Fill(const NumberType& x, const NumberType& y)
    {
        data.push_back(Value(x, y));
    }

    virtual void WriteRootObject()
    {
        if(!GetOutputDirectory()) return;
        std::unique_ptr<TTree> rootTree(new TTree(Name().c_str(), Name().c_str()));
        rootTree->SetDirectory(GetOutputDirectory());
        NumberType branch_value_x, branch_value_y;
        rootTree->Branch("x", &branch_value_x);
        rootTree->Branch("y", &branch_value_y);
        for(const Value& value : data) {
            branch_value_x = value.x;
            branch_value_y = value.y;
            rootTree->Fill();
        }
        root_ext::WriteObject(*rootTree);
    }

private:
    std::deque<Value> data;
};

} // namespace detail

template<typename ValueType>
class SmartHistogram;

template<>
class SmartHistogram<double> : public detail::Base1DHistogram<double> {
public:
    SmartHistogram(const std::string& name) : Base1DHistogram<double>(name) {}
};

template<>
class SmartHistogram<float> : public detail::Base1DHistogram<float> {
public:
    SmartHistogram(const std::string& name) : Base1DHistogram<float>(name) {}
};

template<>
class SmartHistogram<int> : public detail::Base1DHistogram<int> {
public:
    SmartHistogram(const std::string& name) : Base1DHistogram<int>(name) {}
};

template<>
class SmartHistogram<unsigned> : public detail::Base1DHistogram<unsigned> {
public:
    SmartHistogram(const std::string& name) : Base1DHistogram<unsigned>(name) {}
};

template<>
class SmartHistogram<bool> : public detail::Base1DHistogram<bool> {
public:
    SmartHistogram(const std::string& name) : Base1DHistogram<bool>(name) {}
};

template<>
class SmartHistogram< detail::Base2DHistogram<double>::Value > : public detail::Base2DHistogram<double> {
public:
    SmartHistogram(const std::string& name) : Base2DHistogram<double>(name) {}
};

template<>
class SmartHistogram< detail::Base2DHistogram<float>::Value > : public detail::Base2DHistogram<float> {
public:
    SmartHistogram(const std::string& name) : Base2DHistogram<float>(name) {}
};

template<>
class SmartHistogram< detail::Base2DHistogram<int>::Value > : public detail::Base2DHistogram<int> {
public:
    SmartHistogram(const std::string& name) : Base2DHistogram<int>(name) {}
};

template<>
class SmartHistogram< detail::Base2DHistogram<bool>::Value > : public detail::Base2DHistogram<bool> {
public:
    SmartHistogram(const std::string& name) : Base2DHistogram<bool>(name) {}
};

template<>
class SmartHistogram<TH1D> : public TH1D, public AbstractHistogram {
public:
    SmartHistogram(const std::string& name, size_t nbins, double low, double high)
        : TH1D(name.c_str(), name.c_str(), nbins, low, high), AbstractHistogram(name), store(true), use_log_y(false),
          max_y_sf(1), divide_by_bin_width(false) {}

    SmartHistogram(const std::string& name, const std::vector<double>& bins)
        : TH1D(name.c_str(), name.c_str(), static_cast<int>(bins.size()) - 1, bins.data()), AbstractHistogram(name),
          store(true), use_log_y(false), max_y_sf(1), divide_by_bin_width(false) {}

    SmartHistogram(const std::string& name, size_t nbins, double low, double high, const std::string& x_axis_title,
                   const std::string& y_axis_title, bool _use_log_y, double _max_y_sf, bool _divide_by_bin_width,
                   bool _store)
        : TH1D(name.c_str(), name.c_str(), nbins, low, high), AbstractHistogram(name), store(_store),
          use_log_y(_use_log_y), max_y_sf(_max_y_sf), divide_by_bin_width(_divide_by_bin_width)
    {
        SetXTitle(x_axis_title.c_str());
        SetYTitle(y_axis_title.c_str());
    }

    SmartHistogram(const std::string& name, const std::vector<double>& bins, const std::string& x_axis_title,
                   const std::string& y_axis_title, bool _use_log_y, double _max_y_sf, bool _divide_by_bin_width,
                   bool _store)
        : TH1D(name.c_str(), name.c_str(), static_cast<int>(bins.size()) - 1, bins.data()), AbstractHistogram(name),
          store(_store), use_log_y(_use_log_y), max_y_sf(_max_y_sf), divide_by_bin_width(_divide_by_bin_width)
    {
        SetXTitle(x_axis_title.c_str());
        SetYTitle(y_axis_title.c_str());
    }

    SmartHistogram(const TH1D& other, bool _use_log_y, double _max_y_sf, bool _divide_by_bin_width)
        : TH1D(other), AbstractHistogram(other.GetName()), store(false), use_log_y(_use_log_y), max_y_sf(_max_y_sf),
          divide_by_bin_width(_divide_by_bin_width) {}

    virtual void WriteRootObject() override
    {
        if(store && GetOutputDirectory())
            root_ext::WriteObject(*this);
    }

    virtual void SetOutputDirectory(TDirectory* directory) override
    {
        TDirectory* dir = store ? directory : nullptr;
        AbstractHistogram::SetOutputDirectory(dir);
        SetDirectory(dir);
    }

    bool UseLogY() const { return use_log_y; }
    double MaxYDrawScaleFactor() const { return max_y_sf; }
    std::string GetXTitle() const { return GetXaxis()->GetTitle(); }
    std::string GetYTitle() const { return GetYaxis()->GetTitle(); }
    bool NeedToDivideByBinWidth() const { return divide_by_bin_width; }
    void SetLegendTitle(const std::string _legend_title) { legend_title = _legend_title; }
    const std::string& GetLegendTitle() const { return legend_title; }

    void CopyContent(const TH1D& other)
    {
        if(other.GetNbinsX() != GetNbinsX())
            throw analysis::exception("Unable to copy histogram content: source and destination have different number"
                                      " of bins.");
        for(Int_t n = 0; n <= other.GetNbinsX() + 1; ++n) {
            if(GetBinLowEdge(n) != other.GetBinLowEdge(n) || GetBinWidth(n) != other.GetBinWidth(n))
                throw analysis::exception("Unable to copy histogram content: bin %1% is not compatible between the"
                                          " source and destination.") % n;
            SetBinContent(n, other.GetBinContent(n));
            SetBinError(n, other.GetBinError(n));
        }
    }

private:
    bool store;
    bool use_log_y;
    double max_y_sf;
    bool divide_by_bin_width;
    std::string legend_title;
};

template<>
class SmartHistogram<TH2D> : public TH2D, public AbstractHistogram {
public:
    SmartHistogram(const std::string& name,
                   size_t nbinsx, double xlow, double xup,
                   size_t nbinsy, double ylow, double yup)
        : TH2D(name.c_str(), name.c_str(), nbinsx, xlow, xup, nbinsy, ylow, yup), AbstractHistogram(name), store(true),
          use_log_y(false), max_y_sf(1) {}

    SmartHistogram(const std::string& name, size_t nbinsx, double xlow, double xup, size_t nbinsy, double ylow,
                   double yup, const std::string& x_axis_title, const std::string& y_axis_title, bool _use_log_y,
                   double _max_y_sf, bool _store)
        : TH2D(name.c_str(), name.c_str(), nbinsx, xlow, xup, nbinsy, ylow, yup), AbstractHistogram(name),
          store(_store), use_log_y(_use_log_y), max_y_sf(_max_y_sf)
    {
        SetXTitle(x_axis_title.c_str());
        SetYTitle(y_axis_title.c_str());
    }

    virtual void WriteRootObject() override
    {
        if(store && GetOutputDirectory())
            root_ext::WriteObject(*this);
    }

    virtual void SetOutputDirectory(TDirectory* directory) override
    {
        TDirectory* dir = store ? directory : nullptr;
        AbstractHistogram::SetOutputDirectory(dir);
        SetDirectory(dir);
    }

    bool UseLogY() const { return use_log_y; }
    double MaxYDrawScaleFactor() const { return max_y_sf; }
    std::string GetXTitle() const { return GetXaxis()->GetTitle(); }
    std::string GetYTitle() const { return GetYaxis()->GetTitle(); }

private:
    bool store;
    bool use_log_y;
    double max_y_sf;
};

template<>
class SmartHistogram<TGraph> : public AbstractHistogram {
public:
    using DataVector = std::vector<double>;
    using AbstractHistogram::AbstractHistogram;

    void AddPoint(double x, double y)
    {
        x_vector.push_back(x);
        y_vector.push_back(y);
    }

    const DataVector& GetXvalues() const { return x_vector; }
    const DataVector& GetYvalues() const { return y_vector; }

    virtual void WriteRootObject() override
    {
        std::unique_ptr<TGraph> graph(new TGraph(x_vector.size(), x_vector.data(), y_vector.data()));
        if(GetOutputDirectory())
            root_ext::WriteObject(*graph, GetOutputDirectory(), Name());
    }

private:
    DataVector x_vector, y_vector;
};


template<typename ValueType>
struct HistogramFactory {
    template<typename ...Args>
    static SmartHistogram<ValueType>* Make(const std::string& name, Args... args)
    {
        return new SmartHistogram<ValueType>(name, args...);
    }
};

template<>
struct HistogramFactory<TH1D> {
private:
    struct HistogramParameters {
        size_t nbins;
        double low;
        double high;
    };

    using HistogramParametersMap = std::map<std::string, HistogramParameters>;

    static const HistogramParameters& GetParameters(const std::string& name)
    {
        static const std::string configName = "Analysis/config/histograms.cfg";
        static bool parametersLoaded = false;
        static HistogramParametersMap parameters;
        if(!parametersLoaded) {
            std::ifstream cfg(configName);
            while (cfg.good()) {
                std::string cfgLine;
                std::getline(cfg,cfgLine);
                if (!cfgLine.size() || cfgLine.at(0) == '#') continue;
                std::istringstream ss(cfgLine);
                std::string param_name;
                HistogramParameters param;
                ss >> param_name;
                ss >> param.nbins;
                ss >> param.low;
                ss >> param.high;
                if(parameters.count(param_name)) {
                    std::ostringstream ss_error;
                    ss_error << "Redefinition of default parameters for histogram '" << param_name << "'.";
                    throw std::runtime_error(ss_error.str());
                }
                parameters[param_name] = param;
              }
            parametersLoaded = true;
        }
        std::string best_name = name;
        for(size_t pos = name.find_last_of('_'); !parameters.count(best_name) && pos != 0 && pos != std::string::npos;
            pos = name.find_last_of('_', pos - 1))
            best_name = name.substr(0, pos);

        if(!parameters.count(best_name)) {
            std::ostringstream ss_error;
            ss_error << "Not found default parameters for histogram '" << name;
            if(best_name != name)
                ss_error << "' or '" << best_name;
            ss_error << "'. Please, define it in '" << configName << "'.";
            throw std::runtime_error(ss_error.str());
        }
        return parameters.at(best_name);
    }

public:
    template<typename ...Args>
    static SmartHistogram<TH1D>* Make(const std::string& name, Args... args)
    {
        return new SmartHistogram<TH1D>(name, args...);
    }

    static SmartHistogram<TH1D>* Make(const std::string& name)
    {
        const HistogramParameters& params = GetParameters(name);
        return new SmartHistogram<TH1D>(name, params.nbins, params.low, params.high);
    }
};

} // root_ext
