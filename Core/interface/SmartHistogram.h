/*! Definition of class SmartHistogram that allows to create ROOT-compatible histograms.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <deque>
#include <string>
#include <limits>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <mutex>

#include <TObject.h>
#include <TH1.h>
#include <TH2.h>
#include <TTree.h>
#include <TGraph.h>

#include "RootExt.h"
#include "TextIO.h"
#include "NumericPrimitives.h"
#include "PropertyConfigReader.h"

namespace root_ext {

class AbstractHistogram {
public:
    using Mutex = std::recursive_mutex;

    AbstractHistogram(const std::string& _name)
        : name(_name), outputDirectory(nullptr) {}
    AbstractHistogram(const AbstractHistogram& other) : name(other.name), outputDirectory(other.outputDirectory) {}
    virtual ~AbstractHistogram() {}

    virtual void WriteRootObject() = 0;
    virtual void SetOutputDirectory(TDirectory* directory) { outputDirectory = directory; }

    TDirectory* GetOutputDirectory() const { return outputDirectory; }
    const std::string& Name() const { return name; }
    virtual void SetName(const std::string& _name) { name = _name; }

    Mutex& GetMutex() { return mutex; }

private:
    std::string name;
    TDirectory* outputDirectory;
    Mutex mutex;
};

namespace detail {

template<typename ValueType>
class Base1DHistogram : public AbstractHistogram {
public:
    using const_iterator = typename std::deque<ValueType>::const_iterator;
    using RootContainer = TTree;

    Base1DHistogram(const std::string& name) : AbstractHistogram(name) {}

    const std::deque<ValueType>& Data() const { return data; }
    size_t size() const { return data.size(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }

    void Fill(const ValueType& value)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        data.push_back(value);
    }

    virtual void WriteRootObject()
    {
        std::lock_guard<Mutex> lock(GetMutex());
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

    void CopyContent(TTree& rootTree)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        data.clear();
        ValueType branch_value;
        TBranch* branch;
        rootTree.SetBranchAddress("values", &branch_value, &branch);
        Long64_t N = rootTree.GetEntries();
        for(Long64_t n = 0; n < N; ++n) {
            rootTree.GetEntry(n);
            data.push_back(branch_value);
        }
        rootTree.ResetBranchAddress(branch);
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
    using RootContainer = TTree;

    Base2DHistogram(const std::string& name) : AbstractHistogram(name) {}

    const std::deque<Value>& Data() const { return data; }
    size_t size() const { return data.size(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }

    void Fill(const NumberType& x, const NumberType& y)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        data.push_back(Value(x, y));
    }

    virtual void WriteRootObject()
    {
        std::lock_guard<Mutex> lock(GetMutex());
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

    void CopyContent(TTree& rootTree)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        data.clear();
        NumberType branch_value_x, branch_value_y;
        TBranch *branch_x, *branch_y;
        rootTree.SetBranchAddress("x", &branch_value_x, &branch_x);
        rootTree.SetBranchAddress("y", &branch_value_y, &branch_y);
        Long64_t N = rootTree.GetEntries();
        for(Long64_t n = 0; n < N; ++n) {
            rootTree.GetEntry(n);
            data.push_back(Value(branch_value_x, branch_value_y));
        }
        rootTree.ResetBranchAddress(branch_x);
        rootTree.ResetBranchAddress(branch_y);
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
    using RootContainer = TH1D;
    using Range = ::analysis::Range<double>;
    using MultiRange = ::analysis::MultiRange<Range>;

    SmartHistogram(const std::string& name, int nbins, double low, double high)
        : TH1D(name.c_str(), name.c_str(), nbins, low, high), AbstractHistogram(name), store(true),
          use_log_y(false), max_y_sf(1), divide_by_bin_width(false) {}

    SmartHistogram(const std::string& name, const std::vector<double>& bins)
        : TH1D(name.c_str(), name.c_str(), static_cast<int>(bins.size()) - 1, bins.data()), AbstractHistogram(name),
          store(true), use_log_y(false), max_y_sf(1), divide_by_bin_width(false) {}

    SmartHistogram(const std::string& name, int nbins, double low, double high, const std::string& x_axis_title,
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

    SmartHistogram(const std::string& name, const analysis::PropertyConfigReader::Item& p_config)
        : AbstractHistogram(name)
    {
        static constexpr int DefaultNumberOfBins = 100;
        static constexpr int DefaultBufferSize = 1000;
        try {
            TH1D::SetName(name.c_str());
            TH1D::SetTitle(name.c_str());
            if(p_config.Has("x_range")) {
                const auto x_range = p_config.Get<analysis::RangeWithStep<double>>("x_range");
                TH1D::SetBins(static_cast<int>(x_range.n_bins()), x_range.min(), x_range.max());
                divide_by_bin_width = false;
            } else if(p_config.Has("x_bins")){
                const std::vector<std::string> bin_values =
                        analysis::SplitValueList(p_config.Get<std::string>("x_bins"), false, ", \t", true);
                std::vector<double> bins;
                for(const auto& bin_str : bin_values)
                    bins.push_back(analysis::Parse<double>(bin_str));
                TH1D::SetBins(static_cast<int>(bins.size()) - 1, bins.data());
                divide_by_bin_width = true;
            } else{
                SetBins(DefaultNumberOfBins, 0, 0);
                GetXaxis()->SetCanExtend(true);
                SetBuffer(DefaultBufferSize);
            }

            std::string x_title, y_title;
            if(p_config.Read("x_title", x_title))
                SetXTitle(x_title.c_str());
            if(p_config.Read("y_title", y_title))
                SetYTitle(y_title.c_str());
            p_config.Read("log_x", use_log_x);
            p_config.Read("log_y", use_log_y);
            p_config.Read("max_y_sf", max_y_sf);
            p_config.Read("min_y_sf", min_y_sf);
            p_config.Read("div_bw", divide_by_bin_width);
            p_config.Read("blind_ranges", blind_ranges);
            if(p_config.Has("y_min"))
                y_min = p_config.Get<double>("y_min");
        } catch(analysis::exception& e) {
            throw analysis::exception("Invalid property set for histogram '%1%'. %2%") % Name() % e.message();
        }
    }

    virtual void SetName(const char* _name) override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        TH1D::SetName(_name);
        AbstractHistogram::SetName(_name);
    }

    virtual void SetName(const std::string& _name) override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        TH1D::SetName(_name.c_str());
        AbstractHistogram::SetName(_name);
    }

    virtual void WriteRootObject() override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        if(store && GetOutputDirectory())
            root_ext::WriteObject(*this);
    }

    virtual void SetOutputDirectory(TDirectory* directory) override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        TDirectory* dir = store ? directory : nullptr;
        AbstractHistogram::SetOutputDirectory(dir);
        SetDirectory(dir);
    }

    bool UseLogX() const { return use_log_x; }
    bool UseLogY() const { return use_log_y; }
    double MaxYDrawScaleFactor() const { return max_y_sf; }
    double MinYDrawScaleFactor() const { return min_y_sf; }
    std::string GetXTitle() const { return GetXaxis()->GetTitle(); }
    std::string GetYTitle() const { return GetYaxis()->GetTitle(); }
    bool NeedToDivideByBinWidth() const { return divide_by_bin_width; }
    const std::string& GetLegendTitle() const { return legend_title; }
    const MultiRange GetBlindRanges() const { return blind_ranges; }

    void SetLegendTitle(const std::string& _legend_title)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        legend_title = _legend_title;
    }

    bool TryGetMinY(double& _y_min) const
    {
        if(!y_min) return false;
        _y_min = *y_min;
        return true;
    }

    double GetSystematicUncertainty() const { return syst_unc; }
    void SetSystematicUncertainty(double _syst_unc) { syst_unc = _syst_unc; }
    double GetPostfitScaleFactor() const { return postfit_sf; }
    void SetPostfitScaleFactor(double _postfit_sf)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        postfit_sf = _postfit_sf;
    }

    void CopyContent(const TH1& other)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        if(other.GetNbinsX() != GetNbinsX())
            throw analysis::exception("Unable to copy histogram content: source and destination have different number"
                                      " of bins.");
        for(Int_t n = 0; n <= other.GetNbinsX() + 1; ++n) {
            if(GetBinLowEdge(n) != other.GetBinLowEdge(n) || GetBinWidth(n) != other.GetBinWidth(n))
                throw analysis::exception("Unable to copy histogram content from histogram '%1%' into '%2%':"
                    " bin %3% is not compatible between the source and destination."
                    " (LowEdge, Width): (%4%, %5%) != (%6%, %7%).")
                    % other.GetName() % Name() % n % other.GetBinLowEdge(n) % other.GetBinWidth(n) % GetBinLowEdge(n)
                    % GetBinWidth(n);
            SetBinContent(n, other.GetBinContent(n));
            SetBinError(n, other.GetBinError(n));
        }
    }

    void AddHistogram(const SmartHistogram<TH1D>& other)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        const double integral = Integral(), other_integral = other.Integral(), tot_integral = integral + other_integral;
        const double post_integral = postfit_sf * integral, other_post_integral = other.postfit_sf * other_integral,
                     tot_post_integral = post_integral + other_post_integral;
        if(tot_integral != 0) {
            postfit_sf = tot_post_integral / tot_integral;
            syst_unc = std::hypot(syst_unc * post_integral, other.syst_unc * other_post_integral) / tot_post_integral;
        }
        Add(&other, 1);
    }

private:
    bool store{true};
    bool use_log_x{false}, use_log_y{false};
    double max_y_sf{1}, min_y_sf{1};
    boost::optional<double> y_min;
    bool divide_by_bin_width{false};
    std::string legend_title;
    MultiRange blind_ranges;
    double syst_unc{0}, postfit_sf{1};
};

template<>
class SmartHistogram<TH2D> : public TH2D, public AbstractHistogram {
public:
    using RootContainer = TH2D;

    SmartHistogram(const std::string& name,
                   int nbinsx, double xlow, double xup,
                   int nbinsy, double ylow, double yup)
        : TH2D(name.c_str(), name.c_str(), nbinsx, xlow, xup, nbinsy, ylow, yup),
          AbstractHistogram(name), store(true), use_log_y(false), max_y_sf(1) {}

    SmartHistogram(const std::string& name, int nbinsx, double xlow, double xup, int nbinsy, double ylow,
                   double yup, const std::string& x_axis_title, const std::string& y_axis_title, bool _use_log_y,
                   double _max_y_sf, bool _store)
        : TH2D(name.c_str(), name.c_str(), nbinsx, xlow, xup, nbinsy, ylow, yup),
          AbstractHistogram(name), store(_store), use_log_y(_use_log_y), max_y_sf(_max_y_sf)
    {
        SetXTitle(x_axis_title.c_str());
        SetYTitle(y_axis_title.c_str());
    }

    SmartHistogram(const std::string& name, const std::vector<double>& binsx, const std::vector<double>& binsy)
        : TH2D(name.c_str(), name.c_str(), static_cast<int>(binsx.size()) - 1, binsx.data(),
               static_cast<int>(binsy.size()) - 1, binsy.data()), AbstractHistogram(name),
          store(true), use_log_y(false), max_y_sf(1) {}

    virtual void WriteRootObject() override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        if(store && GetOutputDirectory())
            root_ext::WriteObject(*this);
    }

    virtual void SetName(const char* _name) override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        TH2D::SetName(_name);
        AbstractHistogram::SetName(_name);
    }

    virtual void SetName(const std::string& _name) override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        TH2D::SetName(_name.c_str());
        AbstractHistogram::SetName(_name);
    }

    virtual void SetOutputDirectory(TDirectory* directory) override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        TDirectory* dir = store ? directory : nullptr;
        AbstractHistogram::SetOutputDirectory(dir);
        SetDirectory(dir);
    }

    bool UseLogY() const { return use_log_y; }
    double MaxYDrawScaleFactor() const { return max_y_sf; }
    std::string GetXTitle() const { return GetXaxis()->GetTitle(); }
    std::string GetYTitle() const { return GetYaxis()->GetTitle(); }

    void CopyContent(const TH2D& other)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        if(other.GetNbinsX() != GetNbinsX() || other.GetNbinsY() != GetNbinsY())
            throw analysis::exception("Unable to copy histogram content: source and destination have different number"
                                      " of bins.");
        for(Int_t n = 0; n <= GetNbinsX() + 1; ++n) {
            for(Int_t k = 0; k <= GetNbinsY() + 1; ++k) {
            if(GetXaxis()->GetBinLowEdge(n) != other.GetXaxis()->GetBinLowEdge(n)
                    || GetXaxis()->GetBinWidth(n) != other.GetXaxis()->GetBinWidth(n)
                    || GetYaxis()->GetBinLowEdge(k) != other.GetYaxis()->GetBinLowEdge(k)
                    || GetYaxis()->GetBinWidth(k) != other.GetYaxis()->GetBinWidth(k))
                throw analysis::exception("Unable to copy histogram content: bin (%1%, %2% is not compatible between"
                                          " the source and destination.") % n % k;
            SetBinContent(n, k, other.GetBinContent(n, k));
            SetBinError(n, k, other.GetBinError(n, k));
            }
        }
    }

private:
    bool store;
    bool use_log_y;
    double max_y_sf;
};

template<>
class SmartHistogram<TGraph> : public AbstractHistogram {
public:
    using DataVector = std::vector<double>;
    using RootContainer = TGraph;
    using AbstractHistogram::AbstractHistogram;

    void AddPoint(double x, double y)
    {
        std::lock_guard<Mutex> lock(GetMutex());
        x_vector.push_back(x);
        y_vector.push_back(y);
    }

    const DataVector& GetXvalues() const { return x_vector; }
    const DataVector& GetYvalues() const { return y_vector; }

    virtual void WriteRootObject() override
    {
        std::lock_guard<Mutex> lock(GetMutex());
        std::unique_ptr<TGraph> graph(new TGraph(static_cast<int>(x_vector.size()), x_vector.data(), y_vector.data()));
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
    using HistogramParameters = analysis::PropertyConfigReader::Item;
    using HistogramParametersMap = analysis::PropertyConfigReader::ItemCollection;

    static HistogramParametersMap& ParametersMap_RW()
    {
        static const auto parameters = std::make_unique<HistogramParametersMap>();
        return *parameters;
    }

    static const HistogramParametersMap& ParametersMap()
    {
        return ParametersMap_RW();
    }

    static std::string& ConfigName()
    {
        static const auto configName = std::make_unique<std::string>("histograms.cfg");
        return *configName;
    }

    static const HistogramParameters& GetParameters(const std::string& name, const std::string& selection_label = "")
    {
        static const HistogramParameters default_params;
        const HistogramParametersMap& parameters = ParametersMap();
        std::string best_name = name;
        if(!selection_label.empty())
        {
            const std::string sl_plus_name = selection_label + "/" + name;
            if(parameters.count(sl_plus_name))
                best_name = sl_plus_name;
        }

        if(!parameters.count(best_name)) {
            std::cerr << "Not found default parameters for histogram '" << name
                      << "'. Please, define it in '" << ConfigName() << "'." << std::endl;
            return default_params;
        }
        return parameters.at(best_name);
    }

public:
    static void LoadConfig(const std::string& config_path)
    {
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        ConfigName() = config_path;
        HistogramParametersMap& parameters = ParametersMap_RW();

        analysis::PropertyConfigReader reader;
        reader.Parse(config_path);
        parameters = reader.GetItems();
    }

    static SmartHistogram<TH1D>* Make(const std::string& name, const std::string& selection_label)
    {
        const HistogramParameters& params = GetParameters(name, selection_label);
        return new SmartHistogram<TH1D>(name, params);
    }
};

} // root_ext
