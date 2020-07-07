/*! Common tools and definitions to apply cuts.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <iostream>

#include <TH1D.h>
#include <Rtypes.h>

#include "SmartHistogram.h"

namespace cuts {

class cut_failed : public std::exception {
public:
    cut_failed(size_t parameter_id) noexcept
        : _param_id(parameter_id)
    {
        std::ostringstream ss;
        ss << "Cut requirements are not fulfilled for parameter id = " << _param_id << ".";
        message = ss.str();
    }

    ~cut_failed() noexcept {}

    virtual const char* what() const noexcept { return message.c_str(); }
    size_t param_id() const noexcept { return _param_id; }

private:
    size_t _param_id;
    std::string message;
};

template<typename ValueType, typename Histogram>
ValueType fill_histogram(ValueType value, Histogram& histogram, double weight)
{
    histogram.Fill(value,weight);
    return value;
}


class ObjectSelector{
public:

    virtual ~ObjectSelector(){}

    void incrementCounter(size_t param_id, const std::string& param_label)
    {
        if (counters.size() < param_id)
            throw std::runtime_error("counters out of range");
        if (counters.size() == param_id){  //counters and selections filled at least once
            counters.push_back(0);
            selections.push_back(0);
            selectionsSquaredErros.push_back(0);
            const std::string label = make_unique_label(param_label);
            labels.push_back(label);
            label_set.insert(label);
        }
        counters.at(param_id)++;
    }

    void fill_selection(double weight = 1.0){
        for (unsigned n = 0; n < counters.size(); ++n){
            if(counters.at(n) > 0) {
                selections.at(n) += weight;
                selectionsSquaredErros.at(n) += weight * weight;
            }
            counters.at(n) = 0;
        }
    }

    template<typename ObjectType, typename Selector, typename Comparitor>
    std::vector<ObjectType> collect_objects(double weight, size_t n_objects, const Selector& selector,
                                            const Comparitor& comparitor)
    {
        std::vector<ObjectType> selected;
        for (size_t n = 0; n < n_objects; ++n) {
            try {
                const ObjectType selectedCandidate = selector(n);
                selected.push_back(selectedCandidate);
            } catch(cuts::cut_failed&) {}
        }

        fill_selection(weight);
        std::sort(selected.begin(), selected.end(), comparitor);

        return selected;
    }

private:
    std::string make_unique_label(const std::string& label)
    {
        if(!label_set.count(label)) return label;
        for(size_t n = 2; ; ++n) {
            std::ostringstream ss;
            ss << label << "_" << n;
            if(!label_set.count(ss.str())) return ss.str();
        }
    }

protected:
    std::vector<unsigned> counters;
    std::vector<double> selections;
    std::vector<double> selectionsSquaredErros;
    std::vector<std::string> labels;
    std::set<std::string> label_set;
};

namespace detail {
struct DefaultSelectionManager{
    template<typename ValueType>
    void FillHistogram(ValueType value, const std::string& histogram_name) {}
};
} // namespace detail

template<typename SelectionManager = detail::DefaultSelectionManager>
class Cutter {
public:
    explicit Cutter(ObjectSelector* _objectSelector, SelectionManager* _selectionManager = nullptr)
        : objectSelector(_objectSelector), selectionManager(_selectionManager), param_id(0) {}

    bool Enabled() const { return objectSelector != nullptr; }
    int CurrentParamId() const { return param_id; }

    void operator()(bool expected, const std::string& label)
    {
        (*this)(expected, label, expected);
    }

    template<typename ValueType>
    void operator()(bool expected, const std::string& label, const ValueType& value)
    {
        if(selectionManager) {
            try {
                selectionManager->FillHistogram(value, label);
            }catch(std::exception& e) {
                std::cout << "ERROR: " << e.what() << std::endl;
            }
        }
        if(Enabled()) {
            ++param_id;
            if(!expected)
                throw cut_failed(param_id -1);
            objectSelector->incrementCounter(param_id - 1, label);
        }
    }

    bool test(bool expected, const std::string& label)
    {
        try {
            (*this)(expected, label);
            return true;
        } catch(cut_failed&) {}
        return false;
    }

private:
    ObjectSelector* objectSelector;
    SelectionManager* selectionManager;
    size_t param_id;
};

} // cuts

namespace root_ext {

template<>
class SmartHistogram<cuts::ObjectSelector> : public cuts::ObjectSelector, public AbstractHistogram {
public:
    using RootContainer = TH1D;

    SmartHistogram(const std::string& name) : AbstractHistogram(name) {}

    void SetSave(bool _save)
    {
        save = _save;
    }

    virtual void WriteRootObject()
    {
        if(!save || !selections.size() || !GetOutputDirectory() )
            return;
        std::unique_ptr<TH1D> selection_histogram(
                    new TH1D(Name().c_str(), Name().c_str(),selections.size(),-0.5,-0.5+selections.size()));
        for (unsigned n = 0; n < selections.size(); ++n){
            const std::string label = labels.at(n);
            selection_histogram->GetXaxis()->SetBinLabel(n+1, label.c_str());
            selection_histogram->SetBinContent(n+1,selections.at(n));
            selection_histogram->SetBinError(n+1,std::sqrt(selectionsSquaredErros.at(n)));
        }
        root_ext::WriteObject(*selection_histogram, GetOutputDirectory());
  }

private:
    bool save{true};
};
}
