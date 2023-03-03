/*! Common CERN ROOT extensions.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <memory>
#include <map>

#include <TROOT.h>
#include <TClass.h>
#include <TLorentzVector.h>
#include <TMatrixD.h>
#include <TFile.h>
#include <Compression.h>
#include <TH2.h>

#include "exception.h"

namespace root_ext {

std::shared_ptr<TFile> inline CreateRootFile(const std::string& file_name, ROOT::ECompressionAlgorithm compression = ROOT::kZLIB,
                                             int compression_level = 9){
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "RECREATE", "", compression * 100 + compression_level));
    if(!file || file->IsZombie())
        throw analysis::exception("File '%1%' not created.") % file_name;
    return file;
}

std::shared_ptr<TFile> inline OpenRootFile(const std::string& file_name){
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "READ"));
    if(!file || file->IsZombie())
        throw analysis::exception("File '%1%' not opened.") % file_name;
    return file;
}


void WriteObject(const TObject& object, TDirectory* dir, const std::string& name = "");

template<typename Object>
void WriteObject(const Object& object)
{
    TDirectory* dir = object.GetDirectory();
    WriteObject(object, dir);
}


template<typename Object>
Object* ReadObject(TDirectory& file, const std::string& name)
{
    if(!name.size())
        throw analysis::exception("Can't read nameless object.");
    TObject* root_object = file.Get(name.c_str());
    if(!root_object)
        throw analysis::exception("Object '%1%' not found in '%2%'.") % name % file.GetName();
    Object* object = dynamic_cast<Object*>(root_object);
    if(!object)
        throw analysis::exception("Wrong object type '%1%' for object '%2%' in '%3%'.") % typeid(Object).name()
            % name % file.GetName();
    return object;
}

template<typename Object>
Object* TryReadObject(TDirectory& file, const std::string& name)
{
    try {
        return ReadObject<Object>(file, name);
    } catch(analysis::exception&) {}
    return nullptr;
}

template<typename Object>
Object* CloneObject(const Object& original_object, const std::string& new_name = "")
{
    const std::string new_object_name = new_name.size() ? new_name : original_object.GetName();
    Object* new_object = dynamic_cast<Object*>(original_object.Clone(new_object_name.c_str()));
    if(!new_object)
        throw analysis::exception("Type error while cloning object '%1%'.") % original_object.GetName();
    return new_object;
}

template<typename Object>
Object* CloneObject(const Object& original_object, const std::string& new_name, bool detach_from_file)
{
    Object* new_object = CloneObject(original_object, new_name);
    if(detach_from_file)
        new_object->SetDirectory(nullptr);
    return new_object;
}

template<typename Object>
Object* ReadCloneObject(TDirectory& file, const std::string& original_name, const std::string& new_name = "",
                        bool detach_from_file = false)
{
    Object* original_object = ReadObject<Object>(file, original_name);
    return CloneObject(*original_object, new_name, detach_from_file);
}

TDirectory* GetDirectory(TDirectory& root_dir, const std::string& name, bool create_if_needed = true);

enum class ClassInheritance { TH1, TTree, TDirectory };

ClassInheritance FindClassInheritance(const std::string& class_name);

struct WarningSuppressor {
    const Int_t old_ignore_level;
    WarningSuppressor(Int_t ignore_level)
        : old_ignore_level(gErrorIgnoreLevel)
    {
        gErrorIgnoreLevel = ignore_level;
    }
    ~WarningSuppressor()
    {
        gErrorIgnoreLevel = old_ignore_level;
    }
};

void RebinAndFill(TH2& new_hist, const TH2& old_hist)
{
    static const auto check_range = [](const TAxis* old_axis, const TAxis* new_axis) {
        const double old_min = old_axis->GetBinLowEdge(1);
        const double old_max = old_axis->GetBinUpEdge(old_axis->GetNbins());
        const double new_min = new_axis->GetBinLowEdge(1);
        const double new_max = new_axis->GetBinUpEdge(new_axis->GetNbins());
        return old_min <= new_min && old_max >= new_max;
    };

    static const auto get_new_bin = [](const TAxis* old_axis, const TAxis* new_axis, int bin_id_old) {
        const double old_low_edge = old_axis->GetBinLowEdge(bin_id_old);
        const double old_up_edge = old_axis->GetBinUpEdge(bin_id_old);
        const int bin_low_new = new_axis->FindFixBin(old_low_edge);
        const int bin_up_new = new_axis->FindFixBin(old_up_edge);

        const double new_up_edge = new_axis->GetBinUpEdge(bin_low_new);
        if(bin_low_new != bin_up_new
                && !(std::abs(old_up_edge-new_up_edge) <= std::numeric_limits<double>::epsilon() * std::abs(old_up_edge+new_up_edge) * 2))
            throw analysis::exception("Uncompatible bin edges");
        return bin_low_new;
    };

    if(!check_range(old_hist.GetXaxis(), new_hist.GetXaxis()))
        throw analysis::exception("x ranges not compatible");

    if(!check_range(old_hist.GetYaxis(), new_hist.GetYaxis()))
        throw analysis::exception("y ranges not compatible");

    for(int x_bin_old = 0; x_bin_old <= old_hist.GetNbinsX() + 1; ++x_bin_old) {
        const int x_bin_new = get_new_bin(old_hist.GetXaxis(), new_hist.GetXaxis(), x_bin_old);
        for(int y_bin_old = 0; y_bin_old <= old_hist.GetNbinsY() + 1; ++y_bin_old) {
            const int y_bin_new = get_new_bin(old_hist.GetYaxis(), new_hist.GetYaxis(), y_bin_old);
            const int bin_old = old_hist.GetBin(x_bin_old, y_bin_old);
            const int bin_new = new_hist.GetBin(x_bin_new, y_bin_new);

            const double cnt_old = old_hist.GetBinContent(bin_old);
            const double err_old = old_hist.GetBinError(bin_old);
            const double cnt_new = new_hist.GetBinContent(bin_new);
            const double err_new = new_hist.GetBinError(bin_new);

            new_hist.SetBinContent(bin_new, cnt_new + cnt_old);
            new_hist.SetBinError(bin_new, std::hypot(err_new, err_old));
        }
    }
}

} // namespace root_ext


std::ostream& operator<<(std::ostream& s, const TVector3& v);
std::ostream& operator<<(std::ostream& s, const TLorentzVector& v);
std::ostream& operator<<(std::ostream& s, const TMatrixD& matrix);
