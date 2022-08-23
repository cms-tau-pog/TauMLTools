/*! Common CERN ROOT extensions.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/RootExt.h"

#include <memory>
#include <map>

#include <TROOT.h>
#include <TClass.h>
#include <TLorentzVector.h>
#include <TMatrixD.h>
#include <TFile.h>
#include <Compression.h>
#include <TAxis.h>

#include "TauMLTools/Core/interface/exception.h"

namespace root_ext {

void WriteObject(const TObject& object, TDirectory* dir, const std::string& name)
{
    if(!dir)
        throw analysis::exception("Can't write object to nullptr.");
    const std::string name_to_write = name.size() ? name : object.GetName();
    dir->WriteTObject(&object, name_to_write.c_str(), "Overwrite");
}


TDirectory* GetDirectory(TDirectory& root_dir, const std::string& name, bool create_if_needed)
{
    if(!name.size() || (name.size() == 1 && name.at(0) == '/'))
        return &root_dir;
    TDirectory* dir = root_dir.GetDirectory(name.c_str());
    if(!dir && create_if_needed) {
        const size_t pos = name.find("/");
        if(pos == std::string::npos || pos == name.size() - 1) {
            root_dir.mkdir(name.c_str());
            dir = root_dir.GetDirectory(name.c_str());
        } else {
            const std::string first_dir_name = name.substr(0, pos), sub_dirs_path = name.substr(pos + 1);
            TDirectory* first_dir = GetDirectory(root_dir, first_dir_name, true);
            dir = GetDirectory(*first_dir, sub_dirs_path, true);
        }
    }

    if(!dir)
        throw analysis::exception("Unable to get directory '%1%' from the root directory '%2%'.")
            % name % root_dir.GetName();
    return dir;
}

ClassInheritance FindClassInheritance(const std::string& class_name)
{
    TClass *cl = gROOT->GetClass(class_name.c_str());
    if(!cl)
        throw analysis::exception("Unable to get TClass for class named '%1%'.") % class_name;

    ClassInheritance inheritance;
    if(cl->InheritsFrom("TH1"))
        inheritance = ClassInheritance::TH1;
    else if(cl->InheritsFrom("TTree"))
        inheritance = ClassInheritance::TTree;
    else if(cl->InheritsFrom("TDirectory"))
        inheritance = ClassInheritance::TDirectory;
    else
        throw analysis::exception("Unknown class inheritance for class named '%1%'.") % class_name;

    return inheritance;
}

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


std::ostream& operator<<(std::ostream& s, const TVector3& v)
{
    s << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
    return s;
}

std::ostream& operator<<(std::ostream& s, const TLorentzVector& v)
{
    s << "(pt=" << v.Pt() << ", eta=" << v.Eta() << ", phi=" << v.Phi() << ", E=" << v.E() << ", m=" << v.M() << ")";
    return s;
}

// Based on TMatrixD::Print code.
std::ostream& operator<<(std::ostream& s, const TMatrixD& matrix)
{
    if (!matrix.IsValid()) {
        s << "Matrix is invalid";
        return s;
    }

    //build format
    static const char *format = "%11.4g ";
    char topbar[100];
    snprintf(topbar,100, format, 123.456789);
    size_t nch = strlen(topbar) + 1;
    if (nch > 18) nch = 18;
    char ftopbar[20];
    for(size_t i = 0; i < nch; i++) ftopbar[i] = ' ';
    size_t nk = 1 + size_t(std::log10(matrix.GetNcols()));
    snprintf(ftopbar+nch/2,20-nch/2,"%s%zud","%",nk);
    size_t nch2 = strlen(ftopbar);
    for (size_t i = nch2; i < nch; i++) ftopbar[i] = ' ';
    ftopbar[nch] = '|';
    ftopbar[nch+1] = 0;

    s << matrix.GetNrows() << "x" << matrix.GetNcols() << " matrix";

    size_t cols_per_sheet = 5;
    if (nch <= 8) cols_per_sheet =10;
    const size_t ncols  = static_cast<size_t>(matrix.GetNcols());
    const size_t nrows  = static_cast<size_t>(matrix.GetNrows());
    const size_t collwb = static_cast<size_t>(matrix.GetColLwb());
    const size_t rowlwb = static_cast<size_t>(matrix.GetRowLwb());
    nk = 5+nch*std::min<size_t>(cols_per_sheet, static_cast<size_t>(matrix.GetNcols()));
    for (size_t i = 0; i < nk; i++)
        topbar[i] = '-';
    topbar[nk] = 0;
    for (size_t sheet_counter = 1; sheet_counter <= ncols; sheet_counter += cols_per_sheet) {
        s << "\n     |";
        for (size_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++) {
            char ftopbar_out[100];
            snprintf(ftopbar_out, 100, ftopbar, j+collwb-1);
            s << ftopbar_out;
        }
        s << "\n" << topbar << "\n";
        if (matrix.GetNoElements() <= 0) continue;
        for (size_t i = 1; i <= nrows; i++) {
            char row_out[100];
            snprintf(row_out, 100, "%4zu |",i+rowlwb-1);
            s << row_out;
            for (size_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++) {
                snprintf(row_out, 100, format, matrix(static_cast<Int_t>(i+rowlwb-1),
                                                      static_cast<Int_t>(j+collwb-1)));
                s << row_out;
            }
            s << "\n";
        }
    }
    return s;
}
