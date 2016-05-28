/*! Common CERN ROOT extensions.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <memory>

#include <TLorentzVector.h>
#include <TMatrixD.h>
#include <TFile.h>
#include <Compression.h>

#include "exception.h"

namespace root_ext {

inline std::shared_ptr<TFile> CreateRootFile(const std::string& file_name)
{
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "RECREATE", "", ROOT::kZLIB * 100 + 9));
    if(file->IsZombie())
        throw analysis::exception("File '%1%' not created.") % file_name;
    return file;
}

inline std::shared_ptr<TFile> OpenRootFile(const std::string& file_name)
{
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "READ"));
    if(!file || file->IsZombie())
        throw analysis::exception("File '%1%' not opened.") % file_name;
    return file;
}

template<typename Object>
void WriteObject(const Object& object)
{
    TDirectory* dir = object.GetDirectory();
    if(!dir)
        throw analysis::exception("Can't write object to nullptr.");
    dir->WriteTObject(&object, object.GetName(), "Overwrite");
}

template<typename Object>
void WriteObject(const Object& object, TDirectory* dir, const std::string& name = "")
{
    if(!dir)
        throw analysis::exception("Can't write object to nullptr.");
    const std::string name_to_write = name.size() ? name : object.GetName();
    dir->WriteTObject(&object, name_to_write.c_str(), "Overwrite");
}

template<typename Object>
Object* ReadObject(TFile& file, const std::string& name)
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
Object* TryReadObject(TFile& file, const std::string& name)
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
Object* ReadCloneObject(TFile& file, const std::string& original_name, const std::string& new_name = "",
                        bool detach_from_file = false)
{
    Object* original_object = ReadObject<Object>(file, original_name);
    return CloneObject(*original_object, new_name, detach_from_file);
}

} // namespace root_ext


inline std::ostream& operator<<(std::ostream& s, const TVector3& v) {
    s << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const TLorentzVector& v) {
    s << "(pt=" << v.Pt() << ", eta=" << v.Eta() << ", phi=" << v.Phi() << ", E=" << v.E() << ", m=" << v.M() << ")";
    return s;
}

// Based on TMatrixD::Print code.
inline std::ostream& operator<<(std::ostream& s, const TMatrixD& matrix)
{
    if (!matrix.IsValid()) {
        s << "Matrix is invalid";
        return s;
    }

    //build format
    const char *format = "%11.4g ";
    char topbar[100];
    snprintf(topbar,100,format,123.456789);
    Int_t nch = strlen(topbar)+1;
    if (nch > 18) nch = 18;
    char ftopbar[20];
    for (Int_t i = 0; i < nch; i++) ftopbar[i] = ' ';
    Int_t nk = 1 + Int_t(std::log10(matrix.GetNcols()));
    snprintf(ftopbar+nch/2,20-nch/2,"%s%dd","%",nk);
    Int_t nch2 = strlen(ftopbar);
    for (Int_t i = nch2; i < nch; i++) ftopbar[i] = ' ';
    ftopbar[nch] = '|';
    ftopbar[nch+1] = 0;

    s << matrix.GetNrows() << "x" << matrix.GetNcols() << " matrix";

    Int_t cols_per_sheet = 5;
    if (nch <= 8) cols_per_sheet =10;
    const Int_t ncols  = matrix.GetNcols();
    const Int_t nrows  = matrix.GetNrows();
    const Int_t collwb = matrix.GetColLwb();
    const Int_t rowlwb = matrix.GetRowLwb();
    nk = 5+nch*std::min(cols_per_sheet, matrix.GetNcols());
    for (Int_t i = 0; i < nk; i++)
        topbar[i] = '-';
    topbar[nk] = 0;
    for (Int_t sheet_counter = 1; sheet_counter <= ncols; sheet_counter += cols_per_sheet) {
        s << "\n     |";
        for (Int_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++) {
            char ftopbar_out[100];
            snprintf(ftopbar_out, 100, ftopbar, j+collwb-1);
            s << ftopbar_out;
        }
        s << "\n" << topbar << "\n";
        if (matrix.GetNoElements() <= 0) continue;
        for (Int_t i = 1; i <= nrows; i++) {
            char row_out[100];
            snprintf(row_out, 100, "%4d |",i+rowlwb-1);
            s << row_out;
            for (Int_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++) {
                snprintf(row_out, 100, format, matrix(i+rowlwb-1,j+collwb-1));
                s << row_out;
            }
            s << "\n";
        }
    }
    return s;
}
