#include <boost/preprocessor/variadic.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>

using namespace ROOT::Math;

std::shared_ptr<TFile> OpenRootFile(const std::string& file_name){
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "READ"));
    if(!file || file->IsZombie())
        throw std::runtime_error("File not opened.");
    return file;
}

std::vector<std::string> SplitValueList(const std::string& _values_str,
                                        bool allow_duplicates = true,
                                        const std::string& separators = " \t",
                                        bool enable_token_compress = true)
{
    std::string values_str = _values_str;
    std::vector<std::string> result;
    if(enable_token_compress)
        boost::trim_if(values_str, boost::is_any_of(separators));
    if(!values_str.size()) return result;
    const auto token_compress = enable_token_compress ? boost::algorithm::token_compress_on
                                                      : boost::algorithm::token_compress_off;
    boost::split(result, values_str, boost::is_any_of(separators), token_compress);
    if(!allow_duplicates) {
        std::unordered_set<std::string> set_result;
        for(const std::string& value : result) {
            if(set_result.count(value))
                throw std::runtime_error("Value "+value+" listed more than once in the value list "
                      +values_str+".");
            set_result.insert(value);
        }
    }
    return result;
}

void CollectInputFiles(const boost::filesystem::path& dir, std::vector<std::string>& files,
                       const boost::regex& pattern, const std::set<std::string>& exclude,
                       const std::set<std::string>& exclude_dirs)
{
    for(const auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dir), {})) {
        if(boost::filesystem::is_directory(entry)
                && !exclude_dirs.count(entry.path().filename().string()))
            CollectInputFiles(entry.path(), files, pattern, exclude, exclude_dirs);
        else if(boost::regex_match(entry.path().string(), pattern)
                && !exclude.count(entry.path().filename().string()))
            files.push_back(entry.path().string());
    }
}

std::vector<std::string> FindInputFiles(const std::vector<std::string>& dirs,
                                        const std::string& file_name_pattern,
                                        const std::string& exclude_list,
                                        const std::string& exclude_dir_list)
{
    auto exclude_vector = SplitValueList(exclude_list, true, ",");
    std::set<std::string> exclude(exclude_vector.begin(), exclude_vector.end());

    auto exclude_dir_vector = SplitValueList(exclude_dir_list, true, ",");
    std::set<std::string> exclude_dirs(exclude_dir_vector.begin(), exclude_dir_vector.end());

    const boost::regex pattern(file_name_pattern);
    std::vector<std::string> files;
    for(const auto& dir : dirs) {
        boost::filesystem::path path(dir);
        CollectInputFiles(path, files, pattern, exclude, exclude_dirs);
    }
    return files;
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
            throw std::invalid_argument("Uncompatible bin edges");
        return bin_low_new;
    };

    if(!check_range(old_hist.GetXaxis(), new_hist.GetXaxis()))
        throw std::invalid_argument("x ranges not compatible");

    if(!check_range(old_hist.GetYaxis(), new_hist.GetYaxis()))
        throw std::invalid_argument("y ranges not compatible");

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
