import ROOT
import os

ROOT.gInterpreter.Declare('''
int getMaxElem(const ROOT::RVec<float>& pt, const ROOT::RVec<UChar_t>& genPartFlav,
               const std::set<int>& selected_flav)
{
    int max_index = -1;
    for(size_t n = 0; n < pt.size(); ++n) {
        if(selected_flav.count(genPartFlav.at(n)) && (max_index < 0 || pt.at(n) > pt.at(max_index)))
            max_index = static_cast<int>(n);
    }
    return max_index;
}
''')

def ListToVector(l, elem_type):
    vec = ROOT.std.vector(elem_type)()
    for elem in l:
        vec.push_back(elem)
    return vec


sample_dir = '/eos/cms/store/mc/RunIIAutumn18NanoAODv5/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/'
root_files = []
for root, dirs, files in os.walk(sample_dir):
    for name in files:
        if name.endswith(".root"):
            root_files.append(root + '/' + name)

root_files = ListToVector(root_files, 'string')

df = ROOT.RDataFrame('Events', root_files)

df = df.Define('tau_idx', 'getMaxElem(Tau_pt, Tau_genPartFlav, {0})').Filter('tau_idx >= 0') \
       .Define('tau_pt', 'Tau_pt[tau_idx]')

hist = df.Histo1D(ROOT.RDF.TH1DModel('tau_pt','tau_pt', 1000, 0, 1000), 'tau_pt')

out = ROOT.TFile('W_jet_pt.root', 'RECREATE')
out.WriteTObject(hist.GetPtr(), 'jet_pt', 'Overwrite')
out.Close()
