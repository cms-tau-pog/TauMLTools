import ROOT

ROOT.gInterpreter.ProcessLine('''

EDataType GetBranchType(TBranch* br) {
    TClass* cl = nullptr;
    EDataType t;
    br->GetExpectedType(cl, t);
    return t;
}
''')

def MakeTupleClass(tree_name, tree_file, namespace_name, data_class_name, tree_class_name):
    
    #known_types_simple_types = [ 'Int_t', 'Float_t', 'Long64_t', 'UInt_t', 'ULong64_t', 'UShort_t' ]
    
    simple_type_names = {}
    for type_name in dir(ROOT.EDataType):
        if type_name[0] == 'k':
            enum_v = getattr(ROOT.EDataType, type_name)
            simple_type_names[enum_v] = type_name[1:]
        # TEMPORATILY HAVE FULL PATH
    class_def = f'''
#include "/home/russell/AdversarialTauML/TauMLTools//Core/interface/SmartTree.h" 
namespace {namespace_name} {{
struct {data_class_name} : public root_ext::detail::BaseDataClass {{
'''
    
    file = ROOT.TFile(tree_file)
    tree = file.Get(tree_name)
    columns = []
    for br in tree.GetListOfBranches():
        column_type = br.GetClassName()
        if len(column_type) == 0:
            column_type = simple_type_names[ROOT.GetBranchType(br)]
        column = br.GetName()        
        columns.append([column, column_type])
    
    for vname, vtype in columns:
        class_def += f'    {vtype} {vname};\n'
    
    class_def += '''
    template<typename Other>
    void CopyTo(Other& other) const {
'''
    for vname, vtype in columns:
        class_def += f'        other.{vname} = this->{vname};\n'
    
    class_def += f'''    }}
}};

class {tree_class_name} : public root_ext::detail::BaseSmartTree<{data_class_name}> {{ 
public: 
    static const std::string& Name() {{ static const std::string name = "{tree_name}"; return name; }}
    {tree_class_name}(TDirectory* directory, bool readMode, const std::set<std::string>& disabled_branches = {{}},
                    const std::set<std::string>& enabled_branches = {{}})
        : BaseSmartTree(Name(), directory, readMode, disabled_branches,enabled_branches) {{ Initialize(); }}
    {tree_class_name}(const std::string& name, TDirectory* directory, bool readMode,
                    const std::set<std::string>& disabled_branches = {{}},
                    const std::set<std::string>& enabled_branches = {{}})
        : BaseSmartTree(name, directory, readMode, disabled_branches,enabled_branches) {{ Initialize(); }}
    {tree_class_name}(const std::string& name, const std::vector<std::string>& files_list,
                    const std::set<std::string>& disabled_branches = {{}},
                    const std::set<std::string>& enabled_branches = {{}})
        : BaseSmartTree(name, files_list, disabled_branches, enabled_branches) {{ Initialize(); }}
private:
    inline void Initialize() {{
'''
    
    for vname, vtype in columns:
        class_def += f'        AddBranch("{vname}", _data->{vname});\n'

    
    class_def += '''
        if (GetEntries() > 0) GetEntry(0);
    }
};
}
'''
    #ROOT.gInterpreter.ProcessLine(class_def)
    return class_def

# class_def = MakeTupleClass('taus', '/home/russell/skimmed_tuples/WJetsToLNu/WJetsToLNu_skimmed_pt30.root', 'input_tuple',
#                'Tau', 'TauTuple')

# with open('my_tauTuple.h', 'w') as f:
#     f.write(class_def)
    
# ROOT.gInterpreter.ProcessLine('#include "my_tauTuple.h"')

# tau = ROOT.input_tuple.Tau()

# print(tau.run)
