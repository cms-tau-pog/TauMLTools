
def create_settings(input_file: str, verbose=False) -> str:
    '''
    The following subroutine parses the yaml config file and
    returns the following structures in the string format:
    1.  The setup namespace where general global variables
        forDataLoader are specified.
    2.  The enume classes for TauFlat, PfCand_electron,
        PfCand_muon, PfCand_chHad, PfCand_nHad, pfCand_gamma,
        Electron, Muon
    3.  Create CellObjectTypes in enume:
        e.g:
        enum class CellObjectType {
          PfCand_electron,
          PfCand_muon,
          ...};
    4.  FeaturesHelper explicit templated structures:
        e.g:
        template<typename T> struct FeaturesHelper;
        template<> struct FeaturesHelper<PfCand_electron_Features> {...
        template<> struct FeaturesHelper<PfCand_muon_Features> {...
        ...
    5.  FeatureTuple (std::tuple<>) with grid feature types.
    The following string onward is fed to R.gInterpreter
    (alternatively Setup_tmp.h is created)
    to create the corresponding structures.
    '''
    import yaml

    def create_namestruc(content: dict) -> str:
        types_map = {
                int   : "size_t",
                float : "Double_t",
                str   : "std::string",
                list  : "std::vector<std::string>",
                dict  : "std::unordered_map<int, std::string>"
            }

        def items_str(input) -> str:
            if type(input) == list:
                return "{"+','.join('"{0}"'.format(w) for w in input)+"}"
            if type(input) == dict:
                return "{{"+'},{'.join('{0},"{1}"'.format(key,input[key]) for key in input)+"}}"
            elif type(input) == str:
                return "\"" + str(input) + "\""
            else:
                return str(input)

        string = "namespace Setup {\n"
        # variables from Setup section:
        for key in content["Setup"]:
            string += "const inline " + types_map[type(content["Setup"][key])] \
                   + " " + key + " = " + items_str(content["Setup"][key]) + ";\n"
        # variables that define the length of feature lists:
        for features in content["Features_all"]:
            number = len(content["Features_all"][features]) -  len(content["Features_disable"][features])
            string += "const inline size_t n_" + str(features) + " = " + str(number) + ";\n"

        string += "const inline std::vector<std::string> CellObjectTypes {\"" + \
                  "\",\"".join(content["CellObjectType"]) + \
                  "\"};\n"
        
        string += "};\n"
        return string

    def create_enum(key_name: str, content: dict) -> str:
        string = "enum class " + key_name +"_Features " + "{\n"
        # enabled features:
        for i, key in enumerate(content["Features_all"][key_name]):
            if key not in content["Features_disable"][key_name]:
                string += key +" = " + str(i) + ",\n"
        # disabled features:
        for i, key in enumerate(content["Features_disable"][key_name]):
            if key not in content["Features_all"][key_name]:
                raise Exception("Disabled feature {0} is not listed in \"Features_all\" section of cofig file".format(key))
            string += key +" = " + "-1" + ",\n"
        return string[:-2] + "};\n"

    def create_gridobjects(content: dict) -> str:
        string  = "\nenum class CellObjectType {\n"
        string += ",\n".join(content["CellObjectType"])
        string += "};\n\n"

        string +="template<typename T> struct FeaturesHelper;\n"
        for celltype in content["CellObjectType"]:
            number = len(content["Features_all"][celltype]) - len(content["Features_disable"][celltype])
            string += "template<> struct FeaturesHelper<{0}_Features> ".format(celltype) + "{\n"
            string += "static constexpr CellObjectType object_type = CellObjectType::{0};\n".format(celltype)
            string += "static constexpr size_t size = {0};\n".format(number)
            string += "using scaler_type = Scaling::{0};\n".format(celltype) + "};\n\n"

        string += "using FeatureTuple = std::tuple<" \
               + "_Features,\n".join(content["CellObjectType"])\
               + "_Features>;\n"

        return string

    with open(input_file) as file:
        data = yaml.safe_load(file)
    settings  = create_namestruc(data)
    settings  += "\n".join([create_enum(k,data) for k in data["Features_all"]])
    settings += create_gridobjects(data)
    if verbose:
        print(settings)
    return settings

def create_scaling_input(input_file: str, verbose=False) -> str:
    '''
    The following subroutine parses the json config file and
    returns the string with Scaling namespace, where
    all the scaling parameters are specified.
    The following string onward is fed to R.gInterpreter
    to interpret the corresponding c++ vectors in machinary code.

    e.g:
    namespace Scaling {
        struct TauFlat{
            inline static const std::vector<std::vector<float>> mean = {{0},{0},{0},...
            inline static const std::vector<std::vector<float>> std = {{1},{1},{1},...
            ...
        struct PfCand_electron{
            inline static const std::vector<std::vector<float>> mean = {{0,0},{0,0},...
            inline static const std::vector<std::vector<float>> std = {{1,1},{1,1},....
            ...
        ...
    '''
    import json

    groups = [ 'outer', 'inner' ]
    subgroups = [ 'mean', 'std', 'lim_min', 'lim_max' ]

    def conv_str(input) -> str:
        if(input == "-inf"):
            return "-std::numeric_limits<double>::infinity()"
        elif(input == "inf"):
            return "std::numeric_limits<double>::infinity()"
        else:
            return str(input)

    def depth(d):
        if isinstance(d, dict):
            return 1 + (max(map(depth, d.values())) if d else 0)
        return 0

    def create_scaling(content: dict) -> str:
        string = "namespace Scaling {\n"
        for FeatureT in content:
            if depth(content[FeatureT]) == 2:
                form = 1
            elif depth(content[FeatureT]) == 3:
                form = 2
            else:
                raise Exception("Wrong scaling config formatting")

            string += "struct "+FeatureT+"{\n"
            all_vars = content[FeatureT].values()

            for i, subg in enumerate(subgroups):
                string += "inline static const "
                string += "std::vector<std::vector<float>> "
                string += subg + " = "
                if form == 2:
                    string += "{{"+"},{".join([ ","
                            .join([ conv_str(var[inner][subg])
                            for inner in groups ])
                            for var in all_vars ])+"}}"
                else:
                    string += "{{"+"},{".join([
                            conv_str(var[subg])
                            for var in all_vars ])+"}}"

                string += ";\n"
            string += "};\n"
        string += "};\n"
        return string

    with open(input_file) as file:
        data = json.load(file)
    settings  = create_scaling(data)
    if verbose:
        print(settings)
    return settings
