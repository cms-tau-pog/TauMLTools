
def create_settings(data: dict, verbose=False) -> str:
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
                bool  : "Bool_t",
                int   : "long int",
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
            elif type(input) == bool:
                if input: return "true"
                else: return "false"
            else:
                return str(input)
        def items_float(input):
            return "{"+','.join([str(k)  for k in input])+"}"
        def items_float_vector(input):
            return "{"+','.join([items_float(it) for it in input])+"}"

        string = "namespace Setup {\n"
        # variables from Setup section:
        for key in content["Setup"]:

            value = content["Setup"][key]
            if type(value) == list and (type(value[0]) == float or type(value[0]) == int):
                string += "const inline std::vector<Double_t>" \
                       + " " + key + " = " + items_float(value) + ";\n"
            elif type(value) == list and type(value[0]) == list and (type(value[0][0]) == float or type(value[0][0]) == int):
                string += "const inline std::vector<std::vector<Double_t>>"\
                       + " " + key + " = " + items_float_vector(value) + ";\n"
            else:
                string += "const inline " + types_map[type(value)] \
                       + " " + key + " = " + items_str(value) + ";\n"

        # variables that define the length of feature lists:
        for features in content["Features_all"]:
            number = len(content["Features_all"][features]) -  len(content["Features_disable"][features])
            string += "const inline size_t n_" + str(features) + " = " + str(number) + ";\n"
            if "SequenceLength" in  content: # sequence length
                string += "const inline size_t nSeq_" + str(features) + " = " + str(content["SequenceLength"][features]) + ";\n"

        string += "const inline std::vector<std::string> CellObjectTypes {\"" + \
                  "\",\"".join(content["CellObjectType"]) + \
                  "\"};\n"

        string += "};\n"
        return string

    def create_enum(key_name: str, content: dict) -> str:
        feature_list_enabled, feature_list_disabled = [], []
        for feature_dict in content["Features_all"][key_name]:
            assert len(feature_dict) == 1 and type(feature_dict) == dict
            if list(feature_dict)[0] in content["Features_disable"][key_name]:
                feature_list_disabled.append(list(feature_dict)[0])
                continue
            feature_list_enabled.append(list(feature_dict)[0])
        string = "enum class " + key_name +"_Features " + "{\n"
        # enabled features:
        for i, feature in enumerate(feature_list_enabled):
            string += feature +" = " + str(i) + ",\n"
        # disabled features:
        for i, feature in enumerate(content["Features_disable"][key_name]):
            if feature not in feature_list_disabled:
                raise Exception("Disabled feature {0} is not listed in \"Features_all\" section of cofig file".format(feature))
            string += feature +" = " + "-1" + ",\n"
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
            if "SequenceLength" in content:
                if celltype in content["SequenceLength"]:
                    string += "static constexpr size_t length = {0};\n".format(
                        content["SequenceLength"][celltype])
            string += "using scaler_type = Scaling::{0};\n".format(celltype) + "};\n\n"

        string += "using FeatureTuple = std::tuple<" \
               + "_Features,\n".join(content["CellObjectType"])\
               + "_Features>;\n"

        return string

    settings  = create_namestruc(data)
    settings  += "\n".join([create_enum(k,data) for k in data["Features_all"]])
    settings += create_gridobjects(data)
    if verbose:
        print(settings)
    return settings

def create_scaling_input(input_scaling_file: str, training_cfg_data: dict, verbose=False) -> str:
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
    import yaml

    global_group = 'global'
    cone_groups = [ 'outer', 'inner' ]
    subgroups = [ 'mean', 'std', 'lim_min', 'lim_max' ]

    def conv_str(input) -> str:
        if(input == "-inf"):
            return "-std::numeric_limits<double>::infinity()"
        elif(input == "inf"):
            return "std::numeric_limits<double>::infinity()"
        else:
            return str(input)

    def create_scaling(content_scaling: dict, content_cfg: dict) -> str:
        string = "namespace Scaling {\n"
        for FeatureT in content_cfg["Features_all"]:
            string += "struct "+FeatureT+"{\n"
            duplicate = FeatureT in content_cfg['CellObjectType'] # whether to duplicate scaling param values across cone_groups
            for subg in subgroups:
                string += "inline static const "
                string += "std::vector<std::vector<float>> "
                string += subg + " = "
                var_string = []

                feature_list_enabled= []
                for feature_dict in content_cfg["Features_all"][FeatureT]:
                    assert len(feature_dict) == 1 and type(feature_dict) == dict
                    if list(feature_dict)[0] in content_cfg["Features_disable"][FeatureT]:
                        continue
                    feature_list_enabled.append(list(feature_dict)[0])

                for var in feature_list_enabled:
                    var_params = content_scaling[FeatureT][var]
                    if len(var_params)==len(cone_groups) and all([g in var_params.keys() for g in cone_groups]):
                        var_string.append(",".join([conv_str(var_params[cone_group][subg]) for cone_group in cone_groups]))
                    elif len(var_params)==1 and global_group in var_params.keys():
                        if duplicate:
                            var_string.append(",".join([conv_str(var_params[global_group][subg]) for cone_group in cone_groups]))
                        else:
                            var_string.append(conv_str(var_params[global_group][subg]))
                    else:
                        raise Exception(f"wrong format for scaling params in json for variable {var}: expect either dictionary with either a key {global_group}, or keys {cone_groups}")
                string += "{{"+"},{".join(var_string)+"}};\n"
            string += "};\n"
        string += "};\n"
        return string

    with open(input_scaling_file) as scaling_file:
        scaling_data = json.load(scaling_file)
    settings  = create_scaling(scaling_data, training_cfg_data)
    if verbose:
        print(settings)
    return settings
