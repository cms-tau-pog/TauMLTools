import yaml

def create_namestruc(name: str, content: dict) -> str:
    types_map = {
        int   : "size_t",
        float : "Double_t",
        str   : "std::string",
        dict  : "std::vector<std::string>"
        }

    items_str = lambda c : str(c) if type(c) != dict \
        else "{"+','.join('"{0}"'.format(w) for w in c.keys())+"}"

    string = "namespace " + name + "{\n"
    for key in content:
        string += "const inline " + types_map[type(content[key])] \
               + " " + key + " = " + items_str(content[key]) + ";\n"
    string += "};\n"
    return string

def create_enum(key_name: str, content: dict) -> str:
    string = "enum class " + key_name +"_Features " + "{\n"
    # enabled features
    for i, key in enumerate(content["Features_enable"][key_name]):
        string += key +" = " + str(i) + ",\n"
    # disabled features
    for i, key in enumerate(content["Features_disable"][key_name]):
        string += key +" = " + "-1" + ",\n"
    return string[:-2] + "};\n"

def create_settings(input_file: str, verbose=False) -> str:
    with open(input_file) as file:
        data = yaml.load(file)
    settings  = create_namestruc("Setup", data["Setup"])
    settings  += "\n".join([create_enum(k,data) for k in data["Features_enable"]])
    if verbose:
        print(settings)
    return settings
