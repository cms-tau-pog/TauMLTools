import json

def create_namestruc(name: str, content: dict) -> str:
    string = "namespace " + name + "{\n"
    # string = "struct " + name + "{\n"
    for key in content:
        string += "const inline " + key + " = " \
                + (str(content[key]),content[key])[type(content[key])==str] \
                + ";\n"
    string += "};\n"
    return string

def create_enum(name: str, content: dict) -> str:
    string = "enum class " + name + "{\n"
    for i, key in enumerate(content):
        string += key +" = " + str(i) + ",\n"
    return string[:-2] + "};\n"

def create_settings(input_file: str, verbose=False) -> str:
    with open(input_file) as file:
        data = json.load(file)
    settings  = create_namestruc("Setup", data["Setup"])
    settings += "".join([create_enum(k+"_Features",data["Features"][k]) for k in data["Features"]])
    if verbose:
        print(settings)
    return settings
