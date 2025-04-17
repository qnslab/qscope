import inspect

from qscope.server import client


def generate_method_declarations():
    output = []
    for name, func in inspect.getmembers(client, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(func)
        # Skip first param (client_connection)
        params = list(sig.parameters.items())[1:]
        param_str = ", ".join(f"{name}: {param.annotation}" for name, param in params)
        return_anno = sig.return_annotation
        output.append(f"""                                                                         
     def {name}({param_str}) -> {return_anno}:                                                     
         \"""{func.__doc__ or ""}\"""                                                              
         return self.__getattr__('{name}')({", ".join(p[0] for p in params)})                      
         """)
    return "\n".join(output)


if __name__ == "__main__":
    print(generate_method_declarations())
