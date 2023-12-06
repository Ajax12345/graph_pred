import ast, typing, json


def find_func(f_obj) -> None:
    name = f_obj.func.id if isinstance(f_obj.func, ast.Name) else f_obj.func.attr
    return {'name':name, 'args':[ast.unparse(i) for i in f_obj.args], 'kwargs':[[i.arg, ast.unparse(i.value)] for i in f_obj.keywords]}


def layer_mappings(c_obj:ast.ClassDef) -> dict:
    f_n = {i.name:i for i in c_obj.body if isinstance(i, ast.FunctionDef)}
    if '__init__' not in f_n and 'forward' not in f_n:
        return

    var_mappings = {i.targets[0].attr:find_func(i.value) for i in f_n['__init__'].body if isinstance(i, ast.Assign)}

    return [var_mappings.get((p:=find_func(i.value))['name'], p) for i in f_n['forward'].body if isinstance(i, ast.Assign) and isinstance(i.value, ast.Call)]


def to_diagram(f_name:str) -> dict:
    with open(f_name) as f:
        src = ast.parse(f.read())

    return {i.name:layer_mappings(i) for i in src.body if isinstance(i, ast.ClassDef) 
                and any(j.attr == 'Module' for j in getattr(i, 'bases', []))}

    
    

if __name__ == '__main__':
    print(json.dumps(to_diagram('pred_tox.py'), indent=4))