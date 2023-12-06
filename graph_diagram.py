import ast, typing, json
import graphviz


def find_func(f_obj) -> None:
    name = f_obj.func.id if isinstance(f_obj.func, ast.Name) else f_obj.func.attr
    return {'name':name, 'args':[ast.unparse(i) for i in f_obj.args], 'kwargs':[[i.arg, ast.unparse(i.value)] for i in f_obj.keywords]}

def to_graphviz(layers:typing.List[dict]) -> str:
    template = """digraph G  {{
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif"]
    edge [fontname="Helvetica,Arial,sans-serif"]
    concentrate=True;
    rankdir=TB;
    node [shape=record];
    {}
    {}
}}
    """.format('\n'.join('n{} [label="{}\\n{}"];'.format(i, a["name"], 
        '' if not a['args']+a['kwargs'] else 
        '|'+'{'+'|'.join(['args']*bool(a['args'])+[j for j, _ in a['kwargs']]) \
            + '}'+'|' + '{'+'|'.join(['{('+', '.join(a['args'])+')}']*bool(a['args']) + 
            ['{'+k+'}' for _, k in a['kwargs']])+'}') 
        for i, a in enumerate(layers)), 
        '\n'.join(f'n{i} -> n{i+1};' for i, _ in enumerate(layers) if i + 1 < len(layers))
    )

    return template

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


def render_graphviz(diagrams) -> None:
    for a, b in diagrams.items():
        if b:
            src = graphviz.Source(to_graphviz(b))
            src.render(f'graph_schematics/graph_gen_{a}')
            

if __name__ == '__main__':
    render_graphviz(to_diagram('pred_tox.py'))