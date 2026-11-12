"""Stdlib-only byte-pinned notebook packaging helpers; no chemistry runtime."""
import ast,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(4<<20):h.update(b)
    return h.hexdigest()


def stamp():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def unwrap(source):
    tree=ast.parse(source)
    if len(tree.body)==1 and isinstance(tree.body[0],ast.Try):
        node=tree.body[0].body[0]
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='exec':
            return ast.literal_eval(node.value.args[0].args[0])
    return source


def value(source,name):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1,name
    return ast.literal_eval(nodes[0].value)


def replace_value(source,name,replacement):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1,name
    before=ast.get_source_segment(source,nodes[0]);assert source.count(before)==1
    return source.replace(before,name+' = '+repr(replacement))
