"""Explicit GL numerical portability candidate, separate from the C75 parent.

Original learned functions and parameters are retained. The small molecular
encoding/fragment-decision path executes on CPU in both inference lanes. The
intensity network remains on the requested device. Existing atom masses are
summed in float64 before the original runner's float32 output/postprocess.
"""
import hashlib,inspect,textwrap,types
import torch

ORIGINAL_INTENSITY_METHOD_SHA256='c69309c379e7a2f169533e610c92dda1c1aa44ec35bf5a56779cf2c8ff4f696b'
REPAIRED_INTENSITY_METHOD_SHA256='a65e522a3e0e210ad696532a5f4ef2510dd90efc9fb4e0637ecc69d053f1d97d'


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    return value


def install(model,decision_model):
    assert model is not decision_model and type(model) is type(decision_model)
    assert not model.training and not decision_model.training
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in decision_model.parameters())
    assert torch.get_num_threads()==2,'Portable decision lane requires the declared two-thread CPU runtime'
    assert not getattr(model,'_c76_portability_installed',False)
    weights=model.state_dict();reference=decision_model.state_dict();assert weights.keys()==reference.keys()
    assert all(torch.equal(v.detach().cpu(),reference[k]) for k,v in weights.items()),'Decision provider weights differ'
    original=type(model).inten_calculation
    before=textwrap.dedent(inspect.getsource(original))
    assert hashlib.sha256(before.encode()).hexdigest()==ORIGINAL_INTENSITY_METHOD_SHA256
    needle='net_fragment_mass_flat = (masses_expanded * frag_targs_f).sum(dim=-1)'
    replacement='net_fragment_mass_flat = (masses_expanded.double() * frag_targs_f.double()).sum(dim=-1)'
    assert before.count(needle)==1;after=before.replace(needle,replacement)
    assert hashlib.sha256(after.encode()).hexdigest()==REPAIRED_INTENSITY_METHOD_SHA256
    namespace={};exec(compile(after,'c76_explicit_mass_reduction','exec'),original.__globals__,namespace)
    model.inten_calculation=types.MethodType(namespace['inten_calculation'],model)
    def embedding(self,*args,**kwargs):
        return move(decision_model.molecular_embedding(*move(args,'cpu'),**move(kwargs,'cpu')),args[0].device)
    def forward(self,*args,**kwargs):
        return move(decision_model.breakpoint_forward(*move(args,'cpu'),**move(kwargs,'cpu')),args[0].device)
    def inference(self,logit,card,num_atoms,debug=False):
        return decision_model.breakpoint_inference(logit.cpu(),card.cpu(),num_atoms.cpu(),debug).to(logit.device)
    model.molecular_embedding=types.MethodType(embedding,model)
    model.breakpoint_forward=types.MethodType(forward,model)
    model.breakpoint_inference=types.MethodType(inference,model)
    model._c76_portability_installed=True
    return dict(original_intensity_method_sha256=ORIGINAL_INTENSITY_METHOD_SHA256,
        repaired_intensity_method_sha256=REPAIRED_INTENSITY_METHOD_SHA256,
        decision_device='cpu',cpu_threads=2,weights_exact=True,
        fragment_decision_functions_unchanged=True,intensity_device=str(next(model.parameters()).device),
        only_numeric_expression_change='Atom-mass multiplication/reduction float64')
