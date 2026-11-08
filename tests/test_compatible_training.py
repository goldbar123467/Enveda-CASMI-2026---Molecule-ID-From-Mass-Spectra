"""Logical candidate competition must survive microbatch recovery."""
import copy,sys,unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch=None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA objective test')
class CompatibleObjectiveTests(unittest.TestCase):
    def test_microbatches_keep_all_logical_candidates_and_duplicate_mask(self):
        from compatible_training import accumulated_step
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.spec=torch.nn.Linear(20,8);self.aux=torch.nn.Linear(8,32);self.mol=torch.nn.Linear(32,8)
            def encode_spectra(self,x,*_):
                z=self.spec(x.flatten(1));return torch.nn.functional.normalize(z,dim=1),self.aux(z)
            def encode_molecules(self,x):return torch.nn.functional.normalize(self.mol(x),dim=1)
        torch.manual_seed(123);torch.set_num_threads(2)
        a=Tiny().cuda();b=copy.deepcopy(a)
        batch=(torch.randn(12,5,4,device='cuda'),);pos=(torch.rand(12,32,device='cuda')>.8).float()
        neg=(torch.rand(12,4,32,device='cuda')>.8).float();mask=torch.zeros(12,4,device='cuda',dtype=torch.bool);mask[:,0]=True
        cfg=dict(temperature=.1,aux_weight=.5);weight=torch.full((32,),3.,device='cuda')
        records=[]
        for model,micro in [(a,12),(b,3)]:
            opt=torch.optim.SGD(model.parameters(),lr=.01);scaler=torch.amp.GradScaler('cuda',init_scale=16.)
            records.append(accumulated_step(model,opt,scaler,batch,pos,neg,micro,cfg,mask,weight))
        self.assertAlmostEqual(records[0]['loss'],records[1]['loss'],delta=.003)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=.001,atol=.00003)
        self.assertFalse(any(r['amp_skipped'] for r in records))


if __name__=='__main__':unittest.main()
