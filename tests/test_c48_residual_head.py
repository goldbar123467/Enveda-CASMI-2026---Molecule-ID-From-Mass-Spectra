"""Check exact ridge initialization, frozen anchor and a true residual bound."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c48_gpu_model import BoundedResidualHead


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        self.weights=torch.randn(1040,768)*.01;self.inputs=torch.randn(4,1039)
        self.model=BoundedResidualHead(dict(hidden=32,dropout=.1,maximum_relative_residual_norm=.15),self.weights)
    def test_initial_model_equals_clean_ridge_in_train_and_eval_modes(self):
        expected=torch.nn.functional.normalize(self.inputs@self.weights[:-1]+self.weights[-1],dim=-1)
        for training in [True,False]:
            self.model.train(training);torch.testing.assert_close(self.model.encode_spectra(self.inputs),expected,rtol=0,atol=0)
    def test_residual_bound_holds_even_for_extreme_network_outputs(self):
        with torch.no_grad():self.model.residual[-1].weight.fill_(1000);self.model.residual[-1].bias.fill_(1000)
        anchor,delta=self.model.components(self.inputs)
        self.assertTrue(torch.all(delta.norm(dim=-1)<=.150001*anchor.norm(dim=-1)))
    def test_anchor_and_molecular_geometry_are_not_trainable(self):
        self.assertFalse(self.model.anchor_weights.requires_grad)
        self.assertNotIn('anchor_weights',dict(self.model.named_parameters()))
        y=torch.randn(3,768);torch.testing.assert_close(self.model.encode_molecules(y),torch.nn.functional.normalize(y,dim=-1),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
