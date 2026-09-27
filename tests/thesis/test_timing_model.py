import unittest
import torch
from esa_thesis.models import MultivariateAE
from esa_thesis.timing_model import TimingAwareAE


class TimingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_predictions_match_base_and_targets_remain_telemetry(self):
        torch.manual_seed(42);base=MultivariateAE(8).eval()
        torch.manual_seed(42);model=TimingAwareAE(8).eval()
        x=torch.randn(2,8,256,1);timing=torch.rand(2,8,256,2)
        with torch.no_grad():
            torch.testing.assert_close(base(x),model(x,timing),rtol=0,atol=0)
            torch.testing.assert_close(model(x),model(x,torch.zeros_like(timing)),rtol=0,atol=0)
        self.assertEqual(model(x,timing).shape,x.shape)

    def test_context_receives_gradient_but_zero_control_does_not(self):
        model=TimingAwareAE(8).eval();x=torch.randn(1,8,256,1)
        model(x,torch.ones(1,8,256,2)).square().mean().backward()
        self.assertGreater(model.timing_proj.weight.grad.abs().sum().item(),0)
        model.zero_grad();model(x).square().mean().backward()
        self.assertEqual(model.timing_proj.weight.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):model(x,torch.zeros(1,8,256,1))


if __name__=='__main__':unittest.main()
