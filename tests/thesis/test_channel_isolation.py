import unittest
import torch
from esa_thesis.models import MultivariateAE
from esa_thesis.channel_isolation import ChannelIsolatedAE
from esa_thesis.isolation_comparison import isolated_model
from esa_thesis import inner_validation


class ChannelIsolationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_same_seed_identical_parameters(self):
        torch.manual_seed(42);joint=MultivariateAE(8)
        torch.manual_seed(42);isolated=ChannelIsolatedAE(8)
        self.assertEqual(joint.state_dict().keys(),isolated.state_dict().keys())
        for key,value in joint.state_dict().items():
            torch.testing.assert_close(value,isolated.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in joint.parameters()),sum(p.numel() for p in isolated.parameters()))

    def test_other_channel_outputs_and_gradients_are_isolated(self):
        torch.manual_seed(42);model=ChannelIsolatedAE(8).eval()
        x=torch.randn(2,8,256,1,requires_grad=True)
        y=model(x);changed=x.detach().clone();changed[:,7]+=4
        shifted=model(changed)
        torch.testing.assert_close(y[:,:7],shifted[:,:7],rtol=0,atol=1e-6)
        self.assertGreater(float((y[:,7]-shifted[:,7]).abs().max()),1e-4)
        y[:,0].sum().backward()
        torch.testing.assert_close(x.grad[:,1:],torch.zeros_like(x.grad[:,1:]),rtol=0,atol=0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_restores_factory_after_failure(self):
        original=inner_validation.MultivariateAE
        with self.assertRaises(RuntimeError):
            with isolated_model():
                self.assertIs(inner_validation.MultivariateAE,ChannelIsolatedAE)
                raise RuntimeError('probe')
        self.assertIs(inner_validation.MultivariateAE,original)


if __name__=='__main__':unittest.main()
