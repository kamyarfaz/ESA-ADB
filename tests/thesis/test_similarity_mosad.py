"""Scientific invariants for the full-period similarity/MOSAD adaptation."""
import unittest
import copy
import numpy as np
import pandas as pd
import torch
from esa_thesis.similarity_mosad_methods import (nominal_starts, windows, nearest_distance,
    training_shots, FEATURES, normal_contrastive_loss, MOSAD, MOSAD_ARGS,
    mosad_batch_loss, mosad_components, combined_mosad, make_mosad, FixedBinMaxPool)
from esa_thesis.vendor.mosad.loss import supcon_loss
from esa_thesis.vendor.mosad.utils import coe_batch
from esa_thesis.development_recovery import capture_rng, restore_rng

class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)

    def test_fixed_pool_forward_backward_matches_adaptive_including_ties(self):
        for x in [torch.randn(3,4,256), torch.ones(3,4,256)]:
            a=x.clone().requires_grad_(); b=x.clone().requires_grad_()
            u=FixedBinMaxPool(8)(a); v=torch.nn.AdaptiveMaxPool1d(8)(b)
            torch.testing.assert_close(u,v,rtol=0,atol=0)
            u.sum().backward(); v.sum().backward()
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_optimizer_resume_exact_next_step(self):
        args=dict(encoder_decoder_layers=1, tcn_out_channels=4, embedding_dim=12,
                  projection_dim=4, tcn_kernel_size=3)
        model=make_mosad(**args)
        opt=torch.optim.Adam(model.parameters(),lr=.001,amsgrad=True)
        normal=torch.randn(8,8,256); positives=torch.randn(2,8,256)
        def step(m,o):
            o.zero_grad(); loss,_=mosad_batch_loss(m,normal,positives)
            loss.backward(); o.step()
            return loss.item()
        step(model,opt)
        state=copy.deepcopy(model.state_dict()); optim=copy.deepcopy(opt.state_dict()); rng=capture_rng()
        loss=step(model,opt)
        restored=make_mosad(**args); restored.load_state_dict(state)
        ropt=torch.optim.Adam(restored.parameters(),lr=.001,amsgrad=True); ropt.load_state_dict(optim)
        restore_rng(rng); self.assertEqual(loss,step(restored,ropt))
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)

    def test_nearest_matches_bruteforce_and_blocking(self):
        q = torch.randn(7, 8, 16); r = torch.randn(11, 8, 16)
        d, i = nearest_distance(q, r, block=3)
        brute = (q[:, None]-r[None, :]).square().mean((2, 3))
        torch.testing.assert_close(d, brute.min(1).values)
        self.assertTrue(torch.equal(i, brute.argmin(1)))
        # No per-window centering: a sustained level offset must remain visible.
        a = torch.ones(1, 8, 16)*3; b = torch.ones(1, 8, 16)
        self.assertAlmostEqual(nearest_distance(a, b)[0].item(), 4.)

    def test_nominal_windows_exclude_any_positive_and_do_not_cross_end(self):
        y = np.zeros(31, dtype=int); y[10] = 1
        s = nominal_starts(y, length=8, stride=2)
        self.assertTrue(all(not y[i:i+8].any() for i in s))
        self.assertTrue(all(i+8 <= len(y) for i in s))

    def test_training_shots_never_use_future_or_outside_input_events(self):
        times = pd.date_range('2000-01-01', periods=1024, freq='30s').asi8
        rows = [('in', FEATURES[0], '2000-01-01 02:00', '2000-01-01 02:30', 'Anomaly'),
                ('outside', 'channel_2', '2000-01-01 02:00', '2000-01-01 02:30', 'Anomaly'),
                ('future', FEATURES[0], '2006-01-01', '2006-01-02', 'Anomaly')]
        a = pd.DataFrame(rows, columns=['ID','Channel','StartTime','EndTime','Category'])
        for n in ['StartTime','EndTime']: a[n] = pd.to_datetime(a[n], utc=True, format='mixed')
        shots = training_shots(a, times)
        self.assertEqual([r['event_id'] for r in shots], ['in'])
        self.assertEqual(windows(np.zeros((len(times), 8)), [shots[0]['start']]).shape, (1, 8, 256))

    def test_normal_contrastive_matches_upstream_when_well_defined(self):
        z = torch.nn.functional.normalize(torch.randn(9, 32), dim=1)
        y = torch.tensor([0.,0.,0.,0.,0.,1.,1.,1.,1.])
        torch.testing.assert_close(normal_contrastive_loss(z, y), supcon_loss(z, y))
        z = z[:6].clone().requires_grad_(); y = y[:6]
        loss = normal_contrastive_loss(z, y); loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_coe_intervals_nonempty_on_distinct_samples(self):
        x = torch.arange(64.).view(64,1,1).expand(64,256,8).clone()
        out, label = coe_batch(x, torch.zeros(64), 1., 256)
        for row in out:
            self.assertGreater(torch.unique(row).numel(), 1)
        self.assertTrue((label == 1).all())

    def test_mosad_gradients_recovery_and_fixed_scoring(self):
        args = {**MOSAD_ARGS, 'encoder_decoder_layers':1, 'tcn_out_channels':4,
                'embedding_dim':12, 'projection_dim':4, 'tcn_kernel_size':3}
        model = make_mosad(**args)
        normal = torch.randn(8,8,256); positive = torch.randn(2,8,256)
        rng = capture_rng()
        state = {k:v.clone() for k,v in model.state_dict().items()}
        loss, _ = mosad_batch_loss(model, normal, positive); loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        value = loss.item()
        model.load_state_dict(state); model.zero_grad(); restore_rng(rng)
        replay, _ = mosad_batch_loss(model, normal, positive)
        self.assertEqual(value, replay.item())
        model.eval()
        with torch.no_grad(): proj = model.conhead(model.encoder(normal))
        a = mosad_components(model, normal[:2], proj)
        b = mosad_components(model, normal[:2], proj)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Adapter score agrees with the upstream reconstruction/deviation paths.
        flat=normal[:2].reshape(-1,256)
        model.con_inf=False
        with torch.no_grad(): original,mse,_,_=model.inference(flat,None,None,8)
        torch.testing.assert_close(a[:,0],mse)
        torch.testing.assert_close(a[:,0]+a[:,1],original)
        self.assertTrue(np.isfinite(combined_mosad(a.numpy(),0.,2.)).all())

if __name__ == '__main__': unittest.main()
