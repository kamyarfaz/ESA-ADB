"""The diagnostic must isolate level changes and preserve other inputs."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('level_probe', Path(__file__).resolve().parents[2] / 'scripts/research/probe_channel_level_coupling.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LevelInterventionTests(unittest.TestCase):
    def test_level_changes_preserve_waveform_and_other_channels(self):
        x = np.random.default_rng(42).normal(size=(3,256,8)).astype(np.float32)
        original=x.copy()
        y=module.recenter_channel(x,7,2.5)
        np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(y[:,:,:7],x[:,:,:7])
        np.testing.assert_allclose(y[:,:,7].mean(axis=1),2.5,atol=1e-6)
        np.testing.assert_allclose(np.diff(y[:,:,7],axis=1),np.diff(x[:,:,7],axis=1),atol=1e-6)

    def test_each_window_gets_its_own_offset(self):
        x=np.zeros((2,256,8),dtype=np.float32)
        x[0,:,7]=1.;x[1,:,7]=5.
        y=module.recenter_channel(x,7,3.)
        np.testing.assert_array_equal(y[:,:,7],3.*np.ones((2,256)))


if __name__=='__main__':unittest.main()
