import random
import unittest
from unittest.mock import patch
import torch
from esa_thesis.augmentation import make_pseudo_anomaly
from esa_thesis.short_augmentation import make_training_anomaly

class ShortAugmentationTests(unittest.TestCase):
    def test_legacy_identical_output_and_rng(self):
        x=torch.linspace(-1,1,8*256).reshape(1,8,256,1).repeat(3,1,1,1)
        random.seed(42);torch.manual_seed(42);expected=make_pseudo_anomaly(x)
        py=random.getstate();tr=torch.get_rng_state()
        random.seed(42);torch.manual_seed(42);actual=make_training_anomaly(x)
        self.assertTrue(torch.equal(expected,actual));self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(tr,torch.get_rng_state()))
    def test_single_sample_all_channels_and_support(self):
        x=torch.linspace(-1,1,256).reshape(1,1,256,1).repeat(1,8,1,1);original=x.clone()
        with patch('esa_thesis.short_augmentation.random.random',return_value=1.),patch('esa_thesis.short_augmentation.random.randint',side_effect=[1,255,8]),patch('esa_thesis.short_augmentation.random.choice',side_effect=['offset',1.]):
            y,records=make_training_anomaly(x,mode='short_coordinated',return_metadata=True)
        self.assertTrue(torch.equal(x,original));self.assertTrue(torch.equal(x[:,:,:255],y[:,:,:255]));self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(records[0]['changed_channels'],8);self.assertEqual(records[0]['changed_samples'],8)
    def test_constant_roll_noop_is_reported(self):
        x=torch.ones(1,8,256,1)
        with patch('esa_thesis.short_augmentation.random.random',return_value=1.),patch('esa_thesis.short_augmentation.random.randint',side_effect=[4,10,6,1]),patch('esa_thesis.short_augmentation.random.choice',return_value='roll'):
            y,r=make_training_anomaly(x,mode='short_coordinated',return_metadata=True)
        self.assertTrue(torch.equal(x,y));self.assertEqual(r[0]['changed_samples'],0)
    def test_mixture_repeatability_and_bounds(self):
        x=torch.linspace(-2,2,256).reshape(1,1,256,1).repeat(128,8,1,1)
        def run():
            random.seed(71);torch.manual_seed(71);return make_training_anomaly(x,mode='short_coordinated',return_metadata=True)
        y,r=run();z,s=run();self.assertTrue(torch.equal(y,z));self.assertEqual(r,s)
        self.assertEqual({a['branch'] for a in r},{'short','legacy'})
        for i,a in enumerate(r):
            if a['branch']=='short':
                self.assertTrue(1<=a['length']<=10)
                mask=torch.zeros_like(x[i],dtype=torch.bool);mask[a['selected_channels'],a['start']:a['start']+a['length']]=True
                self.assertTrue(torch.equal(y[i][~mask],x[i][~mask]))
        self.assertTrue(any(a['changed_channels']>4 for a in r if a['branch']=='short'))
    def test_invalid_input(self):
        with self.assertRaises(ValueError):make_training_anomaly(torch.zeros(1,8,256,1),mode='bad')
        with self.assertRaises(ValueError):make_training_anomaly(torch.full((1,8,256,1),float('nan')),mode='short_coordinated')

    def test_no_roll_short_branch_support_and_repeatability(self):
        x=torch.linspace(-2,2,256).reshape(1,1,256,1).repeat(128,8,1,1)
        original=x.clone()
        def run():
            random.seed(91);torch.manual_seed(91)
            return make_training_anomaly(x,mode='short_no_roll',return_metadata=True)
        y,records=run();z,other=run()
        self.assertTrue(torch.equal(x,original));self.assertTrue(torch.equal(y,z));self.assertEqual(records,other)
        self.assertEqual({r['branch'] for r in records},{'short','legacy'})
        self.assertEqual({r['kind'] for r in records if r['branch']=='short'},{'offset','noise'})
        for i,r in enumerate(records):
            if r['branch']=='short':
                self.assertTrue(1<=r['length']<=10)
                self.assertTrue(1<=len(r['selected_channels'])<=8)
                mask=torch.zeros_like(x[i],dtype=torch.bool)
                mask[r['selected_channels'],r['start']:r['start']+r['length']]=True
                self.assertTrue(torch.equal(y[i][~mask],x[i][~mask]))

    def test_no_roll_preserves_legacy_branch_and_rng(self):
        x=torch.linspace(-1,1,8*256).reshape(1,8,256,1).repeat(3,1,1,1)
        def run(mode):
            random.seed(42);torch.manual_seed(42)
            with patch('esa_thesis.short_augmentation.random.random',return_value=0.):
                y=make_training_anomaly(x,mode=mode)
            return y,random.getstate(),torch.get_rng_state()
        a,p,r=run('short_coordinated');b,q,s=run('short_no_roll')
        self.assertTrue(torch.equal(a,b));self.assertEqual(p,q);self.assertTrue(torch.equal(r,s))

if __name__=='__main__':unittest.main()
