import unittest
import importlib.util
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.cross_mission_study import medoid_panel,jobs,scopes
from esa_thesis import cross_mission_study as study


class CrossMissionTests(unittest.TestCase):
    def test_groups_and_order_ignore_non_targets(self):
        meta=pd.DataFrame(dict(Channel=['channel_1','channel_2','channel_3','channel_4'],Target=['YES','YES','YES','NO'],Group=[1,1,2,1]))
        inv=pd.DataFrame(dict(channel=meta.Channel,numeric=True,constant=False,coverage=[.99,.98,.9,1.])).set_index('channel')
        matrix=np.eye(4);matrix[0,1]=matrix[1,0]=-.9
        result=medoid_panel(meta,inv,list(meta.Channel),matrix,matrix)
        self.assertEqual(result['base'],['channel_1','channel_3'])
        self.assertEqual(result['expanded'],['channel_1','channel_3','channel_2'])

    def test_medoid_keeps_relationship_structure(self):
        meta=pd.DataFrame(dict(Channel=['channel_1','channel_2','channel_3'],Target='YES',Group=1))
        inv=pd.DataFrame(dict(channel=meta.Channel,numeric=True,constant=False,coverage=1.)).set_index('channel')
        matrix=np.array([[1,.8,.1],[.8,1,.6],[.1,.6,1.]])
        result=medoid_panel(meta,inv,list(meta.Channel),matrix,matrix)
        self.assertEqual(result['base'],['channel_2'])
        self.assertEqual(result['expanded'],['channel_2','channel_1'])

    def test_unsupported_target_refused(self):
        meta=pd.DataFrame(dict(Channel=['channel_1'],Target=['YES'],Group=[1]))
        inv=pd.DataFrame(dict(channel=meta.Channel,numeric=False,constant=False,coverage=1.)).set_index('channel')
        with self.assertRaises(ValueError):medoid_panel(meta,inv,['channel_1'],np.eye(1),np.eye(1))

    def test_all_missions_methods_seeds_in_plan(self):
        sel={'panels':{str(m):dict(base=['a'],expanded=['a','b']) for m in (1,2)}}
        plan=jobs(sel)
        self.assertEqual(len(plan),18);self.assertEqual(len({j['name'] for j in plan}),18)
        self.assertEqual({j['seed'] for j in plan},{42,43,44})
        self.assertEqual([j['mission'] for j in plan],[2]*9+[1]*9)
        for j in plan:self.assertEqual(len(j['features']),2 if j['method']=='expanded_short_no_roll' else 1)

    def test_mapping_enforces_unique_compatible_inputs(self):
        path=Path(__file__).resolve().parents[2]/'scripts/research/cross_mission_channel_audit_20261118.py'
        spec=importlib.util.spec_from_file_location('channel_audit',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        u=np.array([[0.,1.,10.],[0.,1.,2.]])
        allowed=np.array([[True,False,False],[True,True,True]])
        assignment,_=mod.assign(u,allowed,[np.eye(2)]*2,[np.eye(3)]*2)
        self.assertEqual(assignment,[0,1])
        with self.assertRaises(ValueError):mod.assign(u,np.array([[True,False,False],[True,False,False]]),[np.eye(2)]*2,[np.eye(3)]*2)

    def test_training_coverage_utc_and_boundary(self):
        path=Path(__file__).resolve().parents[2]/'scripts/research/cross_mission_channel_audit_20261118.py'
        spec=importlib.util.spec_from_file_location('coverage_audit',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            mod.ROOT=Path(d);raw=Path(d)/'data/ESA-Mission2';raw.mkdir(parents=True)
            pd.DataFrame(dict(ID=['past','train','future'],Channel='channel_1',
                StartTime=['1999-12-31T23:59:59Z','2000-12-31T23:59:59Z','2001-01-01T00:00:00Z'],
                EndTime=['2000-01-01T00:00:00Z','2001-01-01T00:00:01Z','2001-01-01T00:00:01Z'])).to_csv(raw/'labels.csv',index=False)
            pd.DataFrame(dict(ID=['past','train','future'],Category='Anomaly')).to_csv(raw/'anomaly_types.csv',index=False)
            rows=mod.coverage(2,{'included':['channel_1'],'excluded':['channel_2']})
            for row in rows:
                self.assertEqual(row['system_events'],2)
                self.assertEqual(row['covered_ids'],['past','train'] if row['panel']=='included' else [])

    def test_preparation_preserves_anomalous_values_masks_training_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'data/ESA-Mission2';(raw/'channels').mkdir(parents=True)
            t=pd.date_range('2000-01-01',periods=40,freq='10s')
            pd.DataFrame({'value':np.arange(40,dtype=np.float32)},index=t).to_pickle(raw/'channels/channel_1.zip',compression='zip')
            pd.DataFrame(dict(ID=['a'],Channel=['channel_1'],StartTime=[t[1].isoformat()+'Z'],EndTime=[t[2].isoformat()+'Z'])).to_csv(raw/'labels.csv',index=False)
            pd.DataFrame(dict(ID=['a'],Category=['Anomaly'])).to_csv(raw/'anomaly_types.csv',index=False)
            bounds=(str(t[0]),str(t[-1]+pd.Timedelta(seconds=1)))
            with patch.object(study,'ROOT',root),patch.object(study,'PREP',root/'prepared'),patch.object(study,'SPLITS',{2:{'train':bounds,'calibration':bounds}}):
                for split in ('train','calibration'):study.prepare_split('m2',2,split,['channel_1'],10)
                x=np.load(root/'prepared/m2/train/values.npy');v=np.load(root/'prepared/m2/calibration/values.npy')
                np.testing.assert_array_equal(x[:,0],np.arange(40,dtype=np.float32));np.testing.assert_array_equal(x,v)
                excluded=np.load(root/'prepared/m2/train/excluded.npy')
                np.testing.assert_array_equal(np.flatnonzero(excluded),[1,2])
                self.assertFalse(np.load(root/'prepared/m2/calibration/excluded.npy').any())
                study.prepare_split('m2',2,'train',['channel_1'],10) # Verify completed receipt on resume.

    def test_system_scope_keeps_events_outside_base(self):
        ann=pd.DataFrame(dict(ID=['inside','outside'],Channel=['a','b'],StartTime=pd.to_datetime([1,5],unit='s'),
                             EndTime=pd.to_datetime([2,6],unit='s'),Category=['Anomaly','Anomaly']))
        with patch.object(study,'labels',return_value=ann):
            ev=scopes(2,np.arange(10)*10**9,['a'])
        self.assertEqual(set(ev['system_all_channels'].by_id),{'inside','outside'})
        self.assertEqual(set(ev['fixed_base_channels'].by_id),{'inside'})

if __name__=='__main__':unittest.main()
