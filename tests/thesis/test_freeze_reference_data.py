"""Boundary, preservation and tamper checks for frozen reference packaging."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('freeze_reference', Path(__file__).resolve().parents[2]/'scripts/research/freeze_reference_data.py')
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)


class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bounds = ('2000-01-01 00:00:00','2000-01-01 00:01:00',
                       '2000-01-01 00:02:00','2000-01-01 00:03:00')
        header = b'timestamp,channel_1,is_anomaly_channel_1\n'
        self.rows = [f'2000-01-01 00:0{i//2}:{30*(i%2):02},1.25,{i%4}\n'.encode() for i in range(7)]
        self.sources = [self.root/'source_train.csv',self.root/'source_test.csv']
        self.sources[0].write_bytes(header+b''.join(self.rows[:5]))
        self.sources[1].write_bytes(header+b''.join(self.rows[4:]))
        self.header = header
        self.labels=self.root/'labels.csv.gz';self.labels.write_bytes(b'fixture')
        self.out=self.root/'frozen'

    def build(self, hashes=None):
        return frozen.package(self.sources, hashes or [frozen.digest(p) for p in self.sources],
                              self.labels,frozen.digest(self.labels),self.out,self.bounds)

    def test_exact_half_open_partition_and_no_overwrite(self):
        self.build()
        manifest=frozen.verify(self.out)
        self.assertEqual(manifest['rows'],dict.fromkeys(frozen.NAMES,2))
        for i,name in enumerate(frozen.NAMES):
            self.assertEqual((self.out/name).read_bytes(),self.header+b''.join(self.rows[2*i:2*i+2]))
        with self.assertRaises(FileExistsError):self.build()

    def test_source_checksum_failure_never_publishes(self):
        with self.assertRaisesRegex(ValueError,'checksum'):self.build(['0'*64,'0'*64])
        self.assertFalse(self.out.exists())
        self.assertTrue(self.out.with_name('frozen.incomplete').exists())

    def test_missing_timestamp_rejected(self):
        self.sources[0].write_bytes(self.header+self.rows[0]+b''.join(self.rows[2:5]))
        with self.assertRaisesRegex(ValueError,'timestamp'):self.build()
        self.assertFalse(self.out.exists())

    def test_changed_data_and_unexpected_inventory_rejected(self):
        self.build()
        p=self.out/'testing.csv';p.chmod(0o644);p.write_bytes(p.read_bytes().replace(b'1.25',b'9.25',1))
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):frozen.verify(self.out)

    def test_wrong_label_checksum_rejected(self):
        with self.assertRaisesRegex(ValueError,'label file'):
            frozen.package(self.sources,[frozen.digest(p) for p in self.sources],self.labels,'0'*64,self.out,self.bounds)
        self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()
