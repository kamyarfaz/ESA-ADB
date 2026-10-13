import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.research.project_snapshot import (
    checked_path, inventory, pack, verify_archive, verify_destination,
)


@unittest.skipUnless(shutil.which('zstd'), 'zstd required for snapshot integration checks')
class SnapshotTests(unittest.TestCase):
    def test_archive_roundtrip_including_links_empty_dirs_and_ignored_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';root.mkdir()
            (root/'empty').mkdir()
            (root/'data').mkdir();(root/'data/raw.npy').write_bytes(b'artifact\0'*1000)
            (root/'link').symlink_to('data/raw.npy')
            (root/'node_modules').mkdir();(root/'node_modules/cache').write_text('runtime')
            (root/'.~lock.file#').write_text('lock')
            os.link(root/'data/raw.npy',root/'copy.npy')
            manifest=Path(tmp)/'manifest.json'
            payload=inventory(root);manifest.write_text(json.dumps(payload))
            self.assertEqual(len(payload['excluded']),2)
            archive=Path(tmp)/'snapshot.tar.zst'
            pack(root,manifest,archive)
            self.assertTrue(verify_archive(archive,manifest)['verified'])
            dest=Path(tmp)/'restore';dest.mkdir()
            subprocess.run(['tar','--zstd','-xf',str(archive),'-C',str(dest)],check=True)
            verify_destination(dest,manifest)
            (dest/'data/raw.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_destination(dest,manifest)

    def test_changed_source_refuses_publication_and_existing_archive_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';root.mkdir()
            (root/'file').write_text('original')
            manifest=Path(tmp)/'manifest.json';manifest.write_text(json.dumps(inventory(root)))
            (root/'file').write_text('changed')
            archive=Path(tmp)/'snapshot.tar.zst'
            with self.assertRaises(ValueError):pack(root,manifest,archive)
            with self.assertRaises(FileExistsError):pack(root,manifest,archive)

    def test_unsafe_member_names_rejected(self):
        for name in ('/etc/passwd','../outside','data/../../outside',''):
            with self.assertRaises(ValueError):checked_path(Path('/tmp/project'),name)


if __name__=='__main__':unittest.main()
