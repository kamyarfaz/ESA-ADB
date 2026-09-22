# Move this complete project to another Linux server

## What is preserved

Copy the **entire ESA-ADB directory**, including hidden files, `.git`, `data`,
`results`, `results_longrun`, `archive`, and `migration`. The source occupied about
119 GiB on 22 September 2026. Allow at least 150 GiB free on the external disk and
new server for the copy, plus working space for new experiments. Keep the original
until the destination passes verification. No new training was started for migration.

The project's datasets are real files within this folder. The main research paths
resolve relative to the project, so the username and parent directory can change.
Clear any inherited `ESA_ADB_ROOT` pointing at the old server. Some historical
reports/protocols contain old absolute paths as provenance; do not rewrite them.
Strict interrupted-run resume checks include paths, metadata and source hashes,
so resuming an unfinished run after relocation may require a reviewed migration
of its protocol. Completed checkpoints/results can be inspected without resuming.

Three legacy W&B debug-log links point outside the project. Their shared target
has been preserved in `migration/external-logs/`; these logs are not required by
training. Original links are retained for provenance. Codex chats, GitHub login,
Conda installations and files elsewhere in your home directory are not transferred
by copying this project. Research findings and run instructions are in `docs/`.

## 1. Freeze and fingerprint

Finish/stop writers before copying: training, Git operations, notebooks and loggers.
The migration manifest records every file's SHA256 and every symlink's target,
including ignored research artifacts and Git history. It excludes only itself
and its temporary file. Empty directories and file permissions are not verified.

A manifest is prepared with:

```bash
python scripts/research/verify_transfer.py create
```

If files change after it was created, regenerate it immediately before transfer.
Do not run training or Git operations while hashing or copying. The script refuses
files changing during their read; it cannot guarantee a snapshot of a live project.
`migration/installed-packages.txt` records the current environment for reference,
not as a portable environment installer. This migration directory is local and
intentionally excluded from Git, but must be copied to the disk.

## 2. Copy to a Linux-formatted external disk

Replace `/media/YOUR_DISK` with the actual mounted external-disk path. Verify it
is mounted and has sufficient free space before running commands. Do not literally
create that placeholder directory. Use an ext4 or another Linux filesystem that
preserves symlinks and permissions.

```bash
rsync -a --info=progress2 /home/k_faz/projects/ESA-ADB/ /media/YOUR_DISK/ESA-ADB/
python /media/YOUR_DISK/ESA-ADB/scripts/research/verify_transfer.py verify --root /media/YOUR_DISK/ESA-ADB
sync
```

Require **zero failures**, then safely eject/unmount the drive. rsync is rerunnable
if interrupted. No `--delete` is used. Verification checks recorded files; extra
files on a reused destination are not flagged. Prefer a new destination directory.

### If the disk uses exFAT or NTFS

Use a tar archive to preserve Linux metadata and symlinks inside a regular file:

```bash
tar -cf /media/YOUR_DISK/ESA-ADB-transfer.tar -C /home/k_faz/projects ESA-ADB
```

Do not use FAT32: individual files and the archive exceed its 4 GiB limit.
Check tar's successful exit status, then `sync` and safely unmount. On the new
server extract to a new directory, then run the manifest verification below.
If storing both the archive and an extracted copy on one disk, budget twice the
space. Avoid extracting an unreviewed archive as root.

## 3. Copy onto the new server and verify BEFORE running Python imports or Git

For a directory copy:

```bash
mkdir -p ~/projects/ESA-ADB
rsync -a --info=progress2 /media/YOUR_DISK/ESA-ADB/ ~/projects/ESA-ADB/
```

For a tar archive instead:

```bash
mkdir -p ~/projects
tar -xf /media/YOUR_DISK/ESA-ADB-transfer.tar -C ~/projects
```

Then, using any available Python 3 (verification uses only the standard library):

```bash
cd ~/projects/ESA-ADB
python3 scripts/research/verify_transfer.py verify
```

Require **zero failures**. Verification reads the full project and can take time.
Run this before tools that alter cached files or Git metadata. It proves the
recorded file bytes survived transfer, not that every historical experiment is
scientifically valid. Do not delete the source until these checks succeed.

## 4. Recreate the thesis environment

Install Conda/Miniconda on the destination first if unavailable. Do not copy the
old Conda directory as a substitute for installing an environment.

```bash
conda create -n esa-thesis python=3.9 pip -y
conda activate esa-thesis
unset ESA_ADB_ROOT
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-test.txt
python -m esa_thesis doctor
python -m unittest discover -s tests/thesis -p 'test_*.py' -q
```

The CUDA command matches the old PyTorch build; the new server needs a compatible
NVIDIA GPU and driver. For CPU checks, use the `/whl/cpu` index instead. Confirm
`doctor` reports CUDA available before requesting `--device cuda`. Old-server
assistant checks report CUDA unavailable in its sandbox, not a certification of
the user's GPU environment. Expected current test count: **36**.
The existing prepared CSVs need no regeneration. The separate older preprocessing
and TimeEval environment is described in README if you need that workflow later.

Finally inspect `git status --short` and `git remote -v`. Reauthenticate GitHub
on the new server if needed; credentials are not stored in this transfer guide.
No fresh training is required to validate migration.
