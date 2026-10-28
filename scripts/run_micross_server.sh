#!/usr/bin/env bash
# First full-data CPU verification. Keep each run isolated and log its exit status.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${1:-micross_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Invalid run id" >&2
  exit 2
fi
RESULT="$ROOT/artifacts/$RUN_ID"
DATA="$ROOT/data/processed/$RUN_ID"
mkdir -p "$RESULT" "$ROOT/data/tmp/$RUN_ID"
if [[ -f "$RESULT/exit_status" || -f "$DATA/feature_schema.json" ]]; then
  echo "Choose a new run id; existing run outputs found" >&2
  exit 2
fi
trap 'status=$?; printf "%s\n" "$status" > "$RESULT/exit_status"; date -Is > "$RESULT/finished_at"' EXIT
date -Is > "$RESULT/started_at"
free -h
# The initial run uses existing packages in an isolated uv environment. No driver changes.
if [[ ! -x .venv/bin/python ]]; then
  task_python="$(python3 -c 'import sys; print(sys.executable)')"
  uv venv --python "$task_python" --system-site-packages .venv
fi
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
uv run --no-sync python - <<'CHECK'
import json, sys
from pathlib import Path
import numpy, torch
root=Path.cwd()
report={'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,
        'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,
        'execution_device':'cpu','dependency_mode':'uv venv with existing system site packages; no lockfile sync'}
print(json.dumps(report,indent=2))
# The exact environment is also captured in the run log.
CHECK
export PATH="$ROOT/.tools/7zip-23.01:$HOME/.local/bin:$PATH"
if ! command -v 7zz >/dev/null && ! command -v 7z >/dev/null; then
  mkdir -p "$ROOT/.tools/7zip-23.01"
  task_archive="$ROOT/.tools/7zip-23.01/linux-x64.tar.xz"
  task_download_ok=false
  for task_attempt in 1 2 3 4 5 6; do
    if [[ -s "$task_archive" ]] && tar -tJf "$task_archive" >/dev/null 2>&1; then
      task_download_ok=true
      break
    fi
    echo "7-Zip download/resume attempt $task_attempt of 6"
    if /usr/bin/curl -fsSL -C - --connect-timeout 20 --max-time 300 \
      https://www.7-zip.org/a/7z2301-linux-x64.tar.xz -o "$task_archive"; then
      if tar -tJf "$task_archive" >/dev/null 2>&1; then
        task_download_ok=true
        break
      fi
    fi
    echo "Download incomplete; retaining bytes for the next attempt"
    sleep 5
  done
  if [[ "$task_download_ok" != true ]]; then
    echo "7-Zip archive is still incomplete after retries" >&2
    exit 18
  fi
  echo "7-Zip archive verified"
  tar -xJf "$task_archive" -C "$ROOT/.tools/7zip-23.01" 7zz
fi
uv run --no-sync python - <<'INPUTS'
from pathlib import Path
from zipfile import ZipFile
root=Path('data/MicroSS')
for folder,count in [('metric',34),('trace',31),('business',45)]:
    directory=root/folder
    paths=[directory/f'{folder}_split.z{i:02d}' for i in range(1,count+1)]
    paths.append(directory/f'{folder}_split.zip')
    missing=[str(p) for p in paths if not p.is_file() or p.stat().st_size==0]
    if missing: raise FileNotFoundError(missing)
    print(folder, 'original archive volumes present:',len(paths))
run=root/'run/run_table_2021-07.csv'
if not run.exists():
    with ZipFile(root/'run/run.zip') as archive:
        members=[name for name in archive.namelist() if name.endswith('run_table_2021-07.csv')]
        if len(members)!=1: raise ValueError('July run CSV not uniquely identified')
        run.write_bytes(archive.read(members[0]))
print('July run CSV:',run)
INPUTS
if [[ ! -f data/MicroSS/metric/metric_full.zip ]]; then
  echo "Combining original metric ZIP volumes (original files retained)"
  zip -s 0 data/MicroSS/metric/metric_split.zip --out data/MicroSS/metric/metric_full.pending.zip     > "$RESULT/metric_merge.log" 2>&1
  mv data/MicroSS/metric/metric_full.pending.zip data/MicroSS/metric/metric_full.zip
fi
uv run --no-sync python scripts/run_micross.py --check-inputs-only
# Full preprocessing and graph extraction, then one CPU training epoch and RCA export.
/usr/bin/time -v uv run --no-sync python scripts/run_micross.py   --epochs 1 --batch-size 32 --device cpu --source-timezone UTC   --data-dir "$DATA" --output-dir "$RESULT" --work-dir "$ROOT/data/tmp/$RUN_ID"
