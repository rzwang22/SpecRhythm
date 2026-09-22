"""Execute the pinned Bash entry; checkout and the already-tested export CLI substituted."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "scripts/reexport_ordinary_cpu_pinned.sh"
SHA = "f83edb43d22dcbb605468b8518bcca7c1bc36f45"


@pytest.mark.parametrize("code", [0, 41, 23])
def test_export_only_fixed_entry_and_outer_shell(tmp_path, code):
    binary = tmp_path / "bin"
    binary.mkdir()
    source = tmp_path / "old evidence"
    source.mkdir()
    (source / "old.tar.gz").write_bytes(b"unchanged")
    git = binary / "git"
    git.write_text("#!" + sys.executable + "\n" + '''import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\\n')
if 'cat-file' in args:assert args[-1]==os.environ['SHA']+'^{commit}'
if 'worktree' in args:
 assert args[-1]==os.environ['SHA'] and '--detach' in args
 root=pathlib.Path(args[-2]);(root/'src/specrhythm/serving').mkdir(parents=True)
 (root/'src/specrhythm/serving/ordinary_reexport.py').write_text('# fixture')
''')
    git.chmod(0o755)
    python = binary / "export-python"
    python.write_text("#!" + sys.executable + "\n" + '''import os,sys
a=sys.argv[1:]
assert a[:2]==['-m','specrhythm.serving.ordinary_reexport']
assert a[a.index('--source')+1]==os.environ['SOURCE']
assert a[a.index('--expected-execution')+1]=='944328263b02a915f397bcb03c8c743c71e9a56c'
assert a[a.index('--exporter-commit')+1]==os.environ['SHA']
assert a[a.index('--tag')+1].startswith('a100-ordinary-cpu-reexport-')
print('UPLOAD ONLY: /tmp/reexport-fixture.tar.gz')
raise SystemExit(int(os.environ['CODE']))
''')
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "CALLS": str(tmp_path / "calls"), "SHA": SHA, "CODE": str(code),
           "SOURCE": str(source), "SR_K3_REPO": str(tmp_path / "repo"),
           "SR_FIXED_PYTHON": str(python)}
    shell = '''if bash "$1" "$2"; then rc=0; else rc=$?; fi
printf 'ORIGINAL_RC=%s\nSHELL_SURVIVED\n' "$rc"
'''
    result = subprocess.run(["bash", "-c", shell, "entry", str(ENTRY), str(source)],
                            env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert f"ORIGINAL_RC={code}" in result.stdout and "SHELL_SURVIVED" in result.stdout
    assert result.stdout.count("UPLOAD ONLY:") == 1
    assert (source / "old.tar.gz").read_bytes() == b"unchanged"
    calls = [json.loads(s) for s in (tmp_path / "calls").read_text().splitlines()]
    assert sum("worktree" in c for c in calls) == 1
    if code:
        assert f"Reexport stopped (rc={code})" in result.stderr
