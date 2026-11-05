"""Isolated process fixtures verify tmux survival, locks, exit codes and scoped stop.

These are explicit test programs, never robotics rollout evidence.
"""
from pathlib import Path
import json
import shutil
import subprocess
import time
import pytest

ROOT = Path(__file__).resolve().parents[1]


def wait_stopped(path, seconds=12):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        try:
            status=json.loads((path/'status.json').read_text())
            if 'exit_code' in status:
                return status
        except (OSError,ValueError):
            pass
        time.sleep(.1)
    raise AssertionError(f'No final exit status in {path}')


@pytest.mark.integration
def test_detached_runtime_failure_lock_and_scoped_stop(tmp_path):
    if not shutil.which('tmux'):
        pytest.skip('tmux unavailable')
    fixture = tmp_path/'runtime_fixture'
    fixture.mkdir()
    shutil.copytree(ROOT/'scripts', fixture/'scripts')
    (fixture/'.venv').symlink_to(ROOT/'.venv', target_is_directory=True)
    (fixture/'configs').mkdir()
    (fixture/'configs/bootstrap.yaml').write_text('mode: bootstrap\nmax_runtime_minutes: 1\n')
    package=fixture/'cross_robot_real2sim'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'cli.py').write_text('''import sys,time
print("EXPLICIT RUNTIME TEST FIXTURE",flush=True)
if "_stop" in " ".join(sys.argv): time.sleep(60)
raise SystemExit(7)
''')
    stamp=str(time.time_ns())
    failed_id='fixture_fail_'+stamp
    stopped_id='fixture_stop_'+stamp
    sessions=['crossrobot_'+failed_id,'crossrobot_'+stopped_id]
    def run(script,*args,check=True):
        return subprocess.run(['bash',str(fixture/'scripts'/script),*args],text=True,capture_output=True,check=check)
    try:
        started=run('launch.sh','bootstrap',failed_id)
        assert 'Run ID:' in started.stdout
        failed=wait_stopped(fixture/'runs'/failed_id)
        assert failed['status']=='failed' and failed['exit_code']==7
        assert 'EXPLICIT RUNTIME TEST FIXTURE' in (fixture/'runs'/failed_id/'logs/pipeline.log').read_text()
        # A disappeared process alone cannot turn a nonzero exit into success.
        output=run('status.sh',failed_id).stdout
        assert '"exit_code": 7' in output
        run('launch.sh','bootstrap',stopped_id)
        runtime=fixture/'runs'/stopped_id/'runtime.json'
        end=time.monotonic()+8
        while time.monotonic()<end:
            if json.loads(runtime.read_text()).get('supervisor_pid') and (runtime.parent/'pipeline.pid').exists(): break
            time.sleep(.1)
        assert run('launch.sh','bootstrap',stopped_id,check=False).returncode!=0
        assert run('launch.sh','bootstrap','fixture_other_'+stamp,check=False).returncode!=0
        run('stop.sh',stopped_id)
        stopped=wait_stopped(runtime.parent)
        assert stopped['status']=='interrupted' and stopped['exit_code']==143
    finally:
        for session in sessions:
            subprocess.run(['tmux','kill-session','-t',session],capture_output=True)
