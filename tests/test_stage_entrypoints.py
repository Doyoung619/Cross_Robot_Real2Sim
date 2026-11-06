"""Verify the user-facing five-stage navigation actually routes executable code."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import pytest

ROOT=Path(__file__).resolve().parents[1]
STAGES=[
    ('1_Data Selection',1),
    ('2_Task Understanding',2),
    ('3_Cross-Robot-Retargeting',3),
    ('4_Policy Learning',4),
    ('5_Failure Analysis',5),
]


@pytest.mark.parametrize('folder,step',STAGES)
def test_wrapper_routes_canonical_stage_without_simulation(folder,step,monkeypatch):
    path=ROOT/folder/'run.py'
    spec=importlib.util.spec_from_file_location(f'stage_entry_{step}',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured=[]
    def canonical_main():
        captured.extend(sys.argv)
        return 17
    monkeypatch.setitem(sys.modules,'cross_robot_real2sim.cli',types.SimpleNamespace(main=canonical_main))
    original=sys.argv
    assert module.main(['--config','configs/fast.yaml','--run-id','entry_fixture','--resume'])==17
    assert sys.argv is original
    assert captured[1:5]==['stage','--step',str(step),'--config']
    assert Path(captured[5])==ROOT/'configs/fast.yaml'
    assert captured[6:]==['--run-id','entry_fixture','--resume']
    assert not path.is_symlink()


@pytest.mark.parametrize('folder,step',STAGES)
def test_wrapper_help_works_outside_repository(folder,step,tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/folder/'run.py'),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0
    assert f'Stage {step}' in result.stdout
    for flag in ['--config','--run-id','--resume']:
        assert flag in result.stdout
