"""CLI and a detached whole-run supervisor with an explicit deadline."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .core import METHODS,bootstrap,digest,dump,object_hash,validate


ROOT=Path(__file__).resolve().parents[3]


def source_hashes():
    import hashlib,inspect
    names=list((ROOT/'src/cross_robot_real2sim/benchmark').glob('*.py'))
    names += [ROOT/'scripts'/n for n in ['vla_pilot_common.py','vla_pilot_learning.py','vla_feedback_processing.py']]
    # Only tracked legacy algorithms are dependencies; local humanoid work is
    # snapshotted in benchmark/gr1.py, and is never staged or overwritten.
    result={str(p.relative_to(ROOT)):digest(p) for p in sorted(names)}
    import vla_pilot_processing as processing
    import vla_feedback_rollouts as rollouts
    for module,functions in [(processing,['resolve_xml','source_model','restore','dataset_pose','copy_scene','raw_target','ik','canonical_state','unintended_contacts','camera_calibration']),
                             (rollouts,['_contact_ids','_contact_measurement','_aggregate','diagnose'])]:
        for name in functions:result[f'scripts/{module.__name__}.py::{name}']=hashlib.sha256(inspect.getsource(getattr(module,name)).encode()).hexdigest()
    return result


def check_protocol(run):
    p=json.loads((run/'protocol.json').read_text());config=json.loads((run/'config.json').read_text())
    if p['config_sha256']!=object_hash(config):raise RuntimeError('Configuration changed after preregistration')
    if p['code_sha256']!=source_hashes():raise RuntimeError('Code changed after preregistration; create a new run, never tune against its final evaluation')
    for name,sha in p['state_files'].items():
        if digest(run/'states'/name)!=sha:raise RuntimeError('Sealed state bank changed')
    for name,sha in p['shared_assets'].items():
        if digest(run/'shared'/name)!=sha:raise RuntimeError('Shared camera/robot asset changed')
    return config


def progress(run,stage,details):
    event=dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,details=details)
    with (run/'progress.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)


def prepare(run,config,supervised=False):
    if run.exists() and not supervised:raise FileExistsError('Use a new run directory; old experiments are never overwritten')
    if supervised and ((run/'config.json').exists() or (run/'protocol.json').exists()):raise FileExistsError('Prepared run is never overwritten')
    run.mkdir(parents=True,exist_ok=supervised);(run/'shared').mkdir();dump(run/'config.json',config)
    from .environment import make_banks
    from .data import SourcePool
    start=time.monotonic();make_banks(run,config);pool=SourcePool(run)
    try:pool.inspect()
    finally:pool.close()
    from vla_pilot_common import MATERIALS,DATASET
    import torch
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),config_sha256=object_hash(config),code_sha256=source_hashes(),
        state_files={p.name:digest(p) for p in sorted((run/'states').glob('*'))},
        shared_assets={str(p.relative_to(run/'shared')):digest(p) for p in [run/'shared/camera_calibration.npz',run/'shared/gr1_model/robot.xml']},
        source_hdf5_sha256=digest(DATASET),pretrained_weight_sha256=digest(MATERIALS/'model/model.safetensors'),
        model='lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d',
        hardware=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__),
        common_target='GR1 right arm + Robotiq85; all other joints removed at unchanged fixed poses',
        initial_controller='OSC_POSE20Hz,50 predicted actions,5 executed,10 flow steps',
        selection_rule='No final evaluation model/hyperparameter selection; fixed last checkpoint at each allowed K',
        development_rollout_budget=0,preparation_wall_seconds=time.monotonic()-start,
        methods={n:asdict(METHODS[n]) for n in config['methods']})
    dump(run/'protocol.json',protocol);progress(run,'prepared',dict(state_counts=json.loads((run/'states/manifest.json').read_text())['counts']))


def execute(run,deadline=None):
    config=check_protocol(run)
    from .experiment import adapt
    from .evaluation import evaluate
    from .report import export
    if not (run/'TRAINING_COMPLETE.json').exists():
        if (run/'rollouts.jsonl').exists() and any(json.loads(x)['event']=='started' for x in (run/'rollouts.jsonl').read_text().splitlines() if x.strip()):
            raise RuntimeError('Interrupted adaptation contains counted interactions; unsafe implicit replay is refused. Preserve this run and start a separately named run.')
        adapt(run,config,lambda s,d:progress(run,s,d))
    check_protocol(run)
    evaluate(run,config,lambda s,d:progress(run,s,d),deadline)
    output=export(run);progress(run,'exported',dict(path=str(output)))


def supervise(run,config_path,overrides):
    if run.exists():raise FileExistsError(run)
    # A suite-wide lock prevents accidentally overlapping benchmark GPU jobs.
    lock=(ROOT/'runs/benchmark_gpu.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another benchmark supervisor holds the GPU lease')
    config=validate(json.loads(Path(config_path).read_text()));config.update(overrides);validate(config)
    deadline=time.time()+config['max_hours']*3600
    run.mkdir(parents=True);dump(run/'launch_config.json',config)
    status=dict(status='running',supervisor_pid=os.getpid(),deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),started_at=datetime.now(timezone.utc).isoformat())
    dump(run/'status.json',status)
    # The worker prepares into the existing supervisor folder by using a child
    # temporary directory, then promotes artifacts without replacing old data.
    cmd=[sys.executable,'-u','-m','cross_robot_real2sim.benchmark','worker','--run-dir',str(run),'--deadline',str(deadline)]
    with (run/'worker.log').open('a') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        status['worker_pid']=child.pid;dump(run/'status.json',status)
        while child.poll() is None and time.time()<deadline:time.sleep(2)
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            status['status']='time_budget_exhausted'
        else:status['status']='completed' if child.returncode==0 else 'worker_failed'
    status.update(exit_code=child.returncode,finished_at=datetime.now(timezone.utc).isoformat());dump(run/'status.json',status)
    if (run/'config.json').exists():
        try:
            from .report import export
            export(run)
        except Exception:dump(run/'export_error.json',dict(error=traceback.format_exc()))
    print(json.dumps(status),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['list','prepare','run','worker','supervise','evaluate','report'])
    p.add_argument('--run-dir');p.add_argument('--config',default='configs/benchmark/full.json')
    p.add_argument('--methods',nargs='+');p.add_argument('--budgets',type=int,nargs='+');p.add_argument('--evaluation-states',type=int)
    p.add_argument('--deadline',type=float);args=p.parse_args()
    if args.command=='list':print(json.dumps({n:asdict(v) for n,v in METHODS.items()},indent=2));return
    if not args.run_dir:p.error('--run-dir required')
    run=Path(args.run_dir).resolve();overrides={}
    for key in ['methods','budgets','evaluation_states']:
        value=getattr(args,key)
        if value is not None:overrides[key]=value
    bootstrap()
    if args.command=='supervise':supervise(run,args.config,overrides);return
    if args.command=='worker':
        config=json.loads((run/'launch_config.json').read_text())
        try:
            prepare(run,config,supervised=True);execute(run,args.deadline)
        except Exception:
            dump(run/'worker_error.json',dict(error=traceback.format_exc()));raise
    elif args.command=='prepare':
        c=json.loads(Path(args.config).read_text());c.update(overrides);prepare(run,validate(c))
    elif args.command=='run':execute(run,args.deadline)
    elif args.command=='evaluate':
        from .evaluation import evaluate
        from .report import export
        c=check_protocol(run);evaluate(run,c,lambda s,d:progress(run,s,d),args.deadline);export(run)
    else:
        from .report import export
        print(export(run))


if __name__=='__main__':main()
