#!/usr/bin/env python3
"""Real machine, source HDF5 and installed simulator checks; never synthetic pass flags."""
from __future__ import annotations
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/bootstrap.yaml'))
    parser.add_argument('--output-dir', default=str(ROOT / 'runs/preflight'))
    parser.add_argument('--smoke-worker', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from cross_robot_real2sim.common.config import load_config
    config = load_config(args.config)
    if args.smoke_worker:
        from cross_robot_real2sim.adapters.robosuite_env import smoke_test
        result = smoke_test(config, out)
        write(out / 'smoke.json', result)
        print(json.dumps(result, indent=2))
        return 0
    record = {'schema_version': '1.0', 'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.version, 'executable': sys.executable, 'config': str(Path(args.config).resolve()),
              'mode': config.get('mode'), 'seed': config.get('seed'), 'privileged_state_used': True,
              'rendering_attempts': [], 'status': 'running'}
    try:
        disk = shutil.disk_usage(ROOT)
        record['disk'] = {'free_bytes': disk.free, 'total_bytes': disk.total}
        if disk.free < 5 * 1024**3:
            raise RuntimeError('Less than 5GiB free; refuse new rollout artifacts.')
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free',
                              '--format=csv'], capture_output=True, text=True, check=True)
        record['gpu'] = gpu.stdout.strip()
        record['gpu_processes'] = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                                  '--format=csv'], capture_output=True, text=True).stdout.strip()
        import torch
        record['torch'] = {'version': torch.__version__, 'cuda_runtime': torch.version.cuda,
                           'cuda_available': torch.cuda.is_available()}
        if config.get('device', 'cpu').startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Configuration requests CUDA, but torch.cuda.is_available() is false.')
        packages = ['robosuite', 'mujoco', 'numpy', 'h5py', 'scipy', 'numba', 'pydantic', 'huggingface_hub', 'imageio']
        record['packages'] = {name: importlib.metadata.version(name) for name in packages}
        from cross_robot_real2sim.adapters.source_hdf5 import ensure_dataset
        dataset_path = Path(ensure_dataset(config))
        import h5py
        with h5py.File(dataset_path, 'r') as source:
            demo_names = sorted(source['data'].keys(), key=lambda name: int(name.split('_')[-1]))
            first = source['data'][demo_names[0]]
            obs = first.get('obs')
            record['source_dataset'] = {'path': str(dataset_path), 'bytes': dataset_path.stat().st_size,
                'root_keys': list(source.keys()), 'data_attributes': {k: str(v) for k,v in source['data'].attrs.items()},
                'demo_count': len(demo_names), 'first_demo': demo_names[0], 'first_demo_keys': list(first),
                'actions_shape': list(first['actions'].shape),
                'obs_shapes': {k: list(v.shape) for k,v in obs.items()} if obs else {}}
        for backend in ['egl', 'osmesa']:
            env = os.environ.copy()
            env.update(MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
            command = [sys.executable, str(Path(__file__).resolve()), '--smoke-worker', '--config', args.config,
                       '--output-dir', str(out / backend)]
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
            (out / f'smoke-{backend}.log').write_text(result.stdout + '\n' + result.stderr)
            record['rendering_attempts'].append({'backend': backend, 'exit_code': result.returncode,
                                                'log': str(out / f'smoke-{backend}.log')})
            if result.returncode == 0:
                record['simulator'] = json.loads((out / backend / 'smoke.json').read_text())
                record['renderer_status'] = 'hardware_egl' if backend == 'egl' else 'software_osmesa_fallback'
                break
            print(f'{backend} smoke failed: {result.stderr[-2000:]}', file=sys.stderr, flush=True)
        else:
            raise RuntimeError('Both EGL and OSMesa smoke tests failed; inspect their logs.')
        record['status'] = 'completed'
        write(out / 'environment.json', record)
        print(json.dumps(record, indent=2), flush=True)
        return 0
    except Exception as exc:
        record.update(status='failed', exception=repr(exc), traceback=traceback.format_exc())
        write(out / 'environment.json', record)
        print(json.dumps(record, indent=2), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
