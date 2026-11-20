"""Equal-update LoRA training with complete optimizer/RNG persistence."""
import json
from pathlib import Path
import resource
import time

import numpy as np

from .core import dump,digest,object_hash


def train(run,data,parent,updates,seed,step,new_start=None,priorities=None):
    import torch
    from safetensors.torch import save_file
    from vla_pilot_learning import load_model,InputContract
    run=Path(run)
    key=object_hash(dict(data=digest(data),parent=digest(parent) if parent else None,updates=updates,
                        seed=seed,step=step,new_start=new_start,priorities=None if priorities is None else np.asarray(priorities).tolist()))
    folder=run/'shared/checkpoints'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():return folder/'adapter.safetensors',json.loads((folder/'training.json').read_text()),True
    wall=time.monotonic();torch.manual_seed(seed+42+step*100003);rng=np.random.default_rng(seed+42+step*100003)
    model=load_model(adapter=True,adapter_path=parent);contract=InputContract(model)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert named and all('lora_' in n for n,p in named) and sum(p.numel() for _,p in named)==334592
    optimizer=torch.optim.AdamW([p for _,p in named],lr=1e-4)
    restored=False
    if parent and (Path(parent).parent/'optimizer.pt').exists():
        optimizer.load_state_dict(torch.load(Path(parent).parent/'optimizer.pt',map_location='cuda',weights_only=False)['optimizer']);restored=True
    with np.load(data) as f:arrays={k:f[k].copy() for k in f.files}
    n=len(arrays['states']);assert arrays['actions'].shape==(n,50,7)
    assert np.isfinite(arrays['actions']).all() and np.max(np.abs(arrays['actions']))<=1.001
    cuda_s=0.;torch.cuda.reset_peak_memory_stats();model.train();logs=[]
    weights=None if priorities is None else np.asarray(priorities,dtype=float)
    if weights is not None:weights=weights/weights.sum()
    cpu0=resource.getrusage(resource.RUSAGE_SELF)
    for i in range(updates):
        if weights is not None:ids=rng.choice(n,size=4,replace=True,p=weights)
        elif new_start is not None and 0<new_start<n:ids=np.r_[rng.integers(0,new_start,2),rng.integers(new_start,n,2)]
        else:ids=rng.integers(0,n,4)
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        batch=contract.batch(arrays['images'][ids],arrays['states'][ids],arrays['actions'][ids])
        batch['actions_id_pad']=torch.as_tensor(arrays['actions_id_pad'][ids],device='cuda',dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model(batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite LoRA loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in named],1.);optimizer.step();end.record();end.synchronize()
        cuda_s+=begin.elapsed_time(end)/1000
        if i==0 or i==updates-1 or i%25==0:logs.append(dict(update=i+1,loss=float(loss.detach())))
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if 'lora_' in k},str(folder/'adapter.safetensors'))
    torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state),folder/'optimizer.pt')
    cpu1=resource.getrusage(resource.RUSAGE_SELF)
    metadata=dict(updates=updates,anchors=n,source_data_sha256=digest(data),seed=seed,step=step,trainable_parameters=334592,
        batch_size=4,learning_rate=1e-4,lora_rank=8,lora_alpha=16,optimizer_restored=restored,
        source_checkpoint='lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d',
        adapter_sha256=digest(folder/'adapter.safetensors'),wall_seconds=time.monotonic()-wall,cuda_update_elapsed_seconds=cuda_s,
        cuda_timing_note='CUDA event elapsed timeline, not an exclusive GPU-active-time measurement',
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
        cpu_user_seconds=cpu1.ru_utime-cpu0.ru_utime,cpu_system_seconds=cpu1.ru_stime-cpu0.ru_stime,
        executed_target_action_supervision=0,loss='masked native SmolVLA flow matching',
        sampling='retrieval_priority' if weights is not None else ('half_old_half_new' if new_start is not None else 'uniform_with_replacement'),logs=logs)
    dump(folder/'training.json',metadata)
    del optimizer,model,contract,named,arrays;torch.cuda.empty_cache()
    return folder/'adapter.safetensors',metadata,False
