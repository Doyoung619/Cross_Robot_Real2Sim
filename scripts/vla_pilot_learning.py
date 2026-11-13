"""Pretrained source VLA with small LoRA updates; no executed B training data."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from vla_pilot_common import MATERIALS, TASK, write_json


def load_model(adapter=False, adapter_path=None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    torch.set_num_threads(2)
    config = PreTrainedConfig.from_pretrained(MATERIALS/"model")
    config.vlm_model_name = str(MATERIALS/"backbone")
    config.load_vlm_weights = False
    config.device = "cpu"
    config.train_expert_only = True
    config.freeze_vision_encoder = True
    config.train_state_proj = False
    config.n_action_steps = 5
    config.num_steps = 10
    model = SmolVLAPolicy(config)
    base = load_file(str(MATERIALS/"model/model.safetensors"))
    model.load_state_dict(base, strict=True)
    del base
    model.requires_grad_(False)
    if adapter:
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(r=8, lora_alpha=16,
                         target_modules=r".*(?:lm_expert\..*\.(?:q_proj|v_proj)|action_in_proj|action_out_proj)",
                         lora_dropout=0, bias="none")
        model = inject_adapter_in_model(cfg, model)
        if adapter_path:
            weights = load_file(str(adapter_path))
            actual = model.state_dict()
            if not weights or any(k not in actual or "lora_" not in k for k in weights):
                raise ValueError("Invalid LoRA-only checkpoint")
            actual.update(weights)
            model.load_state_dict(actual, strict=True)
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    return model


class InputContract:
    def __init__(self, model):
        norm = load_file(str(MATERIALS/"model/policy_preprocessor_step_5_normalizer_processor.safetensors"))
        self.mean = {k:norm[k+".mean"].cuda() for k in ("observation.state","action")}
        self.std = {k:norm[k+".std"].cuda().clamp_min(1e-6) for k in ("observation.state","action")}
        tokens = model.model.vlm_with_expert.processor.tokenizer(
            [TASK+"\n"], return_tensors="pt", padding="max_length", max_length=48,truncation=True)
        self.tokens = tokens["input_ids"].cuda()
        self.mask = tokens["attention_mask"].bool().cuda()

    def batch(self, images, states, actions=None):
        images = torch.as_tensor(images, device="cuda")
        # Required by LeRobot's official LiberoProcessorStep: rotate both
        # image axes by 180 degrees before the policy's image preprocessing.
        images = torch.flip(images, dims=[2,3])
        states = torch.as_tensor(states, device="cuda",dtype=torch.float32)
        n = len(states)
        batch = {"observation.state":((states-self.mean["observation.state"])/self.std["observation.state"]).to(torch.bfloat16),
                 "observation.language.tokens":self.tokens.expand(n,-1),
                 "observation.language.attention_mask":self.mask.expand(n,-1)}
        for cam in range(2):
            batch[f"observation.images.camera{cam+1}"] = images[:,cam].permute(0,3,1,2).to(torch.bfloat16)/255
        if actions is not None:
            a = torch.as_tensor(actions,device="cuda",dtype=torch.float32)
            batch["action"] = ((a-self.mean["action"])/self.std["action"]).to(torch.bfloat16)
        return batch

    def decode(self, prediction):
        return (prediction.float()*self.std["action"]+self.mean["action"]).cpu().numpy()


def train(out, progress, steps=100):
    start = time.monotonic()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = load_model(adapter=True)
    contract = InputContract(model)
    with np.load(out/"processed_source_data.npz") as f:
        images = f["images"].copy()
        states = f["states"].copy()
        actions = f["actions"].copy()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=1e-4)
    model.train()
    log = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        ids = rng.integers(0,len(states),4)
        batch = contract.batch(images[ids],states[ids],actions[ids])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,_ = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite adapter training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        if step % 10 == 0 or step == steps-1:
            row = dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start)
            log.append(row)
            progress("training",row)
            write_json(out/"train_log.json",log)
    weights = {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if "lora_" in k}
    save_file(weights,str(out/"adapter.safetensors"))
    result = dict(status="completed",steps=steps,training_frames=len(states),batch_size=4,
                  trainable_parameters=sum(p.numel() for p in parameters),rank=8,learning_rate=1e-4,
                  source_checkpoint="lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d",
                  normalization="exact_saved_source_checkpoint_normalizer",training_origin="source_derived_static_kinematic_render_only",
                  executed_target_training_episodes=0,duration_seconds=time.monotonic()-start,
                  adapter_sha256=hashlib.sha256((out/"adapter.safetensors").read_bytes()).hexdigest(),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  final_training_loss=log[-1]["loss"],source_only_validation="not_performed_in_minimal_compute_smoke")
    write_json(out/"training.json",result)
    del optimizer,model,parameters
    torch.cuda.empty_cache()
    return result
