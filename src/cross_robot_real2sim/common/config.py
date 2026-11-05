from pathlib import Path
import yaml

def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

def resolve_path(path, root=None) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(root or repo_root()) / p

def load_config(path):
    path = resolve_path(path)
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    config.setdefault("seed", 42)
    config.setdefault("mode", "bootstrap")
    config["config_path"] = str(path.resolve())
    config["repo_root"] = str(repo_root())
    for key, ceiling in [("target_attempt_cap", 100), ("bc_train_steps", 5000), ("held_out_eval_seeds", 20)]:
        if key in config and not 0 < int(config[key]) <= ceiling:
            raise ValueError(f"{key} outside explicit bounded budget (1..{ceiling})")
    if config.get("device", "cpu").startswith("cuda") and config.get("device") != "cuda:0":
        raise ValueError("Only GPU 0 is supported")
    return config
