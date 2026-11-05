from cross_robot_real2sim.common.config import resolve_path
from cross_robot_real2sim.common.io import read_json
from cross_robot_real2sim.schemas import TaskSpec


def load_seed(config):
    path=resolve_path(config.get("task_seed","data/task_seeds/can_to_bin.json"),config.get("repo_root"))
    raw=read_json(path)
    raw["task_source"]="provided_seed"
    return TaskSpec.model_validate(raw).model_dump(),str(path.resolve())
