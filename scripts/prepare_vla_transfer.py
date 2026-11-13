#!/usr/bin/env python3
"""Download pinned, bounded materials for a real pretrained-VLA transfer audit.

This prepares source materials only; it does not claim source-policy success,
retargeted target demonstrations, or successful cross-robot fine-tuning.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/vla_readiness_20261113"
MODEL = "lerobot/smolvla_libero"
MODEL_REV = "a24f1f380461b665158d52acd140152c79e8b36d"
DATASET = "lerobot/libero"
DATASET_REV = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def download(repo, revision, filename, kind="model"):
    prefix = "datasets/" if kind in ("dataset", "asset") else ""
    url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{filename}"
    folder = {"dataset": "dataset", "backbone": "backbone", "asset": "assets"}.get(kind, "model")
    target = OUT / folder / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for block in response.iter_content(4 * 1024 * 1024):
                    stream.write(block)
        temporary.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    result = dict(repo=repo, revision=revision, filename=filename, url=url,
                  path=str(target.relative_to(ROOT)), bytes=target.stat().st_size,
                  sha256=digest.hexdigest(), seconds=time.monotonic()-started)
    print(json.dumps(result), flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(MODEL, MODEL_REV, name, "model") for name in (
        "README.md", "config.json", "train_config.json", "model.safetensors",
        "policy_preprocessor.json", "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors")]
    jobs += [(DATASET, DATASET_REV, name, "dataset") for name in (
        "README.md", "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet")]
    jobs.append(("yifengzhu-hf/LIBERO-datasets",
                 "f13aa24a3da8c43c7225569f28c562979fa0e35a",
                 "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
                 "dataset"))
    jobs.append(("yifengzhu-hf/LIBERO-datasets",
                 "f13aa24a3da8c43c7225569f28c562979fa0e35a",
                 "libero_goal/push_the_plate_to_the_front_of_the_stove_demo.hdf5",
                 "dataset"))
    backbone_api = requests.get(
        "https://huggingface.co/api/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        "/revision/4cde9d208e19519c36917aa170fffc7e381aa0ad?blobs=true", timeout=20).json()
    jobs += [("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
              "4cde9d208e19519c36917aa170fffc7e381aa0ad", item["rfilename"], "backbone")
             for item in backbone_api.get("siblings", [])
             if "/" not in item["rfilename"] and item["rfilename"].endswith((".json", ".txt"))]
    assets_api = requests.get(
        "https://huggingface.co/api/datasets/lerobot/libero-assets"
        "/revision/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6?blobs=true", timeout=20).json()
    jobs += [("lerobot/libero-assets", "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
              item["rfilename"], "asset") for item in assets_api.get("siblings", [])
             if item["rfilename"] != ".gitattributes"]
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(download, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(dict(job=futures[future], error=repr(exc)))
            (OUT / "download_manifest.json").write_text(json.dumps(
                dict(files=results, errors=errors), indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
