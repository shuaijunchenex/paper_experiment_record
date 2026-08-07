import sys
from pathlib import Path


_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from entries import run_all


EXPERIMENT_ROOT = (
    Path(__file__).resolve().parent
    / "rbla+"
    / "experiment"
    / "rank_match"
    / "dirichlet_0_4"
)
EXPERIMENT_TARGET_ROOT = "./rbla+/experiment/rank_match/dirichlet_0_4"
TARGET_METHODS_BY_DATASET = {
    "mrpc": ("rbla", "rbla_plus", "sp", "zeropadding"),
    "imdb": ("rbla", "rbla_plus", "sp", "zeropadding"),
    "agnews": ("rbla", "rbla_plus", "sp", "zeropadding"),
    "mnli": ("rbla", "rbla_plus", "sp", "zeropadding"),
}


def collect_configs():
    configs = []
    missing_configs = []
    for dataset, methods in TARGET_METHODS_BY_DATASET.items():
        for method in methods:
            filename = f"{dataset}_dirichlet_0_4_rank_correct_{method}.yaml"
            config_path = EXPERIMENT_ROOT / dataset / filename
            if not config_path.is_file():
                missing_configs.append(config_path)
            configs.append(f"{EXPERIMENT_TARGET_ROOT}/{dataset}/{filename}")

    if missing_configs:
        missing = ", ".join(str(path) for path in missing_configs)
        raise RuntimeError(f"Missing RBLA+ experiment configurations: {missing}")

    normalized = [config.replace("\\", "/") for config in configs]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("Duplicate RBLA+ experiment configuration detected.")

    unexpected = [
        config
        for config, normalized_config in zip(configs, normalized)
        if not normalized_config.startswith(f"{EXPERIMENT_TARGET_ROOT}/")
    ]
    if unexpected:
        raise RuntimeError(
            "AWS entrypoint selected configs outside the experiment root: "
            + ", ".join(unexpected)
        )

    return configs


def main():
    configs = collect_configs()
    print(
        f"Found {len(configs)} Dirichlet-0.4 rank-correct "
        "NLP rerun experiment configurations."
    )
    for config in configs:
        print(f" - {config}")

    run_all.run_all(configs, entry_module="entries.lora")


if __name__ == "__main__":
    main()
