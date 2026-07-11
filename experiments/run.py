import argparse

from .config import ExperimentConfig
from .runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoLLM semantic-cache experiments")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    arguments = parser.parse_args()
    config = ExperimentConfig.from_yaml(arguments.config)
    output = ExperimentRunner.from_config(config).run()
    print(output)


if __name__ == "__main__":
    main()
