import argparse
from pathlib import Path
from time import perf_counter

from .config import ExperimentConfig
from .runner import ExperimentRunner, format_duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoLLM semantic-cache _experiments")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--output-dir", help="Override output.directory")
    parser.add_argument("--run-name", help="Override output.run_name")
    parser.add_argument("--embedding-cache-path", help="Override embedding.cache_path")
    parser.add_argument("--device", help="Override embedding.device, for example cuda or cpu")
    parser.add_argument("--model", help="Override llm.model")
    parser.add_argument("--ollama-host", help="Override llm.host")
    arguments = parser.parse_args()
    pipeline_started = perf_counter()
    status = "failed"
    try:
        config = ExperimentConfig.from_yaml(arguments.config)
        embedding_updates = {}
        llm_updates = {}
        output_updates = {}
        config_updates = {}
        if arguments.embedding_cache_path:
            embedding_updates["cache_path"] = Path(arguments.embedding_cache_path)
        if arguments.device:
            embedding_updates["device"] = arguments.device
        if embedding_updates:
            config_updates["embedding"] = config.embedding.model_copy(
                update=embedding_updates
            )
        if arguments.model:
            llm_updates["model"] = arguments.model
        if arguments.ollama_host:
            llm_updates["host"] = arguments.ollama_host
        if llm_updates:
            config_updates["llm"] = config.llm.model_copy(update=llm_updates)
        if arguments.output_dir:
            output_updates["directory"] = Path(arguments.output_dir)
        if arguments.run_name:
            output_updates["run_name"] = arguments.run_name
        if output_updates:
            config_updates["output"] = config.output.model_copy(update=output_updates)
        if config_updates:
            config = config.model_copy(update=config_updates)
        output = ExperimentRunner.from_config(config).run()
        status = "completed"
        print(output)
    finally:
        pipeline_seconds = perf_counter() - pipeline_started
        print(
            f"Experiment pipeline {status} after {format_duration(pipeline_seconds)} "
            f"({pipeline_seconds:.2f} seconds).",
            flush=True,
        )


if __name__ == "__main__":
    main()
