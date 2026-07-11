from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(results_directory: str | Path) -> list[Path]:
    directory = Path(results_directory)
    summary_path = directory / "summary.csv"
    dataframe = pd.read_csv(summary_path)
    plots_directory = directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    generated = [
        _line_plot(
            dataframe,
            metric="hit_rate",
            ylabel="Semantic hit rate",
            output=plots_directory / "hit_rate_vs_cache_size.png",
        ),
        _line_plot(
            dataframe,
            metric="mean_hit_response_cosine_distance",
            ylabel="Mean cached-vs-reference response cosine distance",
            output=plots_directory / "response_distance_vs_cache_size.png",
        ),
        _line_plot(
            dataframe,
            metric="mean_latency_ms",
            ylabel="Mean simulated latency (ms)",
            output=plots_directory / "latency_vs_cache_size.png",
        ),
        _line_plot(
            dataframe,
            metric="mean_policy_overhead_ms",
            ylabel="Mean policy overhead (ms)",
            output=plots_directory / "policy_overhead_vs_cache_size.png",
        ),
    ]

    quality_columns = sorted(
        column for column in dataframe.columns if column.startswith("quality_adjusted_hit_rate@")
    )
    if quality_columns:
        metric = quality_columns[len(quality_columns) // 2]
        generated.append(
            _line_plot(
                dataframe,
                metric=metric,
                ylabel=f"Quality-adjusted hit rate ({metric.split('@', 1)[1]})",
                output=plots_directory / "quality_adjusted_hit_rate_vs_cache_size.png",
            )
        )

    generated.append(_pareto_plot(dataframe, plots_directory / "hit_quality_pareto.png"))
    return generated


def _line_plot(dataframe: pd.DataFrame, metric: str, ylabel: str, output: Path) -> Path:
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)
    for policy, group in dataframe.groupby("policy"):
        ordered = group.sort_values("cache_size")
        axis.plot(ordered["cache_size"], ordered[metric], marker="o", label=policy)
    axis.set_xlabel("Cache capacity (entries)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def _pareto_plot(dataframe: pd.DataFrame, output: Path) -> Path:
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)
    for policy, group in dataframe.groupby("policy"):
        axis.scatter(
            group["mean_hit_response_cosine_distance"],
            group["hit_rate"],
            label=policy,
        )
        for _, row in group.iterrows():
            axis.annotate(
                str(int(row["cache_size"])),
                (row["mean_hit_response_cosine_distance"], row["hit_rate"]),
                fontsize=8,
            )
    axis.set_xlabel("Mean cached-vs-reference response cosine distance (lower is better)")
    axis.set_ylabel("Semantic hit rate (higher is better)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output
