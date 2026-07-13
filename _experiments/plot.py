import argparse

from .plotting import generate_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots from summary.csv")
    parser.add_argument("--results-dir", required=True)
    arguments = parser.parse_args()
    for path in generate_plots(arguments.results_dir):
        print(path)


if __name__ == "__main__":
    main()
