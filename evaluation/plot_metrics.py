import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import RESULTS_CSV


def main(args):
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Missing results file: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    df = df[df["metric"].isin(args.metrics)]

    os.makedirs(args.output_dir, exist_ok=True)
    for metric in args.metrics:
        sub = df[df["metric"] == metric]
        plt.figure(figsize=(10, 5))
        sns.barplot(data=sub, x="task", y="value", hue="method")
        plt.title(f"{metric} comparison")
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"{metric}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "f1_macro"])
    parser.add_argument("--output_dir", type=str, default="outputs/plots")
    args = parser.parse_args()

    main(args)
