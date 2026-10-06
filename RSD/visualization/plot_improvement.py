# visualization/plot_improvement.py

import matplotlib.pyplot as plt


class PlotImprovement:
    """
    Generates improvement percentage plot.
    """

    def __init__(self):
        pass

    def compute_improvements(self, summary_df):
        """
        Extract improvement values from summary table.
        """

        metrics = summary_df["Metric"].tolist()
        improvements = summary_df["Improvement (%)"].tolist()

        return metrics, improvements

    def plot(self, summary_df):
        """
        Plot improvement percentages.
        """

        metrics, improvements = self.compute_improvements(summary_df)

        plt.figure(figsize=(8, 5))
        plt.bar(metrics, improvements)

        plt.xlabel("Metrics")
        plt.ylabel("Improvement (%)")
        plt.title("Performance Improvement over Best Baseline")

        plt.xticks(rotation=30)

        plt.grid(axis="y")

        plt.tight_layout()
        plt.savefig("improvement_plot.png", dpi=600)
        plt.show()