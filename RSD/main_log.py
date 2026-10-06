import pandas as pd

# Save processed metrics
df_metrics = pd.DataFrame.from_dict(results_metrics, orient='index')
df_metrics.reset_index(inplace=True)
df_metrics.rename(columns={"index": "method"}, inplace=True)

df_metrics.to_csv("data/processed/metrics.csv", index=False)

print("[Saved] Processed metrics → data/processed/metrics.csv")