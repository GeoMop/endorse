import pandas as pd



for i in range(5):
    df = pd.read_csv(f"P30_alpha_pop{i+1}.csv")
    df["alpha"] = df["alpha"] - 1
    df.to_csv(f"P30_alpha_minus_pop{i+1}.csv", index=False, float_format="%.15e")