import pandas as pd
from pathlib import Path

# Change this if your file lives somewhere else
csv_path = Path("methane_grid_calculation") / "output" / "GFEI_DGGS_methane_emissions_ALL_FILES.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Find rows that are not unique by the pair dggsID and Year
dupe_mask = df.duplicated(subset=["dggsID", "Year"], keep=False)
dupes = df.loc[dupe_mask].copy().sort_values(["dggsID", "Year"])

# Summarize duplicate pairs
summary = (
    dupes.groupby(["dggsID", "Year"])
         .size()
         .reset_index(name="count")
         .query("count > 1")
         .sort_values(["dggsID", "Year"])
)

# Report
if summary.empty:
    print("No duplicates found for the combination dggsID and Year.")
else:
    print(f"Duplicate pairs found: {len(summary)}")
    # print(summary.head(20).to_string(index=False))

    # If you want to save the full list, uncomment the next line
    # summary.to_csv("duplicates_dggsID_Year.csv", index=False)




