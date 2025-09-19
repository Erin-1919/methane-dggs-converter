import csv
from pathlib import Path

# Paths
src = Path("methane_grid_calculation") / "output" / "GFEI_DGGS_methane_emissions_ALL_FILES.csv"
dst = Path("methane_grid_calculation") / "output" / "GFEI_DGGS_methane_emissions_sample.csv"

# Read header and first five rows, then write them out
with src.open("r", newline="", encoding="utf-8") as f_in, dst.open("w", newline="", encoding="utf-8") as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    try:
        header = next(reader)
    except StopIteration:
        # Source file is empty, nothing to write
        pass
    else:
        writer.writerow(header)
        for i, row in enumerate(reader):
            writer.writerow(row)
            if i == 4:
                break

print(f"Saved sample to {dst}")