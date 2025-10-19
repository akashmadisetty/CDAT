# CDAT
CDAT is a small research / teaching repository for exploratory data analysis (EDA) and a transfer-learning framework focused on product-category transferability using the BigBasket product dataset. The goal is to analyze categories, auto-suggest domain pairs for transfer learning experiments, generate domain pair datasets, and provide a practical plan for running experiments and building a transferability framework.

## Contents

- `comp_eda.py` — Comprehensive EDA script that analyzes all categories, computes pairwise similarity scores, auto-selects domain pairs and exports CSVs + visualizations.
- `eda_bb.py` — Focused BigBasket EDA (category-level analysis, visualization, and helper functions to define 4 domain pairs for transfer experiments).
- `plan.md` — Project plan / team work distribution and multi-week roadmap used by the project.
- `helper.txt` — Small notes / helpers (project-specific ancillary text).
- `data/` — Datasets used in the repo
	- `data/original/BigBasket.csv` — (original dataset) main product CSV used by `eda_bb.py` and other scripts
	- `data/original/UK.csv` — (UK retail dataset for later validation)
	- `data/processed/` — processed exports and intermediate datasets (may be created by scripts)
- `processed/` — older processed datasets and utilities used previously (kept for reference)
	- `processed/old_version/` — historical scripts and domain CSVs
- `results/` — generated charts, CSVs and experiment results (output directory)
- `src/` — place for reusable library code (currently empty / for future refactor)

## Typical outputs

- CSVs: category statistics, all-category-pairs similarity, domain pair exports (e.g. `domain_pair1_source.csv`, ...)
- Charts: `category_similarity_heatmap.png`, `all_categories_overview.png`, `price_distributions.png`, `missing_values.png`, etc.

## Quickstart — requirements & run

1. Create a virtual environment and install the core dependencies. The scripts use common data-science packages:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install pandas numpy matplotlib seaborn
```

Optionally create a `requirements.txt` with pinned versions for reproducibility:

```txt
pandas
numpy
matplotlib
seaborn
```

2. Place the BigBasket CSV in the expected path(s):
- `eda_bb.py` expects `data/BigBasket.csv` (relative to repo root)
- `comp_eda.py` expects `BigBasket_Products.csv` in the repository root (this script was written to read a file named `BigBasket_Products.csv`) — if you only have `data/BigBasket.csv`, either copy/rename or edit the script to point to the correct path.

3. Run the quick EDA (examples):

```cmd
python eda_bb.py
```

or the comprehensive analysis:

```cmd
python comp_eda.py
```

Each script will print progress to stdout and save CSVs and PNGs in the repository root (or `results/` depending on the script). Review saved files after the run.

## Notes on inputs & outputs

- Input columns expected (BigBasket dataset): typical scripts assume columns like `category`, `sub_category`, `brand`, `sale_price`, `rating` and other product metadata. If your CSV column names differ, update the scripts or rename columns accordingly.
- Outputs are saved in the repository root by default (CSV + PNG). Consider creating a `results/` folder and moving outputs there for cleanliness.

## Project structure (short)

- Data collection & original files: `data/original/`
- Exploratory scripts: `comp_eda.py`, `eda_bb.py`
- Historical processing: `processed/old_version/` (kept intentionally)
- Plan & documentation: `plan.md`, this `README.md`

## Recommended next steps

- Add a `requirements.txt` and small `Makefile` or `tasks.json` for common runs.
- Add a small CLI wrapper (e.g. `cli.py`) to run analysis with configurable input/output paths.
- Move generated artifacts to `results/` and update scripts to write there.

## Contributing

If you want to contribute or run experiments:

- Open an issue describing the improvement or bug.
- Create a feature branch and submit a pull request with tests where relevant.


## Contact

If you need help running the scripts or want suggestions for experiments, open an issue or contact the repository owner.
Akash Madisetty - @akashmadisetty
---
