# Sectoral Decarbonisation

This project constructs carbon-reduced portfolios for each GICS sector within an S&P 500 universe and computes a **Decarbonization Readiness Index (DRI)** that measures how feasible it is for each sector to decarbonise without deviating significantly from its benchmark.

The DRI is built from four dimensions:

| Dimension | Description |
|---|---|
| **Room for Maneuver** | Carbon reduction potential at a tight tracking error budget |
| **Flexibility** | Range of near-optimal portfolios at a target tracking error |
| **Sensitivity** | Stability of optimal portfolios under input perturbations |
| **Robustness** | Out-of-sample tracking error relative to benchmark volatility |

---

## Directory Structure

```
sectoral-decarbonisation/
├── 01_download_yahoo_data.py          # Download adjusted prices from Yahoo Finance
├── 02_merge_scope_emissions.py        # Merge & fill Scope 1/2/3 emissions across periods
├── 03_create_datasets.py              # Build benchmark weights & carbon intensity datasets
├── 04_create_optimal_portfolios.py    # Run TE-carbon frontier optimisation per sector
├── 05_benchmark_replication.py        # Replicate benchmark for out-of-sample testing
├── 06_create_room_for_maneuver_score.py
├── 07_create_flexibility_score.py
├── 08_create_sensitivity_score.py
├── 09_create_robustness_score.py
├── 10_create_decarbonization_readiness_score.py  # Combine into composite DRI
├── utils.py                           # Shared optimisation helpers
├── plot_functions.py                  # Shared plotting helpers
├── requirements.txt                   # Python package dependencies
├── FILLING_LOGIC_README.md            # Details on emissions gap-filling logic
├── benchmark_comparison.ipynb         # Validates reconstructed sector benchmarks against official S&P total return indices
├── sector_summaries_all_periods.ipynb # Sector-level imputation and emissions summaries across all periods
├── visualize_te_carbon_frontiers.ipynb # Plots TE-carbon efficient frontiers and marginal gain curves
├── dri_score_corr.ipynb               # Correlates DRI dimension scores computed with Scope 1–3 vs Scope 1–2 emissions
├── real_estate_example_flexibility_0922_0623.py  # Diagnostic: to check why Real Estate flexibility grew between Sep-2022 and Jun-2023
└── data/
    ├── lseg/
    │   ├── constituents_symbols/      # S&P 500 constituent symbols (per period)
    │   ├── prices_dividends/          # Close prices, dividends, free-float shares
    │   └── scope_emissions/           # Scope 1 / 2 / 3 emissions & revenues
    ├── yahoo/                         # Yahoo Finance adjusted prices (fallback)
    ├── merged_scope_emissions/        # Output of step 2 (merged + filled emissions)
    ├── datasets/                      # Output of step 3 (benchmark weights + carbon)
    ├── log_returns/                   # Sector log returns (output of step 3)
    ├── covariances/                   # Sector covariance matrices computed from monthly log returns
    ├── benchmark_returns_volatility/  # Output of step 5: sector daily returns and annualised volatility summaries
    ├── daily_returns_3m/              # Per-period 3-month forward daily returns (output of step 5, used for out-of-sample TE)
    ├── te-testing-results/            # Tracking error test results
    ├── stocks_with_missing_prices/    # Stocks flagged for missing price history
    ├── wiki/symbol_wiki.xlsx          # GICS sector classification from Wikipedia
    ├── gics_sector_manual_mapping.csv # Manual GICS overrides
    ├── scope_emissions_patch.csv      # Manual emissions corrections
    └── delisted_companies_revenues.csv # Used to fill in the missing revenue of those companies that were delisted during the sample period and therefore did not report annual revenues in LSEG, sourced from CompaniesMarketCap.com
```

---

## Time Periods

All scripts operate on quarterly rebalancing dates from March 2021 to December 2023:

```
0321, 0621, 0921, 1221,
0322, 0622, 0922, 1222,
0323, 0623, 0923, 1223
```

Period codes are in `MMYY` format (e.g. `1221` = December 2021).

---

## Pipeline

Run the numbered scripts in order from the `sectoral-decarbonisation/` directory.

### Step 1 — Download Yahoo Finance Data

```bash
python 01_download_yahoo_data.py
```

Downloads split- and dividend-adjusted close prices (total return) from Yahoo Finance for all constituents. These are used exclusively for **return calculations** — not for market cap or weights. The date range covers 2 years before the period (for covariance estimation) through 4 months after (for out-of-sample evaluation). For symbols unavailable on Yahoo, adjusted prices are calculated from LSEG close prices and dividend data as a fallback.

Output: `data/yahoo/adj_price_yahoo_comp_{period}.xlsx`

---

### Step 2 — Merge Scope Emissions

```bash
python 02_merge_scope_emissions.py
```

Reads per-period LSEG emissions files and creates consolidated time-series for Scope 1, Scope 2, and Scope 3 emissions. Also produces `_filled` versions that forward-fill missing values in 2021–2023 (see [FILLING_LOGIC_README.md](FILLING_LOGIC_README.md)).

Output: `data/merged_scope_emissions/scope_{1,2,3}_all_periods{_filled}.xlsx`

---

### Step 3 — Create Datasets

```bash
python 03_create_datasets.py
```

For each period, merges constituent symbols, GICS sectors, prices, free-float shares, and emissions into a single dataset. **Float market cap and intra-sector weights** are computed from LSEG split-adjusted close prices × free-float shares (dividends not included, preserving true market cap). **Monthly log returns** for the covariance matrix are computed separately from Yahoo Finance total-return prices. Carbon intensity is computed as (Scope 1+2+3) / Revenue. Remaining missing scope emissions are imputed using MICE (multiple imputation by chained equations with predictive mean matching via `miceforest`).

Output:
- `data/datasets/benchmark_weights_carbon_intensity_{period}.xlsx`
- `data/log_returns/sector_log_returns_comp_{period}.xlsx`

---

### Step 4 — Portfolio Optimisation

```bash
python 04_create_optimal_portfolios.py
```

For each sector and period, solves a series of quadratic programs (via `cvxpy`) to map out the efficient frontier of **tracking error vs. carbon reduction**:

```
Minimise:  carbon_intensity · weights
Subject to:  tracking_error(weights, benchmark) ≤ TE_target
             weights ≥ 0,  Σ weights = 1
```

Results are cached as pickle files for use in downstream scoring scripts.

---

### Step 5 — Benchmark Replication

```bash
python 05_benchmark_replication.py
```

Replicates the benchmark portfolio to compute out-of-sample returns used by the robustness score.

---

### Steps 6–9 — Dimension Scores

```bash
python 06_create_room_for_maneuver_score.py
python 07_create_flexibility_score.py
python 08_create_sensitivity_score.py
python 09_create_robustness_score.py
```

Each script computes one dimension of the DRI and writes results to `results/<dimension>/`.

---

### Step 10 — Composite DRI

```bash
python 10_create_decarbonization_readiness_score.py
```

Loads the four dimension scores, applies global min–max normalisation, and combines them into the composite Decarbonization Readiness Index. Produces summary plots (radar charts, time-series evolution).

---

## Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Sources

| Source | Used for |
|---|---|
| **LSEG (Refinitiv)** | Benchmark **weights** (float market cap = close price adjusted for corporate splits only and not dividends × free-float shares), carbon intensity, Scope 1/2/3 emissions, revenues |
| **Yahoo Finance** | **Returns** — monthly log returns for covariance estimation (2-year backward window) and daily returns for out-of-sample TE evaluation (3-month forward window), obtained from corporate splits- and dividend-adjusted prices |
| **Wikipedia** | GICS sector classification (`data/wiki/symbol_wiki.xlsx` sourced from official S&P 500 press communications) |

---


