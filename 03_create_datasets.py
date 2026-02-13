"""
Dataset creation script.

This script creates standardized datasets for multiple time periods:
1. Benchmark weights and carbon intensity DataFrame (GICS Sector, Carbon Intensity, weight_in_sector)
2. Sector log returns for portfolio optimization

Time periods supported: 1221, 0322, 0622, 0922, 1222
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import miceforest as mf


class DatasetCreator:
    """
    Creates standardized datasets for a single S&P 500 rebalancing period.

    The pipeline proceeds as follows:
      1. Load constituent symbols and GICS sectors (LSEG + Wikipedia)
      2. Load end-of-period prices and free-float shares (LSEG)
      3. Calculate float market capitalisation
      4. Remove duplicate tickers (dual-class shares)
      5. Load Scope 1/2/3 emissions and revenue (merged across periods by script 02)
      6. Impute any remaining missing revenue from a delisted-company lookup
      7. Impute residual missing scope emissions with MICE (PMM)
      8. Remove stocks that lack a full two-year price history before the period
      9. Compute intra-sector market-cap weights
      10. Compute monthly log returns from Yahoo adjusted prices
      11. Save benchmark/carbon dataset and per-sector log-return sheets

    Key attributes set in __init__:
        period (str): Period code, e.g. '1221' for December 2021.
        data_dir (Path): Root directory for all input/output data.
        output_dir (Path): data/datasets/ — benchmark weight files.
        log_returns_dir (Path): data/log_returns/ — sector return files.
    """

    def __init__(self, period: str, data_dir: str = "data"):
        """
        Initialize DatasetCreator for a specific period

        Args:
            period: Time period code (e.g., '1221', '0322', '0622', '0922', '1222')
            data_dir: Root data directory path
        """
        self.period = period
        self.data_dir = Path(data_dir)

        # Define file paths
        self.symbol_file = self.data_dir / "lseg" / "constituents_symbols" / f"symbol_comp_{period}.xlsm"
        self.price_div_file = self.data_dir / "lseg" / "prices_dividends" / f"price_div_comp_{period}.xlsm"
        self.carbon_file = self.data_dir / "lseg" / "scope_emissions" / f"carbon_int_comp_{period}.xlsm"
        self.adj_price_file = self.data_dir / "yahoo" / f"adj_price_yahoo_comp_{period}.xlsx"
        self.wiki_file = self.data_dir / "wiki" / "symbol_wiki.xlsx"
        self.patch_file = self.data_dir / "scope_emissions_patch.csv"
        self.delisted_revenue_file = self.data_dir / "delisted_companies_revenues.csv"
        self.missing_prices_file = self.data_dir / "stocks_with_missing_prices" / f"stocks_with_missing_prices_{period}.xlsx"

        # Output paths
        self.output_dir = self.data_dir / "datasets"
        self.log_returns_dir = self.data_dir / "log_returns"

        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_returns_dir.mkdir(parents=True, exist_ok=True)

    def load_symbol_data(self):
        """
        Load constituent symbols, LSEG type codes, and GICS sectors.

        Reads ticker and company-name data from the LSEG symbol file, then
        merges GICS sector labels from a pre-built Wikipedia mapping file.
        Any companies not covered by the Wikipedia file are filled from a
        manual mapping CSV (data/gics_sector_manual_mapping.csv) if present.

        Two hardcoded symbol corrections are applied after the merge:
        - "BERKSHIRE HATHAWAY 'B'" uses the Yahoo Finance ticker "BRK-B"
          (LSEG stores it without the hyphen).
        - "BF.B" is renamed to "BF-B" for the same reason (dot not valid
          in Yahoo tickers).

        Returns:
            pd.DataFrame: Columns NAME, SYMBOL, TYPE, GICS Sector.
        """
        print(f"  Loading symbol data from {self.symbol_file.name}...")

        # Read type and symbol info from LSEG file (following original notebook logic)
        type_lseg = pd.read_excel(self.symbol_file, sheet_name="SYMBOL", dtype=str).iloc[0].values[1:]
        symbol_lseg = pd.read_excel(self.symbol_file, sheet_name="SYMBOL", dtype=str, header=2)
        symbol_lseg = symbol_lseg.iloc[:1].transpose().reset_index().rename(
            columns={"index": "NAME", 0: "SYMBOL"}
        ).iloc[1:]
        symbol_lseg['TYPE'] = type_lseg

        # Read GICS sectors from Wikipedia file
        print(f"  Loading GICS sectors from {self.wiki_file.name}...")
        symbol_wiki = pd.read_excel(self.wiki_file)

        # Merge symbol data with GICS sectors
        df = pd.merge(
            symbol_lseg,
            symbol_wiki[['Symbol', 'GICS Sector']],
            how='left',
            left_on='SYMBOL',
            right_on='Symbol'
        )

        # Drop the redundant 'Symbol' column from the merge
        df = df.drop(columns=['Symbol'])

        # Apply manual GICS sector mappings for companies not found in Wiki
        manual_mapping_file = self.data_dir / "gics_sector_manual_mapping.csv"
        if manual_mapping_file.exists():
            manual_mapping = pd.read_csv(manual_mapping_file)
            # Only fill in missing GICS sectors
            for _, row in manual_mapping.iterrows():
                company_name = row['NAME']
                sector = row['GICS Sector']
                # Update only if GICS Sector is missing
                df.loc[(df['GICS Sector'].isna()) & (df['NAME'] == company_name), 'GICS Sector'] = sector

            filled_count = (df['GICS Sector'].notna()).sum() - (symbol_wiki['Symbol'].isin(df['SYMBOL'])).sum()
            if filled_count > 0:
                print(f"    Applied {filled_count} manual GICS sector mappings")

        print(f"  Dataset shape: {df.shape}")
        print(f"  Companies with GICS sector: {df['GICS Sector'].notna().sum()}/{len(df)}")
        df.loc[(df['NAME'] == "BERKSHIRE HATHAWAY 'B'"), 'SYMBOL'] = 'BRK-B'
        df.loc[(df['SYMBOL'] == "BF.B"), 'SYMBOL'] = 'BF-B'
        return df

    def delete_duplicates(self, df):
        """
        Remove duplicates based on SYMBOL, keeping only entries with " A" or " 'A'" in NAME

        Args:
            df: DataFrame with SYMBOL and NAME columns

        Returns:
            DataFrame with duplicates removed

        Raises:
            ValueError: If a pair of duplicates doesn't have a case with " A" or " 'A'" in NAME
        """
        print(f"  Checking for duplicate SYMBOLs...")

        # Find duplicated symbols
        duplicated_symbols = df[df['SYMBOL'].duplicated(keep=False)]['SYMBOL'].unique()

        if len(duplicated_symbols) == 0:
            print(f"    No duplicates found")
            return df

        print(f"    Found {len(duplicated_symbols)} duplicate SYMBOL(s): {', '.join(duplicated_symbols)}")

        # Process each duplicated symbol
        rows_to_keep = []

        for symbol in duplicated_symbols:
            duplicates = df[df['SYMBOL'] == symbol]

            # Check for entries with " A" or " 'A'" at the end of NAME
            # Pattern matches: " A" at end, or " 'A'" at end (with optional closing quote)
            has_a_suffix = duplicates['NAME'].str.contains(r" A$| 'A'$", regex=True, na=False)

            if has_a_suffix.sum() == 0:
                # No entry has " A" or " 'A'" - raise error
                names = duplicates['NAME'].tolist()
                raise ValueError(
                    f"Duplicate SYMBOL '{symbol}' found but none of the entries have ' A' or \" 'A'\" in NAME. "
                    f"Names found: {names}"
                )

            # Keep only the entries with " A" or " 'A'"
            rows_with_a = duplicates[has_a_suffix]
            rows_to_keep.append(rows_with_a)

            print(f"      {symbol}: Keeping {len(rows_with_a)} row(s) with ' A' or \" 'A'\", "
                  f"removing {len(duplicates) - len(rows_with_a)} row(s)")

        # Combine rows to keep with non-duplicated rows
        non_duplicated = df[~df['SYMBOL'].isin(duplicated_symbols)]
        df_cleaned = pd.concat([non_duplicated] + rows_to_keep, ignore_index=True)

        print(f"    Removed {len(df) - len(df_cleaned)} duplicate row(s)")
        return df_cleaned

    def calculate_float_mcap(self, df):
        """
        Calculate float market capitalization for each company

        Args:
            df: DataFrame with price and ffnosh columns

        Returns:
            DataFrame with float_mcap column added
        """
        print(f"  Calculating float market capitalization...")

        # Get column names for the current period
        price_col = f'Price last day {self._format_period()}'
        ffnosh_col = f'ffnosh last day {self._format_period()}'

        # Check if required columns exist
        if price_col not in df.columns or ffnosh_col not in df.columns:
            raise ValueError(
                f"Required columns not found. Expected '{price_col}' and '{ffnosh_col}'"
            )

        # Calculate float market cap (price * free float shares)
        df['float_mcap'] = pd.to_numeric(df[price_col], errors='coerce') * \
                           pd.to_numeric(df[ffnosh_col], errors='coerce')

        # Report statistics
        valid_mcap = df['float_mcap'].notna().sum()
        print(f"    Companies with valid float_mcap: {valid_mcap}/{len(df)}")

        if valid_mcap > 0:
            total_mcap = df['float_mcap'].sum()
            print(f"    Total float market cap: ${total_mcap:,.0f}")

        return df

    def load_price_and_shares(self, symbols_df):
        """
        Load end-of-period closing prices and free-float shares outstanding.

        Reads the CLOSE PRICE and FFNOSH sheets from the LSEG price/dividend
        file, filters to the last trading day of the target month, and merges
        both series onto the symbol DataFrame. Where a company appears under
        multiple share classes (dual listings), ffnosh values are summed and
        prices are weighted-averaged by ffnosh so each ticker has a single
        combined entry.

        Args:
            symbols_df (pd.DataFrame): Output of load_symbol_data(); must
                contain NAME, SYMBOL, and TYPE columns.

        Returns:
            pd.DataFrame: Input DataFrame extended with
                'Price last day <period>' and 'ffnosh last day <period>'.
        """
        print(f"  Loading price and shares data from {self.price_div_file.name}...")

        # Load close prices - company names are in columns (header row)
        price = pd.read_excel(self.price_div_file, sheet_name="CLOSE PRICE", header=3)
        # Drop the 'Code' row (first row after header)
        price = price.iloc[1:]

        # Convert date column to datetime
        price['Name'] = pd.to_datetime(price['Name'], errors='coerce')

        # Get the target year-month from period (e.g., '1221' -> 2021-12)
        year = int("20" + self.period[2:])
        month = int(self.period[:2])

        # Filter for the target month and year
        price_filtered = price[(price['Name'].dt.year == year) & (price['Name'].dt.month == month)]

        if len(price_filtered) == 0:
            raise ValueError(f"No price data found for period {year}-{month:02d}")

        # Get last price for the last day of the target month
        last_price_row = price_filtered.iloc[-1]
        actual_date = last_price_row['Name']
        print(f"    Using price data from: {actual_date.strftime('%Y-%m-%d')}")

        # Create dataframe with company names and their last prices
        last_price = pd.DataFrame({
            'NAME': price.columns[1:],  # Skip 'Name' column
            f'Price last day {self._format_period()}': last_price_row.values[1:]
        })

        # Load float shares (ffnosh)
        ffnosh = pd.read_excel(self.price_div_file, sheet_name="FFNOSH", header=3)
        # Drop the 'Code' row
        ffnosh = ffnosh.iloc[1:]

        # Convert date column to datetime
        ffnosh['Name'] = pd.to_datetime(ffnosh['Name'], errors='coerce')

        # Filter for the target month and year
        ffnosh_filtered = ffnosh[(ffnosh['Name'].dt.year == year) & (ffnosh['Name'].dt.month == month)]

        if len(ffnosh_filtered) == 0:
            raise ValueError(f"No ffnosh data found for period {year}-{month:02d}")

        # Get last ffnosh for the last day of the target month
        last_ffnosh_row = ffnosh_filtered.iloc[-1]
        actual_date_ffnosh = last_ffnosh_row['Name']
        print(f"    Using ffnosh data from: {actual_date_ffnosh.strftime('%Y-%m-%d')}")

        # Clean column names - remove " - DS FREE FLOAT SHRE" suffix
        ffnosh_columns = [col.replace(' - DS FREE FLOAT SHRE', '') if isinstance(col, str) else col
                          for col in ffnosh.columns]

        # Create dataframe with company names and their last ffnosh
        last_ffnosh = pd.DataFrame({
            'NAME': ffnosh_columns[1:],  # Skip 'Name' column
            f'ffnosh last day {self._format_period()}': last_ffnosh_row.values[1:]
        })

        # Merge with symbols
        df = pd.merge(symbols_df, last_price, on="NAME", how="left")
        df = pd.merge(df, last_ffnosh, on="NAME", how="left")

        # Handle duplicates by aggregating: sum ffnosh, weight-average prices
        price_col = f'Price last day {self._format_period()}'
        ffnosh_col = f'ffnosh last day {self._format_period()}'

        # Check for duplicates by SYMBOL
        duplicates = df[df.duplicated(subset=['SYMBOL'], keep=False)]
        if len(duplicates) > 0:
            print(f"    Found {len(duplicates)} duplicate entries for {duplicates['SYMBOL'].nunique()} symbols")
            print(f"    Aggregating duplicates: summing ffnosh, weight-averaging prices")

            # Convert to numeric for aggregation
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            df[ffnosh_col] = pd.to_numeric(df[ffnosh_col], errors='coerce')

            # Aggregate by SYMBOL (not TYPE)
            aggregated = df.groupby('SYMBOL').apply(
                lambda group: pd.Series({
                    'Total_ffnosh': group[ffnosh_col].sum(),
                    'Weighted_Price': (
                        (group[ffnosh_col] * group[price_col]).sum() / group[ffnosh_col].sum()
                        if group[ffnosh_col].sum() > 0 else group[price_col].iloc[0]
                    )
                })
            ).reset_index()

            # Merge aggregated values back to original df
            df = pd.merge(df, aggregated[['SYMBOL', 'Total_ffnosh', 'Weighted_Price']],
                          on='SYMBOL', how='left')

            # Replace original columns with aggregated values
            df[ffnosh_col] = df['Total_ffnosh']
            df[price_col] = df['Weighted_Price']

            # Drop temporary columns
            df = df.drop(columns=['Total_ffnosh', 'Weighted_Price'])

            print(f"    Aggregated values merged back to {len(df)} rows")

        return df

    def load_scope_emissions(self, df):
        """
        Load Scope 1, 2, and 3 emissions, revenue, and imputation metadata.

        Reads the merged emissions files produced by script 02
        (data/merged_scope_emissions/scope_*_all_periods_filled.xlsx) alongside
        the corresponding unfilled originals. The "_filled" files contain
        forward/backward-filled values from script 02's time-series merge;
        any remaining gaps are later imputed by impute_scope_emissions() in
        this script. For each scope, a binary 'Filled Scope N Count' column
        is created (1 = value was NaN in the unfilled file but present in
        the filled file, 0 = value was originally present). Revenue is read
        directly from the LSEG carbon file (not filled).

        After merging, computes:
        - 'Scope 1+2+3': sum of the three scope values.
        - 'Carbon Intensity': Scope 1+2+3 divided by Revenue
          (units: tCO2e per revenue unit).

        Args:
            df (pd.DataFrame): Must contain TYPE column (LSEG company code)
                used as the join key for emissions series.

        Returns:
            pd.DataFrame: Input DataFrame extended with Scope 1, Scope 2,
                Scope 3, Revenue, Filled Scope 1/2/3 Count, Scope 1+2+3,
                and Carbon Intensity columns.
        """
        print(f"  Loading emissions data from filled files...")

        # Define paths to filled and unfilled files
        filled_dir = self.data_dir / "merged_scope_emissions"
        scope_1_filled_file = filled_dir / "scope_1_all_periods_filled.xlsx"
        scope_2_filled_file = filled_dir / "scope_2_all_periods_filled.xlsx"
        scope_3_filled_file = filled_dir / "scope_3_all_periods_filled.xlsx"
        scope_1_unfilled_file = filled_dir / "scope_1_all_periods.xlsx"
        scope_2_unfilled_file = filled_dir / "scope_2_all_periods.xlsx"
        scope_3_unfilled_file = filled_dir / "scope_3_all_periods.xlsx"

        # Convert period format from "1221" to "2021-12" for date matching
        year = int("20" + self.period[2:])
        month = int(self.period[:2])

        # Read Scope 1 from filled file
        print(f"    Reading {scope_1_filled_file.name}...")
        scope_1_df = pd.read_excel(scope_1_filled_file, index_col=0, parse_dates=True)

        # Filter for the target year and month, take the last available date in that month
        scope_1_period = scope_1_df[(scope_1_df.index.year == year) & (scope_1_df.index.month == month)]

        if len(scope_1_period) == 0:
            raise ValueError(f"No Scope 1 data found for period {year}-{month:02d}")

        # Get the last row for this period
        scope_1_row = scope_1_period.iloc[-1]
        actual_date_scope1 = scope_1_period.index[-1]
        print(f"      Target period: {year}-{month:02d}")
        print(f"      Using Scope 1 data from: {actual_date_scope1.strftime('%Y-%m-%d')}")
        print(f"      Number of companies in Scope 1: {scope_1_row.notna().sum()}")
        scope_1 = pd.Series(scope_1_row, name='Scope 1')

        # Read Scope 1 unfilled file to calculate filled values
        print(f"    Reading {scope_1_unfilled_file.name}...")
        scope_1_unfilled_df = pd.read_excel(scope_1_unfilled_file, index_col=0, parse_dates=True)
        scope_1_unfilled_period = scope_1_unfilled_df[(scope_1_unfilled_df.index.year == year) &
                                                       (scope_1_unfilled_df.index.month == month)]

        if len(scope_1_unfilled_period) > 0:
            scope_1_unfilled_row = scope_1_unfilled_period.iloc[-1]
            # Get common columns between filled and unfilled
            common_cols_s1 = scope_1_row.index.intersection(scope_1_unfilled_row.index)
            # Create binary indicator: 1 if unfilled was NaN but filled is not NaN, 0 otherwise
            scope_1_filled_indicator = ((scope_1_unfilled_row[common_cols_s1].isna()) &
                                       (scope_1_row[common_cols_s1].notna())).astype(int)
            filled_count_s1 = scope_1_filled_indicator.sum()
            print(f"      Number of filled values in Scope 1: {filled_count_s1}")
        else:
            filled_count_s1 = 0
            scope_1_filled_indicator = pd.Series(0, index=scope_1_row.index)
            print(f"      No unfilled Scope 1 data found for comparison")

        scope_1_filled_count = pd.Series(scope_1_filled_indicator, name='Filled Scope 1 Count')

        # Read Scope 2 from filled file
        print(f"    Reading {scope_2_filled_file.name}...")
        scope_2_df = pd.read_excel(scope_2_filled_file, index_col=0, parse_dates=True)

        # Filter for the target year and month
        scope_2_period = scope_2_df[(scope_2_df.index.year == year) & (scope_2_df.index.month == month)]

        if len(scope_2_period) == 0:
            raise ValueError(f"No Scope 2 data found for period {year}-{month:02d}")

        # Get the last row for this period
        scope_2_row = scope_2_period.iloc[-1]
        actual_date_scope2 = scope_2_period.index[-1]
        print(f"      Using Scope 2 data from: {actual_date_scope2.strftime('%Y-%m-%d')}")
        print(f"      Number of companies in Scope 2: {scope_2_row.notna().sum()}")
        scope_2 = pd.Series(scope_2_row, name='Scope 2')

        # Read Scope 2 unfilled file to calculate filled values
        print(f"    Reading {scope_2_unfilled_file.name}...")
        scope_2_unfilled_df = pd.read_excel(scope_2_unfilled_file, index_col=0, parse_dates=True)
        scope_2_unfilled_period = scope_2_unfilled_df[(scope_2_unfilled_df.index.year == year) &
                                                       (scope_2_unfilled_df.index.month == month)]

        if len(scope_2_unfilled_period) > 0:
            scope_2_unfilled_row = scope_2_unfilled_period.iloc[-1]
            # Get common columns between filled and unfilled
            common_cols_s2 = scope_2_row.index.intersection(scope_2_unfilled_row.index)
            # Create binary indicator: 1 if unfilled was NaN but filled is not NaN, 0 otherwise
            scope_2_filled_indicator = ((scope_2_unfilled_row[common_cols_s2].isna()) &
                                       (scope_2_row[common_cols_s2].notna())).astype(int)
            filled_count_s2 = scope_2_filled_indicator.sum()
            print(f"      Number of filled values in Scope 2: {filled_count_s2}")
        else:
            filled_count_s2 = 0
            scope_2_filled_indicator = pd.Series(0, index=scope_2_row.index)
            print(f"      No unfilled Scope 2 data found for comparison")

        scope_2_filled_count = pd.Series(scope_2_filled_indicator, name='Filled Scope 2 Count')

        # Read Scope 3 from filled file
        print(f"    Reading {scope_3_filled_file.name}...")
        scope_3_df = pd.read_excel(scope_3_filled_file, index_col=0, parse_dates=True)

        # Filter for the target year and month
        scope_3_period = scope_3_df[(scope_3_df.index.year == year) & (scope_3_df.index.month == month)]

        if len(scope_3_period) == 0:
            raise ValueError(f"No Scope 3 data found for period {year}-{month:02d}")

        # Get the last row for this period
        scope_3_row = scope_3_period.iloc[-1]
        actual_date_scope3 = scope_3_period.index[-1]
        print(f"      Using Scope 3 data from: {actual_date_scope3.strftime('%Y-%m-%d')}")
        print(f"      Number of companies in Scope 3: {scope_3_row.notna().sum()}")
        scope_3 = pd.Series(scope_3_row, name='Scope 3')

        # Read Scope 3 unfilled file to calculate filled values
        print(f"    Reading {scope_3_unfilled_file.name}...")
        scope_3_unfilled_df = pd.read_excel(scope_3_unfilled_file, index_col=0, parse_dates=True)
        scope_3_unfilled_period = scope_3_unfilled_df[(scope_3_unfilled_df.index.year == year) &
                                                       (scope_3_unfilled_df.index.month == month)]

        if len(scope_3_unfilled_period) > 0:
            scope_3_unfilled_row = scope_3_unfilled_period.iloc[-1]
            # Get common columns between filled and unfilled
            common_cols_s3 = scope_3_row.index.intersection(scope_3_unfilled_row.index)
            # Create binary indicator: 1 if unfilled was NaN but filled is not NaN, 0 otherwise
            scope_3_filled_indicator = ((scope_3_unfilled_row[common_cols_s3].isna()) &
                                       (scope_3_row[common_cols_s3].notna())).astype(int)
            filled_count_s3 = scope_3_filled_indicator.sum()
            print(f"      Number of filled values in Scope 3: {filled_count_s3}")
        else:
            filled_count_s3 = 0
            scope_3_filled_indicator = pd.Series(0, index=scope_3_row.index)
            print(f"      No unfilled Scope 3 data found for comparison")

        scope_3_filled_count = pd.Series(scope_3_filled_indicator, name='Filled Scope 3 Count')

        # Read revenue from original carbon file (not filled)
        print(f"    Reading revenue from {self.carbon_file.name}...")
        revenue_df = pd.read_excel(self.carbon_file, sheet_name="REVENUE", header=4)

        # Create period_date string for matching
        period_date = f"{year}-{month:02d}"
        period_mask = revenue_df.iloc[:, 0].astype(str).str.contains(period_date, na=False)
        target_row = revenue_df[period_mask].iloc[-1]

        # Get the actual date from the revenue data
        actual_date_revenue = target_row.iloc[0]
        print(f"      Using Revenue data from: {actual_date_revenue}")

        revenue_data = {}
        for col_name, value in target_row.items():
            if col_name != 'Code' and '(' in str(col_name):
                code = str(col_name).split('(')[0]
                revenue_data[code] = value

        print(f"      Number of companies in Revenue: {len([v for v in revenue_data.values() if pd.notna(v)])}")
        revenue = pd.Series(revenue_data, name='Revenue')

        # Merge with main dataframe using TYPE as the key
        df = pd.merge(df, scope_1, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, scope_2, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, scope_3, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, revenue, how='left', left_on='TYPE', right_index=True)

        # Merge filled indicator columns - binary flags (1 or 0) for each company
        df = pd.merge(df, scope_1_filled_count, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, scope_2_filled_count, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, scope_3_filled_count, how='left', left_on='TYPE', right_index=True)

        # Fill NaN values in filled count columns with 0 (companies not in the filled data)
        df['Filled Scope 1 Count'] = df['Filled Scope 1 Count'].fillna(0).astype(int)
        df['Filled Scope 2 Count'] = df['Filled Scope 2 Count'].fillna(0).astype(int)
        df['Filled Scope 3 Count'] = df['Filled Scope 3 Count'].fillna(0).astype(int)

        # Print summary of filled counts
        print(f"\n  Summary of filled emissions:")
        print(f"    Filled Scope 1 Count: {filled_count_s1}")
        print(f"    Filled Scope 2 Count: {filled_count_s2}")
        print(f"    Filled Scope 3 Count: {filled_count_s3}")

        # Calculate total emissions and carbon intensity
        df['Scope 1+2+3'] = pd.to_numeric(df['Scope 1'], errors='coerce') + \
                            pd.to_numeric(df['Scope 2'], errors='coerce') + \
                            pd.to_numeric(df['Scope 3'], errors='coerce')
        df['Carbon Intensity'] = df['Scope 1+2+3'] / pd.to_numeric(df['Revenue'], errors='coerce')

        return df

    def impute_revenue_from_delisted(self, df):
        """
        Impute missing revenue values from delisted companies revenue CSV

        Args:
            df: DataFrame with SYMBOL and Revenue columns

        Returns:
            DataFrame with missing Revenue values filled from delisted companies data
        """
        print(f"  Imputing missing revenue from {self.delisted_revenue_file.name}...")

        # Check if the file exists
        if not self.delisted_revenue_file.exists():
            print(f"    Warning: {self.delisted_revenue_file.name} not found, skipping revenue imputation")
            return df

        # Convert revenue to numeric for checking missing values
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')

        # Count missing values before imputation
        missing_before = df['Revenue'].isna().sum()
        print(f"    Companies with missing revenue: {missing_before}/{len(df)}")

        if missing_before == 0:
            print(f"    No missing revenue values to impute")
            return df

        # Load delisted companies revenue data
        delisted_df = pd.read_csv(self.delisted_revenue_file)

        # Strip whitespace from column names and values
        delisted_df.columns = delisted_df.columns.str.strip()
        delisted_df['Symbol'] = delisted_df['Symbol'].str.strip()

        # Get the year from the period (e.g., '1221' -> 2021)
        year = int("20" + self.period[2:])

        # Filter delisted data for the target year
        delisted_year = delisted_df[delisted_df['Year'] == year].copy()

        if len(delisted_year) == 0:
            print(f"    No delisted revenue data found for year {year}")
            return df

        # Create a mapping from Symbol to Revenue for this year
        revenue_map = dict(zip(delisted_year['Symbol'], delisted_year['Revenue']))

        # Impute missing revenue values
        imputed_count = 0
        for idx, row in df[df['Revenue'].isna()].iterrows():
            symbol = row['SYMBOL']
            if symbol in revenue_map:
                df.loc[idx, 'Revenue'] = revenue_map[symbol]
                imputed_count += 1
                print(f"      Imputed revenue for {symbol}: ${revenue_map[symbol]:,.0f}")

        # Count missing values after imputation
        missing_after = df['Revenue'].isna().sum()

        print(f"    Imputed revenue for {imputed_count} companies")
        print(f"    Remaining companies with missing revenue: {missing_after}/{len(df)}")

        # Print companies with missing revenue and break
        if missing_after > 0:
            missing_companies = df[df['Revenue'].isna()]['NAME'].tolist()
            print(f"    Companies with missing revenue: {missing_companies}")
            raise ValueError(f"Stopping execution: {missing_after} companies still have missing revenue: {missing_companies}")

        # Recalculate carbon intensity for imputed values
        df['Carbon Intensity'] = df['Scope 1+2+3'] / df['Revenue']

        return df

    def impute_scope_emissions(self, df):
        """
        Impute missing Scope 1, 2, and 3 emissions using MICE with PMM

        Args:
            df: DataFrame with Scope emissions, Revenue, float_mcap, Carbon Intensity, and GICS Sector columns

        Returns:
            DataFrame with imputed scope emissions and imputation flags
        """
        print(f"  Imputing missing scope emissions using MICE...")

        # Create one-hot encoding for GICS Sector
        gics = df.copy()
        gics = pd.get_dummies(gics, columns=['GICS Sector'], prefix='Sector', drop_first=False)

        # Define columns for imputation
        columns = ["Scope 1", "Scope 2", "Scope 3", "Revenue", "float_mcap"]
        sector_columns = [col for col in gics.columns if col.startswith("Sector_")]
        all_cols = columns + sector_columns

        # Convert all boolean columns to int (0/1)
        bool_cols = gics.select_dtypes(include='bool').columns
        gics[bool_cols] = gics[bool_cols].astype(int)

        # Subset the data
        data = gics[all_cols].copy().reset_index(drop=True)

        # Convert object columns to float
        object_cols = data.select_dtypes(include="object").columns
        for col in object_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Check if there are any missing values to impute
        missing_count = data[columns[:3]].isna().sum().sum()
        if missing_count == 0:
            print(f"    No missing scope emissions to impute")
            return df

        print(f"    Missing values before imputation:")
        print(f"      Scope 1: {data['Scope 1'].isna().sum()}")
        print(f"      Scope 2: {data['Scope 2'].isna().sum()}")
        print(f"      Scope 3: {data['Scope 3'].isna().sum()}")

        # Initialize MICE kernel.
        # num_datasets=3: build 3 independent imputed datasets to reduce
        #   Monte-Carlo variance; we use dataset 0 as the point estimate.
        # mean_match_candidates=5: for each missing value, sample from the
        #   5 closest observed candidates (PMM) to preserve the marginal
        #   distribution and avoid out-of-range predictions.
        # random_state=1: fixed seed for reproducibility.
        kernel = mf.ImputationKernel(
            data=data,
            num_datasets=3,
            mean_match_candidates=5,
            random_state=1
        )

        # Run MICE for 5 iterations; convergence is typically reached by
        # iteration 3-5 for the variable count in this dataset.
        kernel.mice(5)

        # Extract imputed dataset
        completed_data = kernel.complete_data(dataset=0)

        # Reset index for proper alignment
        gics.reset_index(drop=True, inplace=True)

        # Create imputation flags before replacing values
        df['Scope 1 Imputed'] = 0
        df['Scope 2 Imputed'] = 0
        df['Scope 3 Imputed'] = 0

        df.loc[gics['Scope 1'].isna(), 'Scope 1 Imputed'] = 1
        df.loc[gics['Scope 2'].isna(), 'Scope 2 Imputed'] = 1
        df.loc[gics['Scope 3'].isna(), 'Scope 3 Imputed'] = 1

        # Replace missing values with imputed values
        for col in ["Scope 1", "Scope 2", "Scope 3"]:
            df.loc[df[col].isna(), col] = completed_data.loc[df[col].isna(), col]

        print(f"    Imputed values:")
        print(f"      Scope 1: {df['Scope 1 Imputed'].sum()} companies")
        print(f"      Scope 2: {df['Scope 2 Imputed'].sum()} companies")
        print(f"      Scope 3: {df['Scope 3 Imputed'].sum()} companies")

        # Recalculate total emissions and carbon intensity
        df['Scope 1+2+3'] = pd.to_numeric(df['Scope 1'], errors='coerce') + \
                            pd.to_numeric(df['Scope 2'], errors='coerce') + \
                            pd.to_numeric(df['Scope 3'], errors='coerce')
        df['Carbon Intensity'] = df['Scope 1+2+3'] / pd.to_numeric(df['Revenue'], errors='coerce')

        return df

    def filter_stocks_with_missing_prices(self, df):
        """
        Filter out stocks with missing prices in the two years before the period
        (e.g., due to late IPO or other reasons)

        Args:
            df: DataFrame with SYMBOL column

        Returns:
            DataFrame with stocks with missing prices removed
        """
        print(f"  Filtering stocks with missing prices from {self.missing_prices_file.name}...")

        # Check if the file exists
        if not self.missing_prices_file.exists():
            print(f"    Warning: {self.missing_prices_file.name} not found, skipping filtering")
            return df

        # Load the stocks with missing prices
        missing_prices_df = pd.read_excel(self.missing_prices_file)

        # Get list of symbols to remove
        symbols_to_remove = missing_prices_df['symbol'].tolist()

        if len(symbols_to_remove) == 0:
            print(f"    No stocks to filter for this period")
            return df

        # Count how many stocks will be removed
        stocks_before = len(df)

        # Filter out the stocks
        df_filtered = df[~df['SYMBOL'].isin(symbols_to_remove)].copy()

        stocks_after = len(df_filtered)
        removed_count = stocks_before - stocks_after

        print(f"    Stocks before filtering: {stocks_before}")
        print(f"    Stocks to remove: {len(symbols_to_remove)}")
        print(f"    Stocks actually removed: {removed_count}")

        if removed_count > 0:
            removed_symbols = df[df['SYMBOL'].isin(symbols_to_remove)]['SYMBOL'].tolist()
            print(f"    Removed symbols: {', '.join(removed_symbols)}")

        print(f"    Stocks after filtering: {stocks_after}")

        return df_filtered

    def calculate_weights(self, df):
        """
        Calculate float market capitalisation and intra-sector weights.

        Computes float_mcap = price × ffnosh, then normalises within each
        GICS sector so that weights sum to 1.0. Also adds a rank_in_sector
        column (1 = largest by float_mcap within the sector).

        Args:
            df (pd.DataFrame): Must contain 'Price last day <period>',
                'ffnosh last day <period>', and 'GICS Sector' columns.

        Returns:
            pd.DataFrame: Input DataFrame extended with float_mcap,
                weight_in_sector, and rank_in_sector columns.
        """
        print(f"  Calculating market cap and sector weights...")

        # Calculate float market cap
        price_col = f'Price last day {self._format_period()}'
        ffnosh_col = f'ffnosh last day {self._format_period()}'
        df['float_mcap'] = pd.to_numeric(df[price_col], errors='coerce') * pd.to_numeric(df[ffnosh_col], errors='coerce')

        # Warn if any companies are missing a GICS sector; they will receive
        # NaN weights and be excluded from sector-level aggregations.
        missing_gics = df[df['GICS Sector'].isna()]
        if len(missing_gics) > 0:
            print(f"\n  WARNING: Found {len(missing_gics)} companies with missing GICS Sector:")
            for _, row in missing_gics.iterrows():
                print(f"    - {row['NAME']} ({row['SYMBOL']})")
            print(f"  These companies will be excluded from weight calculations.\n")

        # Calculate weights within each sector (handle zero-sum sectors)
        def safe_weight_calc(x):
            total = x.sum()
            if total == 0 or pd.isna(total):
                return pd.Series([0] * len(x), index=x.index)
            return x / total

        df['weight_in_sector'] = df.groupby('GICS Sector')['float_mcap'].transform(safe_weight_calc)

        # Verify that weights sum to 1.0 within each sector
        for sector in df['GICS Sector'].unique():
            sector_weights = df[df['GICS Sector'] == sector]['weight_in_sector'].values
            assert np.isclose(sector_weights.sum(), 1.0), f"Weights in sector {sector} do not sum to 1.0"

        # Rank stocks within sector by market cap
        df['rank_in_sector'] = df.groupby('GICS Sector')['float_mcap'].rank(
            ascending=False, method='dense'
        ).astype('Int64')  # Use nullable integer type

        return df

    def load_log_returns(self):
        """
        Load adjusted closing prices and compute monthly log returns.

        Reads the Yahoo Finance adjusted-price file for the period, resamples
        to month-end, and computes log returns as ln(P_t / P_{t-1}). Zeros
        are replaced with NaN and forward-filled before resampling to avoid
        spurious zero returns from missing data.

        Returns:
            pd.DataFrame: Monthly log returns with a 'Date' column and one
                column per ticker symbol.
        """
        print(f"  Computing log returns from {self.adj_price_file.name}...")

        # Load adjusted prices from Yahoo
        adj_close = pd.read_excel(self.adj_price_file)
        adj_close.index = pd.to_datetime(adj_close['Date'])
        adj_close.drop(columns='Date', inplace=True)

        # Ensure no zeros, forward-fill small gaps
        adj_close = adj_close.replace(0, np.nan).ffill().dropna(how="all")

        # Take last observation each month
        monthly_prices = adj_close.resample("M").last()

        # Compute log returns
        log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna(how="all")
        log_returns.reset_index(inplace=True)

        return log_returns

    def create_sector_log_returns(self, composition_df, log_returns_df):
        """
        Split the full log-return matrix into per-sector DataFrames.

        Args:
            composition_df (pd.DataFrame): Must contain SYMBOL and GICS Sector
                columns (output of calculate_weights).
            log_returns_df (pd.DataFrame): Full log-return matrix with a Date
                column and one column per ticker (output of load_log_returns).

        Returns:
            dict[str, pd.DataFrame]: Mapping from GICS sector name to a
                DataFrame containing Date and the log returns for all
                tickers in that sector that are present in log_returns_df.
        """
        print(f"  Creating sector-wise log returns...")

        # Get list of sectors
        sectors = composition_df['GICS Sector'].unique()

        # Create dictionary to store sector returns
        sector_returns = {}

        for sector in sectors:
            # Get symbols for this sector
            sector_symbols = composition_df[composition_df['GICS Sector'] == sector]['SYMBOL'].tolist()

            # Filter log returns for these symbols
            available_symbols = [s for s in sector_symbols if s in log_returns_df.columns]

            if available_symbols:
                sector_df = log_returns_df[['Date'] + available_symbols].copy()
                sector_returns[sector] = sector_df
            else:
                print(f"    Warning: No returns data for sector {sector}")

        return sector_returns

    def save_benchmark_weights_carbon(self, df):
        """
        Save the benchmark weights and carbon intensity dataset to Excel.

        Selects the columns needed for portfolio construction and carbon
        analysis, sorts by GICS sector and descending intra-sector weight,
        and writes to data/datasets/benchmark_weights_carbon_intensity_<period>.xlsx.

        Args:
            df (pd.DataFrame): Fully processed composition DataFrame.

        Returns:
            pd.DataFrame: The subset of columns written to disk.
        """
        # Select required columns including scope emissions, revenue, and imputation flags
        output_df = df[[
            'SYMBOL', 'NAME', 'GICS Sector',
            'Scope 1', 'Scope 2', 'Scope 3',
            'Revenue',
            'Scope 1 Imputed', 'Scope 2 Imputed', 'Scope 3 Imputed', 
            'Filled Scope 1 Count', 'Filled Scope 2 Count', 'Filled Scope 3 Count',
            'Carbon Intensity', 'weight_in_sector', 'TYPE', 'float_mcap'
        ]].copy()

        # Sort by sector and weight
        output_df = output_df.sort_values(['GICS Sector', 'weight_in_sector'], ascending=[True, False])

        # Save to Excel
        output_file = self.output_dir / f"benchmark_weights_carbon_intensity_{self.period}.xlsx"
        output_df.to_excel(output_file, index=False)
        print(f"  Saved benchmark weights to {output_file}")

        return output_df

    def save_full_composition(self, df):
        """
        Save the full composition DataFrame to Excel.

        Writes every column (including intermediate columns such as raw
        prices and ffnosh) to data/datasets/dataset_comp_<period>.xlsx.
        This file mirrors the schema expected by earlier analysis notebooks
        that pre-date the benchmark_weights_carbon_intensity output.

        Args:
            df (pd.DataFrame): Fully processed composition DataFrame.
        """
        output_file = self.output_dir / f"dataset_comp_{self.period}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"  Saved full composition to {output_file}")

    def save_sector_log_returns(self, sector_returns):
        """
        Save per-sector log returns to a multi-sheet Excel file.

        Writes one sheet per GICS sector to
        data/log_returns/sector_log_returns_comp_<period>.xlsx.
        Sheet names are truncated to 31 characters (Excel limit).

        Args:
            sector_returns (dict[str, pd.DataFrame]): Output of
                create_sector_log_returns().
        """
        output_file = self.log_returns_dir / f"sector_log_returns_comp_{self.period}.xlsx"

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sector, df in sector_returns.items():
                # Excel sheet names have a 31 character limit
                sheet_name = sector[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"  Saved sector log returns to {output_file}")

    def _format_period(self):
        """Format period code for display (e.g., '1221' -> 'Dec 21')"""
        month_map = {
            '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
            '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
            '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
        }
        month_code = self.period[:2]
        year_code = self.period[2:]
        return f"{month_map.get(month_code, month_code)} {year_code}"

    def create_datasets(self, save_full=True):
        """
        Main method to create all datasets for the period

        Args:
            save_full: Whether to save the full composition dataset (default: True)

        Returns:
            tuple: (benchmark_weights_df, sector_returns_dict)
        """
        print(f"\n{'='*60}")
        print(f"Creating datasets for period: {self.period} ({self._format_period()})")
        print(f"{'='*60}")

        # Step 1: Load and merge composition data
        df = self.load_symbol_data()
        df = self.load_price_and_shares(df)
        df = self.calculate_float_mcap(df)
        df = self.delete_duplicates(df)
        df = self.load_scope_emissions(df)
        df = self.impute_revenue_from_delisted(df)
        df = self.impute_scope_emissions(df)
        df = self.filter_stocks_with_missing_prices(df)
        df = self.calculate_weights(df)

        # Step 2: Load log returns
        log_returns = self.load_log_returns()

        # Step 3: Create sector log returns
        sector_returns = self.create_sector_log_returns(df, log_returns)

        # Step 4: Save outputs
        benchmark_df = self.save_benchmark_weights_carbon(df)
        self.save_sector_log_returns(sector_returns)

        if save_full:
            self.save_full_composition(df)

        print(f"\n  Summary:")
        print(f"    Total companies: {len(df)}")
        print(f"    Sectors: {df['GICS Sector'].notna().sum()} ({df['GICS Sector'].nunique()} unique)")
        print(f"    Companies with weights > 0: {(benchmark_df['weight_in_sector'] > 0).sum()}")
        print(f"    Companies with carbon data: {benchmark_df['Carbon Intensity'].notna().sum()}")
        print(f"{'='*60}\n")

        return benchmark_df, sector_returns


def create_all_datasets(periods=None, data_dir="data", save_full=True):
    """
    Create datasets for multiple time periods

    Args:
        periods: List of period codes (default: all 12 quarters from 0321
            to 1223). Each code is 'MMYY', e.g. '1221' = December 2021.
        data_dir: Root data directory path.
        save_full: Whether to save the full composition dataset alongside
            the benchmark weights file for each period.

    Returns:
        dict[str, dict]: Keyed by period code. Each value contains:
            - 'status': 'success' or 'failed'
            - 'benchmark' (pd.DataFrame): benchmark weights/carbon data
              (only present on success)
            - 'sector_returns' (dict): per-sector log-return DataFrames
              (only present on success)
            - 'error' (str): error message (only present on failure)
    """
    if periods is None:
        periods = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222", "0323", "0623", "0923", "1223"]

    print(f"\n{'#'*60}")
    print(f"# Dataset Creation Script - Sectoral Decarbonisation")
    print(f"# Creating datasets for {len(periods)} time periods")
    print(f"{'#'*60}")

    results: dict[str, dict] = {}
    # Each entry has keys: 'status' ('success'|'failed'),
    # and on success: 'benchmark' (pd.DataFrame), 'sector_returns' (dict).
    # On failure: 'error' (str).

    for period in periods:
        try:
            creator = DatasetCreator(period, data_dir)
            benchmark_df, sector_returns = creator.create_datasets(save_full=save_full)
            results[period] = {
                'benchmark': benchmark_df,
                'sector_returns': sector_returns,
                'status': 'success'
            }
        except Exception as e:
            print(f"\n  ERROR: Failed to create dataset for period {period}")
            print(f"  Error message: {str(e)}")
            results[period] = {
                'status': 'failed',
                'error': str(e)
            }

    # Print summary
    print(f"\n{'#'*60}")
    print(f"# SUMMARY")
    print(f"{'#'*60}")
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful
    print(f"  Total periods processed: {len(periods)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print(f"\n  Failed periods:")
        for period, result in results.items():
            if result['status'] == 'failed':
                print(f"    - {period}: {result['error']}")

    print(f"{'#'*60}\n")

    return results


if __name__ == "__main__":
    # Create datasets for all time periods
    results = create_all_datasets()
