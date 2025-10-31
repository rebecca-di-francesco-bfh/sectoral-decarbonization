"""
Dataset Creation Script for Sectoral Decarbonisation Project
============================================================

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
    """Creates datasets for a specific time period"""

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
        """Load symbol, type, and GICS sector information"""
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
        """Load price and float shares data"""
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
        """Load scope emissions and revenue data"""
        print(f"  Loading emissions data from {self.carbon_file.name}...")

        # Create mapping from Type code to company NAME
        # Read the symbol file to get Type codes and corresponding names
        symbol_data = pd.read_excel(self.symbol_file, sheet_name="SYMBOL", header=None)
        type_codes = symbol_data.iloc[1, 1:].astype(str).values  # Row 1 contains Type codes
        company_names = symbol_data.iloc[2, 1:].values  # Row 2 contains company NAMEs

        # Create mapping dictionary: Type code -> Company NAME
        code_to_name = dict(zip(type_codes, company_names))
  
        # Read scope emissions - use most recent data row
        scope_1_df = pd.read_excel(self.carbon_file, sheet_name="SCOPE 1", header=4)
        # Convert period format from "1221" to "2021-12" for date matching
        year = "20" + self.period[2:]
        month = self.period[:2]
        period_date = f"{year}-{month}"

        period_mask = scope_1_df.iloc[:, 0].astype(str).str.contains(period_date, na=False)
        target_row = scope_1_df[period_mask].iloc[-1]

        # Extract scope 1 data using TYPE codes as index
        scope_1_data = {}
        for col_name, value in target_row.items():
            if col_name != 'Code' and '(' in str(col_name):
                # Extract just the code part before the parenthesis
                code = str(col_name).split('(')[0]
                scope_1_data[code] = value

        scope_1 = pd.Series(scope_1_data, name='Scope 1')

      
        # Read scope 2
        scope_2_df = pd.read_excel(self.carbon_file, sheet_name="SCOPE 2", header=4)
        period_mask = scope_2_df.iloc[:, 0].astype(str).str.contains(period_date, na=False)
        target_row = scope_2_df[period_mask].iloc[-1]

        scope_2_data = {}
        for col_name, value in target_row.items():
            if col_name != 'Code' and '(' in str(col_name):
                code = str(col_name).split('(')[0]
                scope_2_data[code] = value

        scope_2 = pd.Series(scope_2_data, name='Scope 2')
   
        # Read scope 3
        scope_3_df = pd.read_excel(self.carbon_file, sheet_name="SCOPE 3", header=4)
        period_mask = scope_3_df.iloc[:, 0].astype(str).str.contains(period_date, na=False)
        target_row = scope_3_df[period_mask].iloc[-1]

        scope_3_data = {}
        for col_name, value in target_row.items():
            if col_name != 'Code' and '(' in str(col_name):
                code = str(col_name).split('(')[0]
                scope_3_data[code] = value

        scope_3 = pd.Series(scope_3_data, name='Scope 3')

        # Read revenue
        revenue_df = pd.read_excel(self.carbon_file, sheet_name="REVENUE", header=4)
        period_mask = revenue_df.iloc[:, 0].astype(str).str.contains(period_date, na=False)
        target_row = revenue_df[period_mask].iloc[-1]

        revenue_data = {}
        for col_name, value in target_row.items():
            if col_name != 'Code' and '(' in str(col_name):
                code = str(col_name).split('(')[0]
                revenue_data[code] = value

        revenue = pd.Series(revenue_data, name='Revenue')

        # Merge with main dataframe using TYPE as the key
        df = pd.merge(df, scope_1, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, scope_2, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, scope_3, how='left', left_on='TYPE', right_index=True)
        df = pd.merge(df, revenue, how='left', left_on='TYPE', right_index=True)

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
        columns = ["Scope 1", "Scope 2", "Scope 3", "Revenue", "float_mcap", "Carbon Intensity"]
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

        # Initialize kernel (this builds multiple trees for imputation)
        kernel = mf.ImputationKernel(
            data=data,
            num_datasets=3,
            mean_match_candidates=5,
            random_state=1
        )

        # Run MICE with PMM
        kernel.mice(5)  # 5 iterations

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
        """Calculate float market cap and sector weights"""
        print(f"  Calculating market cap and sector weights...")

        # Calculate float market cap
        price_col = f'Price last day {self._format_period()}'
        ffnosh_col = f'ffnosh last day {self._format_period()}'
        df['float_mcap'] = pd.to_numeric(df[price_col], errors='coerce') * pd.to_numeric(df[ffnosh_col], errors='coerce')

        # DEBUG: Check for missing GICS Sector
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
        """Load and compute log returns from adjusted prices"""
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
        """Create log returns organized by sector"""
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
        """Save benchmark weights and carbon intensity DataFrame"""
        # Select only required columns
        output_df = df[['SYMBOL', 'NAME', 'GICS Sector', 'Carbon Intensity', 'weight_in_sector']].copy()

        # Sort by sector and weight
        output_df = output_df.sort_values(['GICS Sector', 'weight_in_sector'], ascending=[True, False])

        # Save to Excel
        output_file = self.output_dir / f"benchmark_weights_carbon_intensity_{self.period}.xlsx"
        output_df.to_excel(output_file, index=False)
        print(f"  Saved benchmark weights to {output_file}")

        return output_df

    def save_full_composition(self, df):
        """Save full composition dataset (for backward compatibility)"""
        output_file = self.output_dir / f"dataset_comp_{self.period}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"  Saved full composition to {output_file}")

    def save_sector_log_returns(self, sector_returns):
        """Save sector log returns to Excel file with multiple sheets"""
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
        periods: List of period codes (default: ['1221', '0322', '0622', '0922', '1222'])
        data_dir: Root data directory path
        save_full: Whether to save full composition datasets
    """
    if periods is None:
        periods = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222", "0323", "0623", "0923", "1223"]

    print(f"\n{'#'*60}")
    print(f"# Dataset Creation Script - Sectoral Decarbonisation")
    print(f"# Creating datasets for {len(periods)} time periods")
    print(f"{'#'*60}")

    results = {}

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
