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
        return df

    def load_price_and_shares(self, symbols_df):
        """Load price and float shares data"""
        print(f"  Loading price and shares data from {self.price_div_file.name}...")

        # Load close prices - company names are in columns (header row)
        price = pd.read_excel(self.price_div_file, sheet_name="CLOSE PRICE", header=3)
        # Drop the 'Code' row (first row after header)
        price = price.iloc[1:]

        # Get last price for each company (last row)
        last_price_row = price.iloc[-1]

        # Create dataframe with company names and their last prices
        last_price = pd.DataFrame({
            'NAME': price.columns[1:],  # Skip 'Name' column
            f'Price last day {self._format_period()}': last_price_row.values[1:]
        })

        # Load float shares (ffnosh)
        ffnosh = pd.read_excel(self.price_div_file, sheet_name="FFNOSH", header=3)
        # Drop the 'Code' row
        ffnosh = ffnosh.iloc[1:]

        # Get last ffnosh for each company (last row)
        last_ffnosh_row = ffnosh.iloc[-1]

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

    def calculate_weights(self, df):
        """Calculate float market cap and sector weights"""
        print(f"  Calculating market cap and sector weights...")

        # Calculate float market cap
        price_col = f'Price last day {self._format_period()}'
        ffnosh_col = f'ffnosh last day {self._format_period()}'
        df['float_mcap'] = pd.to_numeric(df[price_col], errors='coerce') * pd.to_numeric(df[ffnosh_col], errors='coerce')

        # Calculate weights within each sector (handle zero-sum sectors)
        def safe_weight_calc(x):
            total = x.sum()
            if total == 0 or pd.isna(total):
                return pd.Series([0] * len(x), index=x.index)
            return x / total

        df['weight_in_sector'] = df.groupby('GICS Sector')['float_mcap'].transform(safe_weight_calc)

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

        # Compute log returns
        log_returns = np.log(adj_close / adj_close.shift(1))
        log_returns = log_returns.dropna()
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
        df = self.load_scope_emissions(df)
        df = self.load_price_and_shares(df)
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
        periods = ['1221', '0322', '0622', '0922', '1222']

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
