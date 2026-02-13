"""
Merge scope emissions data aross all time periods.

This script reads scope emissions data from all time periods and creates
consolidated files with:
- Columns: Company numeric codes (e.g., '891399' extracted from '891399(ENERDP024)')
- Index: All monthly dates (e.g., '01.01.2014', '01.02.2014', etc.)
- One file each for: Scope 1, Scope 2, Scope 3

Additionally creates "_filled" versions that:
- Filter for companies with at least one emission in 2020-2024
- Forward-fill missing values within this period

The original data files are not modified.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ScopeEmissionsMerger:
    """Merges scope emissions data across all time periods.

    Reads per-period LSEG Excel files from `data/lseg/scope_emissions/`,
    consolidates them into single DataFrames (Scope 1, 2, 3), and saves the
    results to `data/merged_scope_emissions/`. When the same date appears in
    multiple period files the first occurrence is kept. Also creates
    forward-filled versions limited to 2021-2023.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the merger

        Args:
            data_dir: Root data directory path
        """
        self.data_dir = Path(data_dir)
        self.scope_emissions_dir = self.data_dir / "lseg" / "scope_emissions"
        self.output_dir = self.data_dir / "merged_scope_emissions"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_available_periods(self):
        """
        Find all available period files in the scope_emissions directory

        Returns:
            List of period codes (e.g., ['0321', '0621', ...])
        """
        pattern = "carbon_int_comp_*.xlsm"
        files = list(self.scope_emissions_dir.glob(pattern))

        periods = []
        for file in files:
            # Extract period from filename: carbon_int_comp_1221.xlsm -> 1221
            period = file.stem.split('_')[-1]
            periods.append(period)

        periods = sorted(periods)
        print(f"Found {len(periods)} period files: {', '.join(periods)}")

        return periods

    def extract_company_code(self, col_name: str) -> str:
        """
        Extract the numeric company code from column name.

        Example: '891399(ENERDP024)' -> '891399'

        Args:
            col_name: Column name from the Excel file

        Returns:
            Numeric company code or None if format doesn't match
        """
        col_str = str(col_name)

        # Check if column has the expected format with parentheses
        if '(' in col_str and ')' in col_str:
            # Extract the part before the parenthesis
            code = col_str.split('(')[0].strip()
            return code

        return None

    def extract_sheet_data(self, file_path: Path, sheet_name: str) -> pd.DataFrame:
        """
        Extract ALL data from a specific sheet (all rows, all dates)

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to read (e.g., 'SCOPE 1', 'SCOPE 2', 'SCOPE 3', 'REVENUE')

        Returns:
            DataFrame with dates as index and company codes as columns
        """
        # Read the sheet with header at row 4 (0-indexed)
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=4)

        # The first column should be the date column
        date_col = df.columns[0]

        # Filter out rows where the date column is NA or not a valid date
        df = df[df[date_col].notna()].copy()

        # Convert date column to datetime, then to string in the original format
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df[date_col].notna()].copy()

        # Set date as index
        df.set_index(date_col, inplace=True)

        # Create mapping of original column names to numeric codes
        column_mapping = {}
        for col in df.columns:
            code = self.extract_company_code(col)
            if code:
                column_mapping[col] = code

        # Keep only columns with valid company codes and rename them
        df = df[[col for col in df.columns if col in column_mapping]]
        df.columns = [column_mapping[col] for col in df.columns]

        return df

    def merge_data_for_sheet(self, periods: list, sheet_name: str) -> pd.DataFrame:
        """
        Merge data across all periods for a specific sheet

        Args:
            periods: List of period codes
            sheet_name: Name of the sheet to merge

        Returns:
            DataFrame with all dates as index and company codes as columns
        """
        print(f"\nProcessing {sheet_name}...")

        all_dataframes_with_periods = []

        for period in periods:
            file_path = self.scope_emissions_dir / f"carbon_int_comp_{period}.xlsm"

            if not file_path.exists():
                print(f"  Warning: File not found for period {period}")
                continue

            print(f"  Reading {period}...", end='')

            try:
                df = self.extract_sheet_data(file_path, sheet_name)

                if len(df) > 0:
                    all_dataframes_with_periods.append((df, period))
                    print(f" {len(df)} dates × {len(df.columns)} companies")
                else:
                    print(" No data")
            except Exception as e:
                print(f" Error: {str(e)}")
                continue

        if len(all_dataframes_with_periods) == 0:
            print(f"  No data collected for {sheet_name}")
            return pd.DataFrame()

        # Concatenate all dataframes
        # This will align by column names (company codes) and stack dates
        print(f"  Combining data from {len(all_dataframes_with_periods)} period files...")

        all_dataframes = [df for df, _ in all_dataframes_with_periods]
        combined_df = pd.concat(all_dataframes, axis=0)

        # Remove duplicate dates (keep first occurrence)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Sort by date
        combined_df = combined_df.sort_index()

        # Get all unique company codes across all periods
        all_codes = sorted(combined_df.columns.tolist())

        # Ensure all columns are present (fill missing with NaN)
        combined_df = combined_df.reindex(columns=all_codes)

        print(f"  Result: {len(combined_df)} unique dates × {len(combined_df.columns)} companies")
        if len(combined_df) > 0:
            print(f"  Date range: {combined_df.index[0].strftime('%d.%m.%Y')} to {combined_df.index[-1].strftime('%d.%m.%Y')}")

        return combined_df

    def merge_all_data(self):
        """
        Main method to merge all scope emissions and revenue data

        Returns:
            Dictionary with keys 'scope_1', 'scope_2', 'scope_3', 'revenue'
        """
        print(f"\n{'='*80}")
        print("Merging Scope Emissions Data Across All Periods")
        print(f"{'='*80}")

        # Get available periods
        periods = self.get_available_periods()

        if len(periods) == 0:
            print("No period files found!")
            return {}

        # Define sheets to process
        sheets = {
            'scope_1': 'SCOPE 1',
            'scope_2': 'SCOPE 2',
            'scope_3': 'SCOPE 3'
        }

        results = {}

        # Process each sheet
        for key, sheet_name in sheets.items():
            df = self.merge_data_for_sheet(periods, sheet_name)

            if not df.empty:
                results[key] = df

                # Save to Excel
                output_excel = self.output_dir / f"{key}_all_periods.xlsx"
                df.to_excel(output_excel)
                print(f"  Saved to: {output_excel}")

        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nMerged data files created:")
        for key in results.keys():
            print(f"  - {key}_all_periods.xlsx")

        print(f"{'='*80}\n")

        return results

    def _print_filling_examples(self, df: pd.DataFrame, all_years: list):
        """
        Print examples of different filling scenarios for companies
        Focuses on filling 2021-2023 using data from 2020-2024

        Args:
            df: DataFrame with filtered data for all years
            all_years: List of all years to consider [2020, 2021, 2022, 2023, 2024]
        """
        import random

        # Get yearly data by taking the last value of each year
        df_yearly = df.groupby(df.index.year).last()

        # Ensure all years are represented
        for year in all_years:
            if year not in df_yearly.index:
                df_yearly.loc[year] = np.nan

        df_yearly = df_yearly.sort_index()
        df_yearly = df_yearly.loc[all_years]

        # Define scenarios to look for (focusing on 2021-2023 filling)
        scenarios = {
            '2021_filled_from_2020': None,  # 2021 missing, filled from 2020
            '2021_filled_from_2022': None,  # 2021 missing, 2020 missing, filled from 2022
            '2022_filled_from_2021': None,  # 2022 missing, filled from 2021
            '2022_filled_from_2023': None,  # 2022 missing, 2021 missing, filled from 2023
            '2023_filled_from_2022': None,  # 2023 missing, filled from 2022
            'two_consecutive_missing': None,  # 2 consecutive years missing (e.g., 2021-2022 or 2022-2023)
            'three_consecutive_missing': None,  # All 3 years 2021-2023 missing
            'complete_2021_2023': None,  # All years 2021-2023 present
        }

        # Shuffle companies for randomness
        companies = list(df_yearly.columns)
        random.shuffle(companies)

        # Find examples for each scenario
        for company in companies:
            company_data = df_yearly[company]

            # Check specific scenarios
            has_2020 = pd.notna(company_data.loc[2020])
            has_2021 = pd.notna(company_data.loc[2021])
            has_2022 = pd.notna(company_data.loc[2022])
            has_2023 = pd.notna(company_data.loc[2023])
            has_2024 = pd.notna(company_data.loc[2024])

            # 2021 missing, filled from 2020
            if scenarios['2021_filled_from_2020'] is None:
                if not has_2021 and has_2020:
                    scenarios['2021_filled_from_2020'] = company

            # 2021 missing, 2020 missing, filled from 2022
            if scenarios['2021_filled_from_2022'] is None:
                if not has_2021 and not has_2020 and has_2022:
                    scenarios['2021_filled_from_2022'] = company

            # 2022 missing, filled from 2021
            if scenarios['2022_filled_from_2021'] is None:
                if not has_2022 and has_2021:
                    scenarios['2022_filled_from_2021'] = company

            # 2022 missing, 2021 missing, filled from 2023
            if scenarios['2022_filled_from_2023'] is None:
                if not has_2022 and not has_2021 and has_2023:
                    scenarios['2022_filled_from_2023'] = company

            # 2023 missing, filled from 2022
            if scenarios['2023_filled_from_2022'] is None:
                if not has_2023 and has_2022:
                    scenarios['2023_filled_from_2022'] = company

            # Two consecutive years missing (only with 2020 and 2024 data)
            if scenarios['two_consecutive_missing'] is None:
                if has_2020 and has_2024:
                    # 2021 and 2022 missing, but 2023 present
                    if not has_2021 and not has_2022 and has_2023:
                        scenarios['two_consecutive_missing'] = company
                    # Or 2022 and 2023 missing, but 2021 present
                    elif has_2021 and not has_2022 and not has_2023:
                        scenarios['two_consecutive_missing'] = company

            # All three years 2021-2023 missing (will use 2020 or 2024)
            if scenarios['three_consecutive_missing'] is None:
                if not has_2021 and not has_2022 and not has_2023 and (has_2020 or has_2024):
                    scenarios['three_consecutive_missing'] = company

            # All years 2021-2023 present
            if scenarios['complete_2021_2023'] is None:
                if has_2021 and has_2022 and has_2023:
                    scenarios['complete_2021_2023'] = company

            # Stop if we found all scenarios
            if all(v is not None for v in scenarios.values()):
                break

        # Print examples
        found_count = 0
        for scenario_name, company in scenarios.items():
            if company is not None:
                found_count += 1
                company_data = df_yearly[company]

                # Create a more descriptive title
                title = scenario_name.replace('_', ' ').title()
                print(f"  Scenario: {title}")
                print(f"    Company: {company}")
                print(f"    Data by year:")
                for year in all_years:
                    value = company_data.loc[year]
                    status = f"{value:.2f}" if pd.notna(value) else "MISSING"
                    # Mark which years will be filled
                    marker = " <- WILL BE FILLED" if year in [2021, 2022, 2023] and pd.isna(value) else ""
                    print(f"      {year}: {status}{marker}")
                print()
            else:
                print(f"  Scenario: {scenario_name.replace('_', ' ').title()}")
                print(f"    No example found in the data")
                print()

        if found_count == 0:
            print(f"  No example scenarios found - data may be too complete or too sparse")
            print()

    def create_filled_versions(self, results: dict):
        """
        Create filled versions of the scope emissions data
        - Filter for companies with at least one emission in 2021-2023
        - Fill missing values in 2021-2023 using data from neighboring years (2020-2024)
        - Do NOT fill 2020 or 2024

        Args:
            results: Dictionary with merged DataFrames (scope_1, scope_2, scope_3)

        Returns:
            Dictionary with the same keys as `results` (scope_1, scope_2, scope_3),
            containing filled DataFrames filtered to 2020-2024. Keys for scopes
            with no data in 2021-2023 are omitted.

        Side effect:
            Writes one Excel file per scope to `self.output_dir`:
            {key}_all_periods_filled.xlsx
        """
        print(f"\n{'='*80}")
        print("Creating Filled Versions (Filling 2021-2023 only)")
        print(f"{'='*80}")

        # Years to consider (for filtering and filling logic)
        all_years = [2020, 2021, 2022, 2023, 2024]
        # Years to actually fill
        years_to_fill = [2021, 2022, 2023]

        filled_results = {}

        for key, df in results.items():
            # Skip if key is not a scope emissions (e.g., if revenue was included)
            if key not in ['scope_1', 'scope_2', 'scope_3']:
                continue

            print(f"\nProcessing {key.upper().replace('_', ' ')}...")

            # Filter for dates in 2020-2024
            df_filtered = df[df.index.year.isin(all_years)].copy()

            if df_filtered.empty:
                print(f"  No data found for years {all_years}")
                continue

            print(f"  Data for 2020-2024: {len(df_filtered)} dates × {len(df_filtered.columns)} companies")

            # Identify companies with at least one non-null value in 2021-2023
            df_target_years = df_filtered[df_filtered.index.year.isin(years_to_fill)]
            companies_with_data = df_target_years.columns[df_target_years.notna().any()].tolist()
            print(f"  Companies with at least one value in 2021-2023: {len(companies_with_data)}")

            # Calculate excluded companies
            excluded_companies = set(df.columns) - set(companies_with_data)
            excluded_count = len(excluded_companies)

            if excluded_count > 0:
                print(f"  Excluded companies (no data in 2021-2023): {excluded_count}")
                print(f"    Examples: {', '.join(list(excluded_companies)[:10])}")
                if excluded_count > 10:
                    print(f"    ... and {excluded_count - 10} more")

            # Keep only companies with data in 2021-2023
            df_filtered = df_filtered[companies_with_data].copy()

            # Print example scenarios before filling
            print(f"\n  {'='*70}")
            print(f"  Example Filling Scenarios (Before Filling)")
            print(f"  {'='*70}")

            # Analyze filling patterns and find examples
            self._print_filling_examples(df_filtered, all_years)

            print(f"  {'='*70}\n")

            # Apply smart filling logic - only fill 2021-2023
            print(f"  Applying fill logic to years 2021-2023 only...")

            # Get yearly data (take last value of each year)
            df_yearly = df_filtered.groupby(df_filtered.index.year).last()

            # Ensure all years are represented
            for year in all_years:
                if year not in df_yearly.index:
                    df_yearly.loc[year] = np.nan
            df_yearly = df_yearly.sort_index()
            df_yearly = df_yearly.loc[all_years]

            # Count missing values in 2021-2023 before filling
            missing_before = df_yearly.loc[years_to_fill].isna().sum().sum()

            # Fill only 2021-2023, column by column
            df_filled_yearly = df_yearly.copy()

            for company in df_yearly.columns:
                # Fill 2021
                if pd.isna(df_filled_yearly.loc[2021, company]):
                    if pd.notna(df_yearly.loc[2020, company]):
                        df_filled_yearly.loc[2021, company] = df_yearly.loc[2020, company]
                    elif pd.notna(df_yearly.loc[2022, company]):
                        df_filled_yearly.loc[2021, company] = df_yearly.loc[2022, company]
                    elif pd.notna(df_yearly.loc[2023, company]):
                        df_filled_yearly.loc[2021, company] = df_yearly.loc[2023, company]
                    elif pd.notna(df_yearly.loc[2024, company]):
                        df_filled_yearly.loc[2021, company] = df_yearly.loc[2024, company]

                # Fill 2022
                if pd.isna(df_filled_yearly.loc[2022, company]):
                    if pd.notna(df_yearly.loc[2021, company]) or pd.notna(df_filled_yearly.loc[2021, company]):
                        # Use 2021 (either original or filled)
                        df_filled_yearly.loc[2022, company] = df_filled_yearly.loc[2021, company]
                    elif pd.notna(df_yearly.loc[2020, company]):
                        df_filled_yearly.loc[2022, company] = df_yearly.loc[2020, company]
                    elif pd.notna(df_yearly.loc[2023, company]):
                        df_filled_yearly.loc[2022, company] = df_yearly.loc[2023, company]
                    elif pd.notna(df_yearly.loc[2024, company]):
                        df_filled_yearly.loc[2022, company] = df_yearly.loc[2024, company]

                # Fill 2023
                if pd.isna(df_filled_yearly.loc[2023, company]):
                    if pd.notna(df_yearly.loc[2022, company]) or pd.notna(df_filled_yearly.loc[2022, company]):
                        # Use 2022 (either original or filled)
                        df_filled_yearly.loc[2023, company] = df_filled_yearly.loc[2022, company]
                    elif pd.notna(df_yearly.loc[2021, company]) or pd.notna(df_filled_yearly.loc[2021, company]):
                        df_filled_yearly.loc[2023, company] = df_filled_yearly.loc[2021, company]
                    elif pd.notna(df_yearly.loc[2020, company]):
                        df_filled_yearly.loc[2023, company] = df_yearly.loc[2020, company]
                    elif pd.notna(df_yearly.loc[2024, company]):
                        df_filled_yearly.loc[2023, company] = df_yearly.loc[2024, company]

            # Count missing values in 2021-2023 after filling
            missing_after = df_filled_yearly.loc[years_to_fill].isna().sum().sum()
            filled_count = missing_before - missing_after

            print(f"  Missing values in 2021-2023 before filling: {missing_before}")
            print(f"  Missing values in 2021-2023 after filling: {missing_after}")
            print(f"  Values filled: {filled_count}")

            # Expand yearly data back to monthly (replicate the yearly value for all months in that year)
            df_filled = df_filtered.copy()
            for year in all_years:
                year_mask = df_filled.index.year == year
                if year_mask.any() and year in df_filled_yearly.index:
                    # For each month in this year, use the yearly value
                    for company in df_filled.columns:
                        df_filled.loc[year_mask, company] = df_filled_yearly.loc[year, company]

            # Store result
            filled_results[key] = df_filled

            # Save to Excel
            output_excel = self.output_dir / f"{key}_all_periods_filled.xlsx"
            df_filled.to_excel(output_excel)
            print(f"  Saved to: {output_excel}")

        print(f"\n{'='*80}")
        print("Filled Versions Summary")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nFilled data files created:")
        for key in filled_results.keys():
            print(f"  - {key}_all_periods_filled.xlsx")
        print(f"{'='*80}\n")

        return filled_results

    def generate_summary_stats(self, results: dict):
        """
        Generate summary statistics for the merged data

        Args:
            results: Dictionary with merged DataFrames
        """
        print(f"\n{'='*80}")
        print("Summary Statistics")
        print(f"{'='*80}")

        for key, df in results.items():
            print(f"\n{key.upper().replace('_', ' ')}:")
            print(f"  Dates: {len(df)}")
            print(f"  Companies: {len(df.columns)}")
            print(f"  Total data points: {df.notna().sum().sum()}")
            print(f"  Missing data points: {df.isna().sum().sum()}")
            print(f"  Data completeness: {df.notna().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%")

            if not df.empty:
                # Companies with data in all dates
                complete_companies = (df.notna().sum() == len(df)).sum()
                print(f"  Companies with data in all dates: {complete_companies}")

                # Dates with complete data for all companies
                complete_dates = (df.notna().sum(axis=1) == len(df.columns)).sum()
                print(f"  Dates with complete data: {complete_dates}")

        print(f"\n{'='*80}\n")


def main():
    """Main execution function"""
    merger = ScopeEmissionsMerger(data_dir="data")
    results = merger.merge_all_data()

    if results:
        merger.generate_summary_stats(results)

        # Create filled versions
        filled_results = merger.create_filled_versions(results)

        if filled_results:
            merger.generate_summary_stats(filled_results)
    else:
        print("No data was merged. Please check the input files.")


if __name__ == "__main__":
    main()
