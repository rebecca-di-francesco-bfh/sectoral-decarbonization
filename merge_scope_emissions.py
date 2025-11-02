"""
Merge Scope Emissions Data Across All Time Periods
===================================================

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
    """Merges scope emissions data across all time periods"""

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

        # Store conflicts for later export
        self.all_conflicts = {}

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

    def check_duplicate_values(self, all_dataframes_with_periods: list, sheet_name: str):
        """
        Check for cases where the same date and company have different values

        Args:
            all_dataframes_with_periods: List of tuples (DataFrame, period_code)
            sheet_name: Name of the sheet being processed
        """
        if len(all_dataframes_with_periods) < 2:
            return

        print(f"  Checking for conflicting values in duplicate dates...")

        conflicts = []
        conflict_details = []  # For detailed Excel export

        # Separate dataframes and periods
        all_dataframes = [df for df, _ in all_dataframes_with_periods]
        periods = [period for _, period in all_dataframes_with_periods]

        # Create a combined dataframe with all data (including duplicates)
        combined_with_dupes = pd.concat(all_dataframes, axis=0)

        # Create a mapping to track which period each row came from
        period_mapping = []
        for df, period in all_dataframes_with_periods:
            period_mapping.extend([period] * len(df))

        # Find duplicate dates
        duplicate_dates = combined_with_dupes.index[combined_with_dupes.index.duplicated(keep=False)]
        unique_duplicate_dates = duplicate_dates.unique()

        if len(unique_duplicate_dates) == 0:
            print(f"    No duplicate dates found")
            return

        print(f"    Found {len(unique_duplicate_dates)} dates that appear in multiple period files")

        # Check each duplicate date
        for date in unique_duplicate_dates:
            # Get all rows for this date
            rows = combined_with_dupes.loc[date]

            # Find which periods this date appears in
            date_indices = [i for i, d in enumerate(combined_with_dupes.index) if d == date]
            periods_for_date = [period_mapping[i] for i in date_indices]

            # If only one row, it's not really a duplicate (shouldn't happen but check anyway)
            if isinstance(rows, pd.Series):
                continue

            # Check each company column
            for company_code in rows.columns:
                values = rows[company_code].dropna()

                if len(values) > 1:
                    # Check if values are different
                    unique_values = values.unique()
                    if len(unique_values) > 1:
                        # Store conflict info
                        conflicts.append({
                            'Date': date,
                            'Company Code': company_code,
                            'Values': unique_values.tolist(),
                            'Count': len(values)
                        })

                        # Store detailed info for Excel export
                        # Get the actual values from each period
                        period_values = {}
                        for i, (val, period) in enumerate(zip(values, periods_for_date)):
                            if not pd.isna(val):
                                period_values[period] = val

                        conflict_details.append({
                            'Date': date,
                            'Company Code': company_code,
                            'Period Values': period_values
                        })

        # Store conflicts for this sheet
        if len(conflict_details) > 0:
            self.all_conflicts[sheet_name] = conflict_details

        if len(conflicts) > 0:
            print(f"\n    ⚠️  WARNING: Found {len(conflicts)} cases with conflicting values:")
            print(f"    {'='*70}")

            # Analyze which periods are involved in conflicts
            period_pairs = {}
            for conflict_detail in conflict_details:
                period_values = conflict_detail['Period Values']
                periods_involved = tuple(sorted(period_values.keys()))

                if periods_involved not in period_pairs:
                    period_pairs[periods_involved] = 0
                period_pairs[periods_involved] += 1

            # Check if 1223 is always involved
            conflicts_with_1223 = sum(count for periods, count in period_pairs.items() if '1223' in periods)
            total_conflicts = len(conflict_details)

            print(f"    Total conflicts: {total_conflicts}")
            print(f"    Conflicts involving period 1223: {conflicts_with_1223} ({conflicts_with_1223/total_conflicts*100:.1f}%)")

            if conflicts_with_1223 == total_conflicts:
                print(f"    ✓ ALL conflicts involve period 1223")

            print(f"\n    Period combinations causing conflicts:")
            for periods, count in sorted(period_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      {' vs '.join(periods)}: {count} conflicts")

            # Show first 5 conflicts in detail
            print(f"\n    First 5 conflict examples:")
            for i, conflict in enumerate(conflicts[:5]):
                date_str = conflict['Date'].strftime('%d.%m.%Y') if hasattr(conflict['Date'], 'strftime') else str(conflict['Date'])
                print(f"    {i+1}. Date: {date_str}, Company: {conflict['Company Code']}")
                print(f"       Different values found: {conflict['Values']}")

            if len(conflicts) > 5:
                print(f"    ... and {len(conflicts) - 5} more conflicts (see conflicts_report.xlsx for full details)")

            print(f"    {'='*70}")
            print(f"    Note: Keeping the first occurrence in each case")
        else:
            print(f"    ✓ No conflicting values found (duplicate dates have identical values)")

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

        # Check for conflicts before removing duplicates
        self.check_duplicate_values(all_dataframes_with_periods, sheet_name)

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

    def analyze_conflict_patterns(self):
        """
        Analyze patterns in conflicts to identify problematic periods
        """
        if not self.all_conflicts:
            return

        print(f"\n{'='*80}")
        print("Conflict Pattern Analysis")
        print(f"{'='*80}")

        # Count conflicts by period across all sheets
        period_conflict_count = {}
        period_pair_count = {}

        # Special analysis for 1223
        conflicts_1223_vs_others_same = 0
        conflicts_1223_examples = []

        for sheet_name, conflicts in self.all_conflicts.items():
            for conflict in conflicts:
                period_values = conflict['Period Values']
                periods_involved = sorted(period_values.keys())

                # Count individual periods
                for period in periods_involved:
                    if period not in period_conflict_count:
                        period_conflict_count[period] = 0
                    period_conflict_count[period] += 1

                # Count period pairs
                periods_tuple = tuple(periods_involved)
                if periods_tuple not in period_pair_count:
                    period_pair_count[periods_tuple] = 0
                period_pair_count[periods_tuple] += 1

                # Check if 1223 is different but all other periods have the same value
                if '1223' in period_values:
                    value_1223 = period_values['1223']
                    other_values = [v for p, v in period_values.items() if p != '1223']

                    if len(other_values) > 0:
                        # Check if all other values are the same
                        other_values_unique = list(set(other_values))

                        # If all other periods have the same value, and it's different from 1223
                        if len(other_values_unique) == 1 and other_values_unique[0] != value_1223:
                            conflicts_1223_vs_others_same += 1

                            # Store example (limit to first 5)
                            if len(conflicts_1223_examples) < 5:
                                conflicts_1223_examples.append({
                                    'sheet': sheet_name,
                                    'date': conflict['Date'],
                                    'company': conflict['Company Code'],
                                    'value_1223': value_1223,
                                    'value_others': other_values_unique[0],
                                    'other_periods': [p for p in period_values.keys() if p != '1223']
                                })

        # Analysis
        total_conflicts = sum(len(conflicts) for conflicts in self.all_conflicts.values())
        conflicts_with_1223 = period_conflict_count.get('1223', 0)

        print(f"\nTotal conflicts across all sheets: {total_conflicts}")
        print(f"\nPeriods involved in conflicts:")
        for period, count in sorted(period_conflict_count.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_conflicts * 100
            print(f"  {period}: {count} conflicts ({percentage:.1f}%)")

        print(f"\nMost common period combinations:")
        for periods, count in sorted(period_pair_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = count / total_conflicts * 100
            print(f"  {' vs '.join(periods)}: {count} conflicts ({percentage:.1f}%)")

        # Check if 1223 is always involved
        if conflicts_with_1223 == total_conflicts:
            print(f"\n⚠️  FINDING: Period 1223 is involved in ALL {total_conflicts} conflicts!")
            print(f"   This suggests the 1223 period file may have different/updated data.")
        elif conflicts_with_1223 > 0:
            print(f"\n⚠️  FINDING: Period 1223 is involved in {conflicts_with_1223}/{total_conflicts} conflicts ({conflicts_with_1223/total_conflicts*100:.1f}%)")
        else:
            print(f"\n✓ Period 1223 is not involved in any conflicts")

        # Special analysis: 1223 different but all others the same
        if conflicts_with_1223 > 0:
            print(f"\n{'='*70}")
            print(f"SPECIAL ANALYSIS: Cases where 1223 differs but all other periods agree")
            print(f"{'='*70}")
            print(f"Conflicts where 1223 has a different value but all other periods")
            print(f"have the SAME value: {conflicts_1223_vs_others_same}/{conflicts_with_1223}")
            print(f"({conflicts_1223_vs_others_same/conflicts_with_1223*100:.1f}% of conflicts involving 1223)")

            if conflicts_1223_vs_others_same == conflicts_with_1223:
                print(f"\n✓ In ALL cases, 1223 is the ONLY period with a different value!")
                print(f"  This strongly suggests that period 1223 has updated/revised data.")
            elif conflicts_1223_vs_others_same > 0:
                print(f"\n⚠️  In {conflicts_1223_vs_others_same} cases, 1223 is the only outlier.")
                print(f"  In {conflicts_with_1223 - conflicts_1223_vs_others_same} cases, other periods also differ.")

            # Show examples
            if conflicts_1223_examples:
                print(f"\nExamples of 1223 vs all others agreeing:")
                for i, ex in enumerate(conflicts_1223_examples, 1):
                    date_str = ex['date'].strftime('%d.%m.%Y') if hasattr(ex['date'], 'strftime') else str(ex['date'])
                    print(f"  {i}. {ex['sheet']} | Date: {date_str} | Company: {ex['company']}")
                    print(f"     Period 1223 value: {ex['value_1223']}")
                    print(f"     All other periods ({', '.join(ex['other_periods'])}) value: {ex['value_others']}")

            print(f"{'='*70}")

        print(f"\n{'='*80}\n")

    def export_conflicts_to_excel(self):
        """
        Export all conflicts to an Excel file with company codes and periods in columns
        """
        if not self.all_conflicts:
            print(f"\n  No conflicts to export")
            return

        print(f"\n{'='*80}")
        print("Exporting Conflicts to Excel")
        print(f"{'='*80}")

        # Create a consolidated conflicts dataframe
        all_conflict_rows = []
        conflicts_without_1223_rows = []

        for sheet_name, conflicts in self.all_conflicts.items():
            for conflict in conflicts:
                date = conflict['Date']
                company_code = conflict['Company Code']
                period_values = conflict['Period Values']

                # Create a row with Date as first column
                row = {
                    'Date': date,
                    'Sheet': sheet_name
                }

                # Add columns for each period with format "CompanyCode (Period)"
                for period, value in period_values.items():
                    col_name = f"{company_code} ({period})"
                    row[col_name] = value

                all_conflict_rows.append(row)

                # Check if 1223 is NOT involved in this conflict
                if '1223' not in period_values:
                    conflicts_without_1223_rows.append(row)

        if not all_conflict_rows:
            print(f"  No conflict rows to export")
            return

        # Create DataFrame for all conflicts
        conflicts_df = pd.DataFrame(all_conflict_rows)

        # Fill NaN with empty string for better readability
        conflicts_df = conflicts_df.fillna('')

        # Sort by Date and Sheet
        conflicts_df = conflicts_df.sort_values(['Sheet', 'Date']).reset_index(drop=True)

        # Save to Excel - All conflicts
        conflicts_file = self.output_dir / "conflicts_report.xlsx"

        with pd.ExcelWriter(conflicts_file, engine='openpyxl') as writer:
            conflicts_df.to_excel(writer, sheet_name='All Conflicts', index=False)

            # Also create separate sheets for each data type
            for sheet_name in self.all_conflicts.keys():
                sheet_conflicts = conflicts_df[conflicts_df['Sheet'] == sheet_name].copy()
                sheet_conflicts = sheet_conflicts.drop(columns=['Sheet'])

                # Truncate sheet name if too long (Excel limit is 31 chars)
                excel_sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                sheet_conflicts.to_excel(writer, sheet_name=excel_sheet_name, index=False)

        print(f"  All conflicts exported to: {conflicts_file}")
        print(f"  Total conflicts: {len(conflicts_df)}")
        print(f"  Sheets with conflicts: {', '.join(self.all_conflicts.keys())}")

        # Export conflicts WITHOUT 1223
        if conflicts_without_1223_rows:
            conflicts_no_1223_df = pd.DataFrame(conflicts_without_1223_rows)
            conflicts_no_1223_df = conflicts_no_1223_df.fillna('')
            conflicts_no_1223_df = conflicts_no_1223_df.sort_values(['Sheet', 'Date']).reset_index(drop=True)

            conflicts_no_1223_file = self.output_dir / "conflicts_excluding_1223.xlsx"

            with pd.ExcelWriter(conflicts_no_1223_file, engine='openpyxl') as writer:
                conflicts_no_1223_df.to_excel(writer, sheet_name='Conflicts Excl 1223', index=False)

                # Also create separate sheets for each data type (excluding 1223)
                for sheet_name in self.all_conflicts.keys():
                    sheet_conflicts_no_1223 = conflicts_no_1223_df[conflicts_no_1223_df['Sheet'] == sheet_name].copy()

                    if len(sheet_conflicts_no_1223) > 0:
                        sheet_conflicts_no_1223 = sheet_conflicts_no_1223.drop(columns=['Sheet'])

                        # Truncate sheet name if too long (Excel limit is 31 chars)
                        excel_sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                        sheet_conflicts_no_1223.to_excel(writer, sheet_name=excel_sheet_name, index=False)

            print(f"\n  Conflicts WITHOUT 1223 exported to: {conflicts_no_1223_file}")
            print(f"  Conflicts excluding 1223: {len(conflicts_no_1223_df)}")
            print(f"  ({len(conflicts_no_1223_df)/len(conflicts_df)*100:.1f}% of total conflicts)")
        else:
            print(f"\n  ✓ No conflicts found that exclude period 1223")
            print(f"    (All conflicts involve period 1223)")

        print(f"{'='*80}\n")

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

        # Analyze conflict patterns
        self.analyze_conflict_patterns()

        # Export conflicts if any were found
        self.export_conflicts_to_excel()

        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nMerged data files created:")
        for key in results.keys():
            print(f"  - {key}_all_periods.xlsx")

        if self.all_conflicts:
            total_conflicts = sum(len(v) for v in self.all_conflicts.values())

            # Count conflicts without 1223
            conflicts_without_1223 = 0
            for conflicts in self.all_conflicts.values():
                for conflict in conflicts:
                    if '1223' not in conflict['Period Values']:
                        conflicts_without_1223 += 1

            print(f"\nConflicts reports created:")
            print(f"  - conflicts_report.xlsx")
            print(f"    (Contains all {total_conflicts} conflicts across {len(self.all_conflicts)} sheets)")

            if conflicts_without_1223 > 0:
                print(f"  - conflicts_excluding_1223.xlsx")
                print(f"    (Contains {conflicts_without_1223} conflicts that do NOT involve period 1223)")
                print(f"    ({conflicts_without_1223/total_conflicts*100:.1f}% of total conflicts)")
            else:
                print(f"  Note: All conflicts involve period 1223")
        else:
            print(f"\n✓ No conflicts found - all duplicate dates have identical values")

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
