"""
Download Yahoo Finance adjusted price data for missing symbols across all time periods.

This script:
1. For each time period, identifies symbols missing from LSEG price data
2. Downloads adjusted close prices from Yahoo Finance with appropriate date ranges
3. Saves the data as adj_price_yahoo_comp_{period}.xlsx

Date ranges are calculated as:
- Start date: 2 years and 1 month before the period
- End date: 4 months after the period
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time


class YahooDataDownloader:
    def __init__(self, period: str, data_dir: str = "data"):
        """
        Initialize the Yahoo data downloader.

        Args:
            period: Time period in format "MMYY" (e.g., "1221" for Dec 2021)
            data_dir: Base directory for data files
        """
        self.period = period
        self.data_dir = Path(data_dir)

        # Set up file paths
        self.symbol_file = self.data_dir / "lseg" / "constituents_symbols" / f"symbol_comp_{period}.xlsm"
        self.price_file = self.data_dir / "lseg" / "prices_dividends" / f"price_div_comp_{period}.xlsm"
        self.output_file = self.data_dir /"tests"/ "yahoo" / f"adj_price_yahoo_comp_{period}.xlsx"

        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Calculate date range
        self.start_date, self.end_date = self._calculate_date_range()
       
    def _calculate_date_range(self):
        """
        Calculate start and end dates based on period.
        Start: 2 years and 1 day before period
        End: 4 months after period
        """
        # Parse period: "1221" -> December 2021
        month = int(self.period[:2])
        year = 2000 + int(self.period[2:])

        period_date = datetime(year, month, 1)

        # Start date: 2 years 1 day before
        start_date = period_date - relativedelta(years=2)

        # End date: 4 months after
        end_date = period_date + relativedelta(months=4)
        print("START DATE: ", start_date)
        print("END DATE: ", end_date)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def load_symbols(self):
        """Load all symbols and TYPE codes from the symbol file"""

        type_lseg = pd.read_excel(self.symbol_file, sheet_name="SYMBOL", dtype=str).iloc[0].values[1:]
        symbol_lseg = pd.read_excel(self.symbol_file, sheet_name="SYMBOL", dtype=str, header=2)
        symbol_lseg = symbol_lseg.iloc[:1].transpose().reset_index().rename(
            columns={"index": "NAME", 0: "SYMBOL"}
        ).iloc[1:]
        symbol_lseg['TYPE'] = type_lseg

        symbol_lseg.loc[symbol_lseg['SYMBOL'] == 'BRK.A', 'SYMBOL'] = 'BRK-B'
        symbol_lseg.loc[symbol_lseg['SYMBOL'] == 'BF.B',  'SYMBOL'] = 'BF-B'

        return symbol_lseg


    def calculate_adjusted_prices_from_lseg(self, symbol_type_map):
        """Calculate adjusted prices from LSEG close prices and dividends"""

        # Load price, dividend rate, and dividend date data
        price = pd.read_excel(self.price_file, sheet_name='CLOSE PRICE', header=4)
        price.columns = price.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()

        div_rate = pd.read_excel(self.price_file, sheet_name='DIV RATE', header=4)
        div_rate.columns = div_rate.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()

        div_date = pd.read_excel(self.price_file, sheet_name='DIV DATE', header=4)
        div_date.columns = div_date.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()

        # Align columns
        div_date.columns = price.columns
        div_rate.columns = price.columns

        # Set index to Code column
        price.index = price['Code']
        div_rate.index = div_rate['Code']
        div_date.index = div_date['Code']

        # Filter by date range (start to end date from period)
        price = price[self.start_date:self.end_date]
        div_rate = div_rate[self.start_date:self.end_date]
        div_date = div_date[self.start_date:self.end_date]

        # Remove Code column
        price = price.iloc[:, 1:]
        div_rate = div_rate.iloc[:, 1:]
        div_date = div_date.iloc[:, 1:]

        # Calculate adjustment factors
        adj_factors = pd.DataFrame(1.0, index=price.index, columns=price.columns)

        for company in price.columns:
            for i in range(1, len(price)):
                date = price.index[i]
                prev_date = price.index[i - 1]

                # If ex-dividend happens on this day
                if pd.notna(div_date.at[date, company]):
                    div = div_rate.at[date, company]
                    price_prev = price.at[prev_date, company]
                    if price_prev and price_prev != 0:
                        factor = (price_prev - div) / price_prev
                        adj_factors.at[date, company] = factor

        # Calculate cumulative adjustment factors in reverse (like Yahoo)
        cum_factors = adj_factors.iloc[::-1].cumprod().iloc[::-1]

        # Build adjusted prices
        adjusted_prices = price * cum_factors

        # Map TYPE codes to SYMBOL names - build dict first, then create DataFrame
        symbol_columns = {}
        for _, row in symbol_type_map.iterrows():
            symbol = row['SYMBOL']
            type_code = row['TYPE']
            if type_code in adjusted_prices.columns:
                symbol_columns[symbol] = adjusted_prices[type_code]

        # Create DataFrame from dict in one operation
        adj_close_lseg = pd.DataFrame(symbol_columns, index=adjusted_prices.index)

        print(f"  Calculated adjusted prices for {len(adj_close_lseg.columns)} symbols from LSEG")
        return adj_close_lseg

    def download_all_from_yahoo(self, symbol_type_map):
        """Download all symbols from Yahoo Finance"""
   
        all_symbols = symbol_type_map['SYMBOL'].tolist()
 
        # Download data from Yahoo Finance
        data_yahoo = yf.download(
            all_symbols,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=False,
            progress=True
        )

        # Extract adjusted close prices - build dict first to avoid fragmentation
        adj_close_dict = {}

        # If only one ticker, yfinance doesn't use a multi-indexed DataFrame
        if len(all_symbols) == 1:
            ticker = all_symbols[0]
            if not data_yahoo.empty and 'Adj Close' in data_yahoo.columns:
                data_series = data_yahoo['Adj Close']
                # Only add if data has valid (non-NaN) values
                if not data_series.isna().all():
                    adj_close_dict[ticker] = data_series
        else:
            for ticker in all_symbols:
                try:
                    data_series = data_yahoo[ticker]['Adj Close']
                    # Only add if data has valid (non-NaN) values
                    if not data_series.isna().all():
                        adj_close_dict[ticker] = data_series
                except (KeyError, AttributeError):
                    pass  # Symbol not found in downloaded data

        # Create DataFrame from dict in one operation
        adj_close_yahoo = pd.DataFrame(adj_close_dict)

        print(f"  Successfully downloaded {len(adj_close_yahoo.columns)}/{len(all_symbols)} symbols from Yahoo")

        return adj_close_yahoo

    def check_nans(self, adj_close_data):
        """Check for NaN values in the adjusted close data"""
        if adj_close_data is None or adj_close_data.empty:
            return

        cols_with_nans = adj_close_data.columns[adj_close_data.isna().any()]

        if len(cols_with_nans) == 0:
            print("  ✓ No NaN values found - all data is complete")
            return

        print(f"  Found NaN values in {len(cols_with_nans)} columns:")
        nan_columns = adj_close_data.loc[:, adj_close_data.isna().any(axis=0)]

        rows = []
        for col in nan_columns.columns:
            first_valid = nan_columns[col].first_valid_index()
            nan_count = nan_columns[col].isna().sum()
            total_rows = len(nan_columns[col])
            print(f"    - {col}: {nan_count}/{total_rows} NaNs, first valid data: {first_valid}")
            rows.append({
            "symbol": col,
            "period": self.period,
            "start_period": adj_close_data.index[0],
            "first_valid": first_valid,
            "nan_count": nan_count,
            "total_rows": total_rows,
            "nan_ratio": nan_count/total_rows
        })

        out = pd.DataFrame(rows)

        fname = f"data/tests/stocks_with_missing_prices/stocks_with_missing_prices_{self.period}.xlsx"
        out.to_excel(fname, index=False)


    def save_data(self, adj_close_data):
        """Save the adjusted close data to Excel"""
        if adj_close_data is None or adj_close_data.empty:
            print("  No data to save")
            return

        # Check for NaN values before saving
        self.check_nans(adj_close_data)

        adj_close_data.to_excel(self.output_file)
 

    def run(self):
        """Run the complete download process"""
        print(f"\n{'='*60}")
        print(f"Processing period: {self.period}")
        print(f"{'='*60}\n")
        
        if self.output_file.exists():
            print(f"  >>> Skipping period {self.period}: file already exists ({self.output_file.name})")
            return
        # Load all symbols with TYPE mapping
        symbol_type_map = self.load_symbols()

        # Strategy:
        # 1. Download from Yahoo for all symbols
        # 2. Fill missing symbols with LSEG calculated adjusted prices
        # 3. Report what's still missing

        # Step 1: Download from Yahoo
        adj_close_data = self.download_all_from_yahoo(symbol_type_map)

        # Step 2: Retry failed downloads (for connection timeouts)
        all_symbols = symbol_type_map['SYMBOL'].tolist()
        missing_from_yahoo = [s for s in all_symbols if s not in adj_close_data.columns]

        if missing_from_yahoo:
            print(f"\n  {len(missing_from_yahoo)} symbols missing from Yahoo")

            # Retry failed symbols individually (helps with connection timeouts)
            max_retries = 1
            recovered_dict = {}

            for attempt in range(max_retries):
                if not missing_from_yahoo:
                    break

                print(f"\n  Retry attempt {attempt + 1}/{max_retries} for {len(missing_from_yahoo)} symbols...")

                still_missing = []
                for symbol in missing_from_yahoo:
                    try:
                        data_single = yf.download(
                            symbol,
                            start=self.start_date,
                            end=self.end_date,
                            auto_adjust=False,
                            progress=False
                        )
               
                        # Check if we got valid data
                        data_series = None
                        if isinstance(data_single, pd.DataFrame) and not data_single.empty:
                            # Check if columns are MultiIndex (grouped by ticker)
                            if isinstance(data_single.columns, pd.MultiIndex):
                                # Multi-index case: access via ticker name
                                if symbol in data_single.columns.get_level_values(0):
                                    data_series = data_single[symbol]['Adj Close']
                            else:
                                # Simple columns - check for 'Adj Close' directly
                                if 'Adj Close' in data_single.columns:
                                    data_series = data_single['Adj Close']

                        # Validate and add the data
                        if data_series is not None and not data_series.isna().all():
                            recovered_dict[symbol] = data_series
                            print(f"    ✓ Recovered {symbol}")
                            continue

                        still_missing.append(symbol)

                    except Exception as e:
                        still_missing.append(symbol)
                        if attempt == max_retries - 1:
                            print(f"    ✗ Failed {symbol}: {str(e)[:60]}")

                    time.sleep(0.05)  # Small delay to avoid rate limiting

                missing_from_yahoo = still_missing

                if missing_from_yahoo and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Waiting {wait_time}s before next retry...")
                    time.sleep(wait_time)

            # Add recovered symbols to data in one operation
            if recovered_dict:
                recovered_df = pd.DataFrame(recovered_dict)
                adj_close_data = pd.concat([adj_close_data, recovered_df], axis=1)
                print(f"\n  ✓ Recovered {len(recovered_dict)} symbols through retries")

            # Step 3: Fill remaining missing with LSEG
            if missing_from_yahoo:
                print(f"\n  {len(missing_from_yahoo)} symbols still missing after retries, filling from LSEG...")
                print(f"  Missing symbols: {', '.join(missing_from_yahoo[:10])}{'...' if len(missing_from_yahoo) > 10 else ''}")

                # Calculate adjusted prices from LSEG
                adj_close_lseg = self.calculate_adjusted_prices_from_lseg(symbol_type_map)
         
                # Collect LSEG data for missing symbols to avoid fragmentation
                lseg_fill_dict = {}
                filled_count = 0
                still_missing = []

                for symbol in missing_from_yahoo:
                    if symbol in adj_close_lseg.columns:
                        lseg_fill_dict[symbol] = adj_close_lseg[symbol]
                        filled_count += 1
                        print(f"    ✓ Filled {symbol} from LSEG calculation")
                    else:
                        still_missing.append(symbol)
                        print(f"    ✗ Still missing: {symbol}")

                # Concatenate LSEG data with Yahoo data in one operation
                if lseg_fill_dict:
                    lseg_fill_df = pd.DataFrame(lseg_fill_dict)
                    # Reindex LSEG data to match Yahoo data index, filling missing dates with NaN
                    lseg_fill_df = lseg_fill_df.reindex(adj_close_data.index)
                    adj_close_data = pd.concat([adj_close_data, lseg_fill_df], axis=1)
         
                print(f"\n  Filled {filled_count}/{len(missing_from_yahoo)} symbols from LSEG")

                if still_missing:
                    print(f"\n  WARNING: {len(still_missing)} symbols still missing: {', '.join(still_missing)}")

        # Align start to the last available trading day of the first month
        if not adj_close_data.empty:
            first_month = adj_close_data.index[0].month
            first_year = adj_close_data.index[0].year
            last_day_first_month = adj_close_data.loc[
                (adj_close_data.index.year == first_year) &
                (adj_close_data.index.month == first_month)
            ].index.max()

            print(f"  Aligning start: first trading day is {adj_close_data.index[0].date()}, "
                  f"using last day of first month: {last_day_first_month.date()}")

            adj_close_data = adj_close_data.loc[last_day_first_month:]

        # Save data
        self.save_data(adj_close_data)

        print(f"\n{'='*60}")
        print(f"Completed period: {self.period}")
        print(f"{'='*60}\n")


def main():
    """Download Yahoo Finance data for all time periods"""
    periods = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222", "0323", "0623", "0923", "1223"]

    print("\n" + "="*60)
    print("Yahoo Finance Data Downloader")
    print("="*60)
    print(f"\nProcessing {len(periods)} time periods: {', '.join(periods)}")
    print("\n")

    for period in periods:
        downloader = YahooDataDownloader(period)
        downloader.run()

    print("\n" + "="*60)
    print("All periods completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
