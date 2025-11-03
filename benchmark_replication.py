import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np


def aggregate_duplicates(price_df, ffnosh_df):
    """
    Calculate weighted price/ffnosh for duplicates (e.g. GOOGL, NWSA etc).
    Weighted price = Σ(price * shares) / Σ(shares)
    """
    weighted_price = (price_df * ffnosh_df).T.groupby(level=0).sum().T / ffnosh_df.T.groupby(level=0).sum().T
    total_ffnosh = ffnosh_df.T.groupby(level=0).sum().T
    return weighted_price, total_ffnosh


def check_duplicate_nans(price_df, ffnosh_df):
    """Check and report NaN values in duplicate ticker columns."""
    dupes = price_df.columns[price_df.columns.duplicated()].unique()
    if len(dupes) > 0:
        print(f"\n   Found {len(dupes)} duplicate tickers: {dupes.tolist()}")
        for dup in dupes:
            cols = [c for c in price_df.columns if c == dup]
            price_nans = price_df[cols].isna().sum().sum()
            shares_nans = ffnosh_df[cols].isna().sum().sum()
            if (price_nans > 0) or (shares_nans > 0):
                print(f"      {dup}: {price_nans} NaN prices, {shares_nans} NaN shares")


def main():
    """Main function to process all periods and compute sector portfolio returns."""
    # Initialize list to store all period results
    all_sector_portfolios = []

    for period in ["0321","0621", "0921", "1221", "0322", "0622", "0922", "1222", "0323", "0623", "0923", "1223"]:
        print("\n" + "="*80)
        print(f"PROCESSING PERIOD: {period}")
        print("="*80)
        # Parse the period code (MMYY format)
        month = int(period[:2])
        year = int("20" + period[2:])

        # Last day of the period month
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

        # Start date: 1st day of next month
        next_3m_start_date = datetime(next_year, next_month, 1)

        # End date: 3 months after start date, then get last day of that month
        end_month_first_day = next_3m_start_date + relativedelta(months=3)
        next_3m_end_date = end_month_first_day - timedelta(days=1)

        print(f"\nDate Range: {next_3m_start_date.strftime('%d/%m/%Y')} to {next_3m_end_date.strftime('%d/%m/%Y')}")
        print("\n[1/5] Loading LSEG data...")
        # Load close price adjusted for corporate splits from LSEG
        price_next_3m = pd.read_excel(f"data/lseg/prices_dividends/price_div_comp_{period}.xlsm",
                                       sheet_name='CLOSE PRICE', header=4)
        price_next_3m.columns = price_next_3m.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()

        # Load float-adjusted shares outstanding from LSEG
        ffnosh_next_3m = pd.read_excel(f"data/lseg/prices_dividends/price_div_comp_{period}.xlsm",
                                        sheet_name='FFNOSH', header=4)
        ffnosh_next_3m.columns = ffnosh_next_3m.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()

        assert (ffnosh_next_3m.columns == price_next_3m.columns).all()
        print("   ✓ Loaded PRICE and FFNOSH data")

        # Create a TYPE-SYMBOL dictionary so that the type columns of ffnosh and price dataframes can be mapped to symbols
        print("\n[2/5] Mapping symbols...")
        type_lseg = pd.read_excel(f"data/lseg/constituents_symbols/symbol_comp_{period}.xlsm", sheet_name="SYMBOL", dtype=str).iloc[0] # to extract the row with TYPE
        type_lseg = type_lseg.values[1:] # to take only the code values of companies (the 'TYPE') without the "Type" text in it
        symbol_lseg = pd.read_excel(f"data/lseg/constituents_symbols/symbol_comp_{period}.xlsm", sheet_name="SYMBOL", dtype=str, header=2)
        symbol_lseg = symbol_lseg.iloc[:1] # to extract only the row with SYMBOL (called WC05601 in LSEG)
        symbol_lseg = symbol_lseg.transpose() # to have the name of companies as index and the SYMBOL as the column
        symbol_lseg = symbol_lseg.reset_index().rename(columns= {"index": "NAME", 0: "SYMBOL"})  # to have the name of companies as a separate additional column
        symbol_lseg = symbol_lseg.iloc[1:] # to remove "NAME" and "WC05601" texts as values of the first row
        symbol_lseg['TYPE'] = type_lseg
        symbol_type_matches  = symbol_lseg.set_index("TYPE")["SYMBOL"].to_dict()
        price_next_3m = price_next_3m.rename(columns=symbol_type_matches)
        ffnosh_next_3m = ffnosh_next_3m.rename(columns=symbol_type_matches)

        print(f"   ✓ Mapped {len(symbol_type_matches)} symbols")
        price_next_3m['Code'] = pd.to_datetime(price_next_3m['Code'])
        ffnosh_next_3m['Code'] = pd.to_datetime(ffnosh_next_3m['Code'])

        # Code is the date, put it as index
        price_next_3m.index = price_next_3m.Code
        ffnosh_next_3m.index = ffnosh_next_3m.Code

        price_next_3m = price_next_3m.iloc[:, 1:]  # to remove the Code column
        ffnosh_next_3m = ffnosh_next_3m.iloc[:, 1:]

        # Assert the indices are equal
        assert price_next_3m.index.equals(ffnosh_next_3m.index), "Indices are not equal between price_next_3m and ffnosh_next_3m"

        mask = (price_next_3m.index >= next_3m_start_date) & (price_next_3m.index <= next_3m_end_date)
        price_next_3m = price_next_3m.loc[mask]
        ffnosh_next_3m = ffnosh_next_3m.loc[mask]

        # change symbol from BRK.A to BRK-B and BF.B to BF-B to match it with Yahoo and previous datasets creation methodology:
        price_next_3m.rename(columns={'BRK.A':'BRK-B'}, inplace=True)
        ffnosh_next_3m.rename(columns={'BRK.A':'BRK-B'}, inplace=True)

        price_next_3m.rename(columns={'BF.B':'BF-B'}, inplace=True)
        ffnosh_next_3m.rename(columns={'BF.B':'BF-B'}, inplace=True)

        print("\n[3/5] Handling duplicates...")
        check_duplicate_nans(price_next_3m, ffnosh_next_3m)
        price_agg, ffnosh_agg = aggregate_duplicates(price_next_3m, ffnosh_next_3m)
        print("   ✓ Aggregated duplicate tickers")

        # --- Check NaNs in PRICE ---
        print("\n[4/5] Checking for missing values...")
        # --- Check NaNs in PRICE ---
        price_next_3m = price_next_3m.loc[:, ~price_next_3m.columns.duplicated()]
        nan_mask = price_next_3m.isna()

        total_price_nans = nan_mask.sum().sum()
        if total_price_nans == 0:
            print("   ✓ No missing PRICE values")
        else:
            print(f"   ⚠ Found {total_price_nans} missing PRICE values:")
            for col in price_next_3m.columns:
                if nan_mask[col].any():
                    nan_count = nan_mask[col].sum()
                    nan_dates = price_next_3m.index[nan_mask[col]].strftime("%Y-%m-%d").tolist()
                    print(f"      {col}: {nan_count} days (e.g., {', '.join(nan_dates[:3])}...)")

            before_fill = price_next_3m.isna().sum().sum()
            price_next_3m = price_next_3m.ffill().bfill()
            after_fill = price_next_3m.isna().sum().sum()
            print(f"   ✓ Filled {before_fill - after_fill} missing PRICE cells")

        # --- Check NaNs in FFNOSH ---
        ffnosh_next_3m = ffnosh_next_3m.loc[:, ~ffnosh_next_3m.columns.duplicated()]
        nan_mask = ffnosh_next_3m.isna()

        total_ffnosh_nans = nan_mask.sum().sum()
        if total_ffnosh_nans == 0:
            print("   ✓ No missing FFNOSH values")
        else:
            print(f"   ⚠ Found {total_ffnosh_nans} missing FFNOSH values:")
            for col in ffnosh_next_3m.columns:
                if nan_mask[col].any():
                    nan_count = nan_mask[col].sum()
                    nan_dates = ffnosh_next_3m.index[nan_mask[col]].strftime("%Y-%m-%d").tolist()
                    print(f"      {col}: {nan_count} days (e.g., {', '.join(nan_dates[:3])}...)")

            before_fill = ffnosh_next_3m.isna().sum().sum()
            ffnosh_next_3m = ffnosh_next_3m.ffill().bfill()
            after_fill = ffnosh_next_3m.isna().sum().sum()
            print(f"   ✓ Filled {before_fill - after_fill} missing FFNOSH cells")

        # Calculate float market cap
        float_mcap = price_agg * ffnosh_agg

        # Load the Sector info per SYMBOL
        benchmark_df = pd.read_excel(f"data/datasets/benchmark_weights_carbon_intensity_{period}.xlsx")[['SYMBOL', 'GICS Sector']]
        sector_map = benchmark_df.set_index("SYMBOL")["GICS Sector"].to_dict()
        sector_map_series = pd.Series(sector_map)

        # Check for mismatches
        extra_in_float_mcap = float_mcap.columns.difference(sector_map_series.index)
        extra_in_sector_map = sector_map_series.index.difference(float_mcap.columns)

        if len(extra_in_float_mcap) > 0:
            print(f"\n   ℹ {len(extra_in_float_mcap)} tickers in float_mcap but not in sector map: {extra_in_float_mcap.tolist()}")
        assert extra_in_sector_map.empty, f"Found unexpected items in sector_map_series but not in float_mcap: {extra_in_sector_map.tolist()}"

        # Compute Sector-internal weights daily
        print("\n[5/5] Computing sector portfolios...")
        sector_weights = {}
        for sector in sector_map_series.unique():
            tickers = sector_map_series[sector_map_series == sector].index
            sector_mcap = float_mcap[tickers].sum(axis=1)
            sector_weights[sector] = float_mcap[tickers].div(sector_mcap, axis=0)

        # Take close price adjusted for corporate splits and dividends from yahoo to then calculate daily returns
        adj_close_bt = pd.read_excel(f"data/yahoo/adj_price_yahoo_comp_{period}.xlsx")
        adj_close_bt.index = adj_close_bt.Date
        adj_close_bt.drop(columns='Date', inplace=True)

        nan_columns = adj_close_bt.columns[adj_close_bt.isna().any()]
        assert list(sorted(nan_columns)) == list(sorted(extra_in_float_mcap.tolist()))
        daily_returns_next_3m = adj_close_bt.pct_change().dropna()

        sector_portfolio_returns = {}
        for sector, weights_df in sector_weights.items():
            common_tickers = weights_df.columns.intersection(daily_returns_next_3m.columns)
            common_index = weights_df.index.intersection(daily_returns_next_3m.index)

            w = weights_df.loc[common_index, common_tickers]
            r = daily_returns_next_3m.loc[common_index, common_tickers]

            sector_returns = (w * r).sum(axis=1)
            sector_portfolio_returns[sector] = sector_returns

        # Put all in a single DataFrame
        sector_portfolio_df = pd.DataFrame(sector_portfolio_returns)
        sector_portfolio_df['period'] = period

        all_sector_portfolios.append(sector_portfolio_df)
        print(f"   ✓ Computed returns for {len(sector_portfolio_returns)} sectors")
        print(f"   ✓ Period {period} complete!\n")

    # Concatenate all periods into one DataFrame
    print("\n" + "="*80)
    print("FINALIZING...")
    print("="*80)
    combined_sector_portfolio = pd.concat(all_sector_portfolios, axis=0)
    print(f"✓ Combined {len(all_sector_portfolios)} periods into final dataset")
    print(f"✓ Total rows: {len(combined_sector_portfolio)}, Columns: {combined_sector_portfolio.shape[1]}")
    print("\nDone!")

    # --- Compute Annualized Volatility by Sector (2021–2023) ---

    # Drop 'period' column temporarily
    returns_df = combined_sector_portfolio.drop(columns='period')

    # Daily standard deviation of each sector's return
    daily_vol = returns_df.std(skipna=True)

    # Convert to annualized volatility assuming ~252 trading days per year
    annualized_vol = daily_vol * np.sqrt(252)

    # Sort descending to see the most volatile sectors
    annualized_vol = annualized_vol.sort_values(ascending=False)

    print("\n" + "="*80)
    print("ANNUALIZED SECTOR VOLATILITY (2021–2023)")
    print("="*80)
    print(annualized_vol.round(4))

    # Optional: Save to Excel for reporting
    annualized_vol.to_excel("data/benchmark_returns_volatility/sector_annualized_volatility_2021_2023.xlsx")


if __name__ == "__main__":
    main()