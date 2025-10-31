"""Debug script for period 0621"""

import pandas as pd
from create_datasets import DatasetCreator

# Create dataset for period 0621
period = "0323"
creator = DatasetCreator(period, "data")

# Run through the steps to find where the issue occurs
print(f"\n{'='*60}")
print(f"Debugging period: {period}")
print(f"{'='*60}\n")

try:
    # Step 1: Load symbol data
    df = creator.load_symbol_data()
    print(f"After load_symbol_data: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Step 2: Load price and shares
    df = creator.load_price_and_shares(df)
    print(f"After load_price_and_shares: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Step 3: Calculate float mcap
    df = creator.calculate_float_mcap(df)
    print(f"After calculate_float_mcap: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Step 4: Delete duplicates
    df = creator.delete_duplicates(df)
    print(f"After delete_duplicates: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Step 5: Load scope emissions
    df = creator.load_scope_emissions(df)
    print(f"After load_scope_emissions: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Step 6: Impute revenue
    df = creator.impute_revenue_from_delisted(df)
    print(f"After impute_revenue_from_delisted: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Step 7: Impute scope emissions
    df = creator.impute_scope_emissions(df)
    print(f"After impute_scope_emissions: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Step 8: Filter stocks with missing prices
    df = creator.filter_stocks_with_missing_prices(df)
    print(f"After filter_stocks_with_missing_prices: {len(df)} companies")
    print(f"  Missing GICS Sector: {df['GICS Sector'].isna().sum()}")

    # Print companies with missing GICS Sector
    missing_gics = df[df['GICS Sector'].isna()]
    if len(missing_gics) > 0:
        print(f"\n{'='*60}")
        print(f"Companies with missing GICS Sector:")
        print(f"{'='*60}")
        for _, row in missing_gics.iterrows():
            print(f"  NAME: '{row['NAME']}'")
            print(f"  SYMBOL: {row['SYMBOL']}")
            print(f"  TYPE: {row['TYPE']}")
            print(f"  float_mcap: {row.get('float_mcap', 'N/A')}")
            print(f"  ---")

    # Step 9: Calculate weights (this is where the error occurs)
    print(f"\nAttempting to calculate weights...")
    df = creator.calculate_weights(df)

    print(f"\nSUCCESS: Dataset created for period {period}")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
