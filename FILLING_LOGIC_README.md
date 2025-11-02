# Scope Emissions Filling Logic Documentation

## Overview

This document explains the filling logic used in `merge_scope_emissions.py` for handling missing emissions data in years 2021-2023.

## Goal

Fill missing values in years **2021, 2022, and 2023 ONLY**. Years 2020 and 2024 are never filled - they are only used as source data.

## General Principle

**Priority**: Always prefer the **closest previous year** (forward-fill), then fall back to future years (backward-fill).

---

## Detailed Logic for Each Year

### Filling 2021

If 2021 is missing, try in this order:

1. **2020** ← Closest previous year (preferred)
2. **2022** ← Next closest (1 year forward)
3. **2023** ← 2 years forward
4. **2024** ← 3 years forward (last resort)

**Example scenarios:**
```
2020: 100, 2021: MISSING → Fill 2021 with 100 (from 2020)
2020: MISSING, 2021: MISSING, 2022: 200 → Fill 2021 with 200 (from 2022)
Only 2024: 400 available → Fill 2021 with 400 (from 2024)
```

---

### Filling 2022

If 2022 is missing, try in this order:

1. **2021** (original OR already filled) ← Closest previous year (preferred)
2. **2020** ← Alternative previous year
3. **2023** ← 1 year forward
4. **2024** ← 2 years forward (last resort)

**Important**: Uses the filled 2021 value if 2021 was just filled!

**Example scenarios:**
```
2021: 100, 2022: MISSING → Fill 2022 with 100 (from 2021)
2020: 100, 2021: MISSING, 2022: MISSING
  → First fill 2021 with 100, then fill 2022 with 100 (from filled 2021)
2020: MISSING, 2021: MISSING, 2022: MISSING, 2023: 300
  → Fill 2022 with 300 (from 2023)
```

---

### Filling 2023

If 2023 is missing, try in this order:

1. **2022** (original OR already filled) ← Closest previous year (preferred)
2. **2021** (original OR already filled) ← Alternative previous year
3. **2020** ← Earliest year
4. **2024** ← 1 year forward (last resort)

**Important**: Uses filled 2022 or filled 2021 values if they were just filled!

**Example scenarios:**
```
2022: 200, 2023: MISSING → Fill 2023 with 200 (from 2022)
2020: 100, 2021: MISSING, 2022: MISSING, 2023: MISSING
  → First fill 2021 with 100, then fill 2022 with 100,
     then fill 2023 with 100 (from filled 2022)
Only 2024: 400 available → Fill 2023 with 400 (from 2024)
```

---

## Key Characteristics

### 1. Sequential Filling
The logic fills years in order: **2021 → 2022 → 2023**

This allows later years to use earlier filled values.

### 2. Forward-Fill Priority
Always prefers previous years (carrying forward old data) over future years.

### 3. Cascading Effect

**Example 1**: All years filled from 2020
```
Before: 2020: 100, 2021: MISSING, 2022: MISSING, 2023: MISSING, 2024: 400
After:  2020: 100, 2021: 100,     2022: 100,     2023: 100,     2024: 400
```

**Example 2**: All years filled from 2024
```
Before: 2020: MISSING, 2021: MISSING, 2022: MISSING, 2023: MISSING, 2024: 400
After:  2020: MISSING, 2021: 400,     2022: 400,     2023: 400,     2024: 400
```

### 4. No Modification of 2020 or 2024
These boundary years are never filled, only used as source data.

---

## Complete Example Walkthrough

**Initial Data**:
```
2020: 100
2021: MISSING
2022: MISSING
2023: MISSING
2024: 400
```

**Step 1** - Fill 2021:
- Check 2020? ✓ Has 100 → **Fill 2021 with 100**

**Step 2** - Fill 2022:
- Check 2021? ✓ Has 100 (filled) → **Fill 2022 with 100**

**Step 3** - Fill 2023:
- Check 2022? ✓ Has 100 (filled) → **Fill 2023 with 100**

**Final Result**:
```
2020: 100
2021: 100  (filled from 2020)
2022: 100  (filled from filled 2021)
2023: 100  (filled from filled 2022)
2024: 400
```

**Note**: Even though 2024 has a different value (400), the logic carries forward the 2020 value (100) through all the missing years. This demonstrates the "forward-fill" preference.

---

## Edge Cases

### Case 1: Gap in the middle with data on both sides
```
Before: 2020: 100, 2021: MISSING, 2022: 200, 2023: 300, 2024: 400
After:  2020: 100, 2021: 100,     2022: 200, 2023: 300, 2024: 400
```
2021 gets filled with 2020 (previous year preferred).

### Case 2: Only recent years available
```
Before: 2020: MISSING, 2021: MISSING, 2022: MISSING, 2023: 300, 2024: 400
After:  2020: MISSING, 2021: 300,     2022: 300,     2023: 300, 2024: 400
```
All missing years get filled with 2023 (backward-fill from earliest available).

### Case 3: Alternating missing values
```
Before: 2020: 100, 2021: MISSING, 2022: 200, 2023: MISSING, 2024: 400
After:  2020: 100, 2021: 100,     2022: 200, 2023: 200,     2024: 400
```
- 2021 filled with 2020
- 2023 filled with 2022 (previous year)

---

## Company Filtering

Before filling, companies are filtered to include only those with **at least one non-missing value in years 2021-2023**.

Companies with data only in 2020 or 2024 (but no data in 2021-2023) are **excluded** from the filled dataset.

---

## Output Files

The script creates two sets of files for each scope:

1. **Regular files**: `scope_X_all_periods.xlsx`
   - Contains all original data (no filling applied)

2. **Filled files**: `scope_X_all_periods_filled.xlsx`
   - Contains filtered companies (with data in 2021-2023)
   - Missing values in 2021-2023 are filled using the logic above
   - Years 2020 and 2024 remain unchanged

---

## Implementation Reference

See `merge_scope_emissions.py`:
- Filling logic: Lines 817-851
- Company filtering: Lines 770-786
- Example scenario printing: Lines 618-731
