# Israeli Smart Meter Electricity Usage Analysis

A comprehensive Python tool for analyzing 15-minute interval electricity consumption data exported from Israeli utility smart meters.

## Features

1. **Data Loading and Cleaning**
   - Automatically skips Hebrew customer info headers
   - Parses date (DD/MM/YYYY) and time (H:MM) columns correctly
   - Combines into datetime index with 15-minute resolution
   - Handles missing or malformed data gracefully

2. **Hourly Resampling**
   - Converts 15-minute readings to hourly totals (sum, not average)
   - Fills missing hours with 0 (can be modified for NaN)
   - Maintains data integrity throughout transformation

3. **Day-of-Week and Hourly Pattern Analysis**
   - Heatmap visualization showing average hourly usage by day of week
   - Clear identification of peak usage times
   - Easy-to-read color-coded patterns

4. **Daily Totals and Comparisons**
   - Bar chart of daily consumption over time
   - Summary chart by day of week (all Mondays, Tuesdays, etc.)
   - Exact values displayed on charts

5. **Statistical Summaries**
   - Total and average consumption metrics
   - Peak/low usage identification
   - Weekend vs weekday comparisons

## Requirements

```bash
pip install pandas matplotlib seaborn --break-system-packages
```

## Usage

### Basic Usage

```python
from electricity_analysis import load_and_clean, make_hourly, plot_heatmap, plot_daily_totals

# Load your data
df = load_and_clean('your_electricity_file.csv')

# Convert to hourly
df_hourly = make_hourly(df)

# Generate visualizations
plot_heatmap(df_hourly)
plot_daily_totals(df_hourly)
```

### Command Line

```bash
# Update the filepath in main() function, then run:
python electricity_analysis.py
```

## File Format

The script expects CSV files with this structure:

```
[Hebrew header rows with customer info]
[Separator line with underscores]
תאריך	מועד תחילת הפעימה	צריכה בקוט"ש
29/06/2025	0:00	0.044
29/06/2025	0:15	0.043
...
```

Key points:
- Tab-separated values
- Date in DD/MM/YYYY format
- Time in H:MM or HH:MM format
- Consumption in kWh (numeric)

## Output Files

The script generates:

1. **usage_heatmap.png** - Heatmap showing hourly patterns by day of week
2. **daily_totals.png** - Two bar charts:
   - Daily usage over time
   - Total usage by day of week
3. **hourly_usage.csv** - Processed hourly data for further analysis

## Function Reference

### `load_and_clean(filepath)`
Loads the CSV file, skips headers, parses dates and times, and returns a cleaned DataFrame.

**Parameters:**
- `filepath` (str): Path to the CSV file

**Returns:**
- DataFrame with datetime index and consumption_kwh column

### `make_hourly(df)`
Resamples 15-minute data to hourly totals.

**Parameters:**
- `df` (DataFrame): 15-minute resolution data

**Returns:**
- DataFrame with hourly data and temporal columns (date, day_of_week, hour)

### `plot_heatmap(df_hourly, save_path=None)`
Creates a heatmap showing average hourly usage by day of week.

**Parameters:**
- `df_hourly` (DataFrame): Hourly data
- `save_path` (str, optional): Path to save figure

**Returns:**
- Pivot table with the heatmap data

### `plot_daily_totals(df_hourly, save_path=None)`
Creates bar charts showing daily total consumption.

**Parameters:**
- `df_hourly` (DataFrame): Hourly data
- `save_path` (str, optional): Path to save figure

**Returns:**
- Tuple of (daily_totals, day_of_week_totals) Series

### `generate_summary_stats(df_hourly)`
Prints comprehensive summary statistics to console.

## Customization

### Handling Missing Data

By default, missing hours are filled with 0. To change this:

```python
# In make_hourly() function, modify this line:
df_hourly['consumption_kwh'] = df_hourly['consumption_kwh'].fillna(0)

# To leave as NaN:
# (comment out or remove the fillna line)
```

### Adjusting Plot Styles

Change the plotting style at the top of the script:

```python
plt.style.use('seaborn-v0_8')  # or 'ggplot', 'fivethirtyeight', etc.
```

### Filtering Date Ranges

```python
# After loading data
df_filtered = df['2025-06-01':'2025-06-30']  # June 2025 only
```

## Example Output

### Console Output
```
Loading data from electricity_usage.csv...
Found columns: Date='תאריך', Time='מועד תחילת הפעימה', Consumption='צריכה בקוט"ש'
Loaded 128 measurements from 2025-06-29 00:00:00 to 2025-06-30 07:45:00
Time resolution: 15 minutes

Resampling to hourly data...
Created 32 hourly records

============================================================
ELECTRICITY USAGE SUMMARY STATISTICS
============================================================

Total consumption: 26.71 kWh
Average hourly consumption: 0.835 kWh
Max hourly consumption: 3.748 kWh
Min hourly consumption: 0.096 kWh

Peak usage hour: 18:00 (3.748 kWh avg)
Lowest usage hour: 2:00 (0.102 kWh avg)

Weekday average hourly: 0.577 kWh
Weekend average hourly: 0.921 kWh
```

## Troubleshooting

**Issue:** "Could not find the data header row"
- Check that your file contains the Hebrew column headers (תאריך, מועד תחילת הפעימה, צריכה)

**Issue:** "Dropped X rows with missing data"
- This is normal if some measurements are missing
- Check the CSV for blank or malformed rows

**Issue:** Plots not displaying
- Make sure you have a display available
- Or use `save_path` parameter to save to file instead

## License

Free to use and modify for personal projects.
