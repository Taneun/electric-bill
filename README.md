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

## Customization

### Filtering Date Ranges

```python
# in main function:
df = load_and_clean(filepath, start_date='01/11/2025', end_date=None) # start from 1/11/25
```


## License

Free to use and modify for personal projects.
