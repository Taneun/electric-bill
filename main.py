"""
Smart Meter Electricity Usage Analysis
Analyzes 15-minute interval electricity consumption data from Israeli utility smart meters.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime, time

# Set plotting style
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

DAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

def load_and_clean(filepath, start_date=None, end_date=None):
    """
    Load and clean the smart meter CSV file.

    Skips Hebrew header rows and parses the measurement table with:
    - Date column (DD/MM/YYYY format)
    - Time column (H:MM format)
    - Consumption column (kWh)

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with datetime index and consumption column
    """
    print(f"Loading data from {filepath}...")

    # Read the file to find where the actual data starts
    # The data table starts after the separator line with underscores
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the header row (contains תאריך, מועד תחילת הפעימה, צריכה)
    data_start_idx = None
    for idx, line in enumerate(lines):
        if 'תאריך' in line and 'צריכה' in line:
            data_start_idx = idx
            break

    if data_start_idx is None:
        raise ValueError("Could not find the data header row in the CSV file")

    # Read the CSV starting from the header row
    # Use tab separator as the file is tab-delimited
    df = pd.read_csv(
        filepath,
        skiprows=data_start_idx,
        encoding='utf-8',
        sep=',',
        skipinitialspace=True
    )

    # Clean column names (remove extra whitespace)
    df.columns = df.columns.str.strip()

    # Identify the three main columns (handles different possible names)
    date_col = [col for col in df.columns if 'תאריך' in col][0]
    time_col = [col for col in df.columns if 'מועד' in col or 'תחילת' in col][0]
    consumption_col = [col for col in df.columns if 'צריכה' in col][0]

    print(f"Found columns: Date='{date_col}', Time='{time_col}', Consumption='{consumption_col}'")

    # Keep only the relevant columns
    df = df[[date_col, time_col, consumption_col]].copy()
    df.columns = ['date', 'time', 'consumption_kwh']

    # Remove any completely empty rows
    df = df.dropna(how='all')

    # Parse date column (day-first format: DD/MM/YYYY)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

    # Parse time column (H:MM or HH:MM format)
    # Convert to datetime.time objects
    def parse_time(time_str):
        try:
            if pd.isna(time_str):
                return None
            time_str = str(time_str).strip()
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            return time(hour=hour, minute=minute)
        except:
            return None

    df['time_obj'] = df['time'].apply(parse_time)

    # Combine date and time into a single datetime
    def combine_datetime(row):
        if pd.isna(row['date']) or row['time_obj'] is None:
            return None
        return datetime.combine(row['date'].date(), row['time_obj'])

    df['datetime'] = df.apply(combine_datetime, axis=1)

    # Parse consumption as numeric
    df['consumption_kwh'] = pd.to_numeric(df['consumption_kwh'], errors='coerce')

    # Drop rows where datetime or consumption is missing
    initial_rows = len(df)
    df = df.dropna(subset=['datetime', 'consumption_kwh'])
    dropped_rows = initial_rows - len(df)

    if dropped_rows > 0:
        print(f"Warning: Dropped {dropped_rows} rows with missing or malformed data")

    # Set datetime as index and sort
    df = df.set_index('datetime').sort_index()

    # Keep only the consumption column
    df = df[['consumption_kwh']]

    # Filter by date range if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date, dayfirst=True)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date, dayfirst=True)]

    print(f"Loaded {len(df)} measurements from {df.index.min()} to {df.index.max()}")
    print(f"Time resolution: 15 minutes")

    return df


def make_hourly(df):
    """
    Resample 15-minute data to hourly totals.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index and consumption_kwh column

    Returns:
    --------
    pd.DataFrame
        Hourly resampled data with additional datetime components
    """
    print("\nResampling to hourly data...")

    # Resample to hourly by summing the 15-minute intervals
    # Each hour should have 4 readings (0:00, 0:15, 0:30, 0:45)
    # Sum gives total kWh consumed in that hour
    df_hourly = df.resample('H').sum()

    # Fill missing hours with 0
    # This assumes missing data means no consumption rather than missing measurements
    # Alternative: use .fillna(0) or leave as NaN to distinguish missing data
    df_hourly['consumption_kwh'] = df_hourly['consumption_kwh'].fillna(0)

    # Add temporal components for analysis
    df_hourly['date'] = df_hourly.index.date
    df_hourly['day_of_week'] = df_hourly.index.day_name()
    df_hourly['hour'] = df_hourly.index.hour

    print(f"Created {len(df_hourly)} hourly records")

    return df_hourly


def plot_heatmap(df_hourly, save_path=None):
    """
    Create a heatmap showing average hourly usage by day of week with marginal totals.

    Parameters:
    -----------
    df_hourly : pd.DataFrame
        Hourly data with day_of_week and hour columns
    save_path : str, optional
        Path to save the figure
    """
    print("\nCreating day-of-week vs hour heatmap...")

    # Create pivot table: rows=day of week, columns=hour, values=mean kWh
    pivot = df_hourly.pivot_table(
        values='consumption_kwh',
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )

    # Reorder rows by day of week
    pivot = pivot.reindex(DAY_ORDER)

    # Calculate hourly totals (sum across all days for each hour)
    hourly_totals = df_hourly.groupby('hour')['consumption_kwh'].sum()

    # Create figure with gridspec for layout control
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[20, 1], hspace=0.02, wspace=0.02)

    # Top subplot for marginal bar plot
    ax_top = fig.add_subplot(gs[0, 0])
    # Bar positions need to be at 0.5, 1.5, 2.5, ... to align with heatmap cell centers
    bar_positions = np.arange(len(hourly_totals)) + 0.5
    ax_top.bar(bar_positions, hourly_totals.values, color='crimson', alpha=0.7, width=1.0)
    ax_top.set_xlim(0, 24)
    ax_top.set_ylabel('Total kWh', fontsize=10)
    ax_top.set_xticks([])  # Remove x-axis labels
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.grid(axis='y', alpha=0.3)

    # Bottom subplot for heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        pivot,
        annot=False,
        cmap='YlOrRd',
        cbar=False,  # add colorbar separately
        ax=ax_heat
    )

    ax_heat.set_xlabel('Hour of Day', fontsize=12)
    ax_heat.set_ylabel('Day of Week', fontsize=12)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)

    # Add colorbar in the right subplot
    ax_cbar = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(ax_heat.collections[0], cax=ax_cbar)
    cbar.set_label('Average kWh', fontsize=10)

    # Add overall title
    fig.suptitle('Hourly Electricity Usage by Day of Week', fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.92,
             'Average consumption patterns:\nTop: Total usage per hour across all days | Heatmap: Average usage by day and hour',
             ha='center', fontsize=10, style='italic', color='gray')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.show()

    return pivot


def plot_daily_totals(df_hourly, save_path=None):
    """
    Create bar charts showing daily total consumption.

    Parameters:
    -----------
    df_hourly : pd.DataFrame
        Hourly data with date column
    save_path : str, optional
        Path to save the figure
    """
    print("\nCreating daily totals charts...")

    # Calculate daily totals
    daily_totals = df_hourly.groupby('date')['consumption_kwh'].sum().sort_index()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Daily usage over time
    dates = pd.to_datetime(daily_totals.index)
    months = dates.month
    colors = plt.cm.Paired(months % 12)

    daily_totals.plot(kind='bar', ax=ax1, color=colors, width=0.8)
    ax1.set_title('Total Daily Electricity Usage', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Total Usage (kWh)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks([])  # Remove x-axis ticks
    # Add month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_positions = []
    month_labels = []

    for month in sorted(months.unique()):
        month_indices = [i for i, m in enumerate(months) if m == month]
        if month_indices:
            middle_pos = (month_indices[0] + month_indices[-1]) / 2
            month_positions.append(middle_pos)
            month_labels.append(month_names[month - 1])

    for pos, label in zip(month_positions, month_labels):
        ax1.text(pos, ax1.get_ylim()[0], label, ha='center', va='top', fontsize=10, fontweight='bold')


    # Plot 2: Total usage by day of week
    # First, recreate day_of_week for daily data
    daily_with_dow = pd.DataFrame({
        'total_kwh': daily_totals.values,
        'day_of_week': pd.to_datetime(daily_totals.index).day_name()
    })

    # Group by day of week
    dow_totals = daily_with_dow.groupby('day_of_week')['total_kwh'].sum().reindex(DAY_ORDER)

    # Create bar chart
    colors = plt.cm.Set3(range(len(dow_totals)))
    dow_totals.plot(kind='bar', ax=ax2, color=colors, width=0.7)
    ax2.set_title('Total Electricity Usage by Day of Week', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Week', fontsize=12)
    ax2.set_ylabel('Total Usage (kWh)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for i, (day, value) in enumerate(dow_totals.items()):
        ax2.text(i, value, f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved daily totals chart to {save_path}")

    plt.show()

    # Print summary table
    print("\n=== Daily Usage Summary ===")
    print(f"Total consumption: {daily_totals.sum():.2f} kWh")
    print(f"Average daily consumption: {daily_totals.mean():.2f} kWh")
    print(f"Highest usage: {daily_totals.max():.2f} kWh on {daily_totals.idxmax()}")
    print(f"Lowest usage: {daily_totals.min():.2f} kWh on {daily_totals.idxmin()}")

    print("\n=== Usage by Day of Week ===")
    print(dow_totals.to_string())

    return daily_totals, dow_totals


def generate_summary_stats(df_hourly):
    """
    Generate and print summary statistics.

    Parameters:
    -----------
    df_hourly : pd.DataFrame
        Hourly data
    """
    print("\n" + "=" * 60)
    print("ELECTRICITY USAGE SUMMARY STATISTICS")
    print("=" * 60)

    total_kwh = df_hourly['consumption_kwh'].sum()
    print(f"\nTotal consumption: {total_kwh:.2f} kWh")
    print(f"Average hourly consumption: {df_hourly['consumption_kwh'].mean():.3f} kWh")
    print(f"Max hourly consumption: {df_hourly['consumption_kwh'].max():.3f} kWh")
    print(f"Min hourly consumption: {df_hourly['consumption_kwh'].min():.3f} kWh")

    # Peak usage hour
    peak_hour = df_hourly.groupby('hour')['consumption_kwh'].mean()
    print(f"\nPeak usage hour: {peak_hour.idxmax()}:00 ({peak_hour.max():.3f} kWh avg)")
    print(f"Lowest usage hour: {peak_hour.idxmin()}:00 ({peak_hour.min():.3f} kWh avg)")

    # Weekend vs weekday
    weekend_days = ['Saturday', 'Sunday']
    weekend = df_hourly[df_hourly['day_of_week'].isin(weekend_days)]['consumption_kwh'].mean()
    weekday = df_hourly[~df_hourly['day_of_week'].isin(weekend_days)]['consumption_kwh'].mean()
    print(f"\nWeekday average hourly: {weekday:.3f} kWh")
    print(f"Weekend average hourly: {weekend:.3f} kWh")

    print("=" * 60 + "\n")


def main():
    """
    Main execution function demonstrating the analysis pipeline.
    """
    # File path - update this to your actual file location
    filepath = 'electricity_bill.csv'

    # Check if file exists
    if not Path(filepath).exists():
        print(f"Error: File '{filepath}' not found.")
        print("Please update the filepath variable in the main() function.")
        return

    try:
        # Step 1: Load and clean data
        df = load_and_clean(filepath, start_date='01/11/2025', end_date=None)

        # Step 2: Resample to hourly
        df_hourly = make_hourly(df)

        # Step 3: Generate summary statistics
        generate_summary_stats(df_hourly)

        # Step 4: Create heatmap
        pivot = plot_heatmap(df_hourly, save_path='usage_heatmap.png')

        # Step 5: Create daily totals charts
        daily_totals, dow_totals = plot_daily_totals(df_hourly, save_path='daily_totals.png')

        print("\nAnalysis complete!")
        print("Generated visualizations:")
        print("  - usage_heatmap.png")
        print("  - daily_totals.png")

        # Optionally, save the processed data
        df_hourly.to_csv('hourly_usage.csv')
        print("\nSaved processed hourly data to: hourly_usage.csv")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()