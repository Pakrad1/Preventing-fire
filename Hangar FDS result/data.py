import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import re

# --- GLOBAL FONT SETTINGS ---
plt.rcParams.update({
    'font.size': 15,          
    'axes.titlesize': 18,     
    'axes.labelsize': 18,     
    'xtick.labelsize': 18,    
    'ytick.labelsize': 18,    
    'legend.fontsize': 13,    
    'figure.titlesize': 23    
})

print("Loading data...")
df = pd.read_excel('Hangar_devc.xlsx', header=1)

# The first column is Time
time_col = df.columns[0]
time = df[time_col]

# --- PAIRING LOGIC ---
columns = df.columns.tolist()
pairs = []

for col in columns:
    # Skip sprinklers
    if 'sprinkler' in str(col).lower():
        continue
        
    # Find floor columns and their matching top columns
    if 'floor' in col:
        expected_top = col.replace('floor', 'top')
        
        # Check if the exact replacement exists
        if expected_top in columns:
            pairs.append((col, expected_top))
        else:
            # Handle edge cases where spacing is inconsistent (e.g., 'right floor1' -> 'right top 1')
            expected_top_spaced = re.sub(r'top(\d)', r'top \1', expected_top)
            if expected_top_spaced in columns:
                pairs.append((col, expected_top_spaced))
            else:
                print(f"Warning: Could not find a 'top' match for {col}")

print(f"Found {len(pairs)} floor-top pairs to plot.")

# --- PLOTTING LOOP ---
for floor_col, top_col in pairs:
    y_floor = df[floor_col]
    y_top = df[top_col]
    
    # Calculate Means
    mean_floor = y_floor.mean()
    mean_top = y_top.mean()
    
    # Calculate smooth trends (Savitzky-Golay filter)
    try:
        window_f = min(51, len(y_floor) - (len(y_floor) % 2 == 0))
        smooth_floor = savgol_filter(y_floor, window_length=window_f, polyorder=3)
    except:
        smooth_floor = y_floor 
        
    try:
        window_t = min(51, len(y_top) - (len(y_top) % 2 == 0))
        smooth_top = savgol_filter(y_top, window_length=window_t, polyorder=3)
    except:
        smooth_top = y_top 

    # --- CREATE PLOT ---
    plt.figure(figsize=(14, 8))
    
    # 1. Floor Data (Blue colors)
    plt.plot(time, y_floor, label=f'Raw {floor_col}', color='#1f77b4', alpha=0.3, linewidth=1.5)
    plt.plot(time, smooth_floor, label=f'Smooth {floor_col}', color='#1f77b4', linewidth=3)
    plt.axhline(mean_floor, color='#1f77b4', linestyle='--', linewidth=2, label=f'Mean: {mean_floor:.1f}°C')
    
    # 2. Top Data (Red colors)
    plt.plot(time, y_top, label=f'Raw {top_col}', color='#d62728', alpha=0.3, linewidth=1.5)
    plt.plot(time, smooth_top, label=f'Smooth {top_col}', color='#d62728', linewidth=3)
    plt.axhline(mean_top, color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {mean_top:.1f}°C')
    
    # --- STYLING ---
    plt.title(f"{floor_col} vs {top_col}", pad=20, fontweight='bold') 
    plt.xlabel('Time (s)', labelpad=15, fontweight='bold')
    plt.ylabel('Temperature (°C)', labelpad=15, fontweight='bold')
    
    # Clean legend slightly outside the plot area so it doesn't cover data
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    
    # --- SAVE / SHOW ---
    # To save the images automatically, uncomment the line below:
    # plt.savefig(f"{floor_col}_vs_{top_col}.png", bbox_inches='tight')
    
    plt.show()
    
    