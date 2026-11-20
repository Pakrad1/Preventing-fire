import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data and clean NaNs
df = pd.read_csv('Hangar_devc (5).csv', skiprows=1)
df = df.dropna(subset=['Time', 'floor', 'top'])

time = df['Time'].values
floor = df['floor'].values
top = df['top'].values

# Perform polynomial regression
degree = 5
z_floor = np.polyfit(time, floor, degree)
p_floor = np.poly1d(z_floor)

z_top = np.polyfit(time, top, degree)
p_top = np.poly1d(z_top)

# Generate smooth lines across the data's timeline
time_lin = np.linspace(min(time), max(time), 500)
floor_reg = p_floor(time_lin)
top_reg = p_top(time_lin)

plt.figure(figsize=(10, 6))

# Plot the scatter points
plt.scatter(time, floor, label='Floor (FDS)', color='blue', alpha=0.4, s=15)
plt.scatter(time, top, label='Top (FDS)', color='red', alpha=0.4, s=15)

# Plot the regression lines
plt.plot(time_lin, floor_reg, label='Floor Regression', color='darkblue', linewidth=2)
plt.plot(time_lin, top_reg, label='Top Regression', color='darkred', linewidth=2)

# Set the limits according to the requests
plt.xlim(0, 1000)      # x-axis from 0 to 1000
plt.ylim(bottom=20)    # y-axis starts at 20

# Axis labels
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Temperature', fontsize=20)

# Make the tick numbers bigger
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Make the legend bigger
plt.legend(fontsize=16)

plt.grid(True)

# Remove top and right border lines (spines)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save the figure
plt.savefig('regression_plot_large_fonts.png')