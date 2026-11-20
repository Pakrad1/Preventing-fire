import pandas as pd
import matplotlib.pyplot as plt

# Load the data skipping the first row of units
df = pd.read_csv('Hangar_hrr.csv', skiprows=1)
df.columns = df.columns.str.strip()

# Create the plot with a large size
plt.figure(figsize=(12, 8))

# Plot Time vs HRR
plt.plot(df['Time'], df['HRR'], color='firebrick', linewidth=2.5)

# Set the X-axis range from 0 to 1000
plt.xlim(0, 1000)
plt.ylim(5000,)

# Eksendeki sağ ve üst çizgileri (spines) kaldır
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set large fonts for readability
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('HRR (kW)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Add a subtle grid and show/save the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('HRR_Results.png')
plt.show()