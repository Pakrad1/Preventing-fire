import pandas as pd
import matplotlib.pyplot as plt

# Load the data, skipping the first row (units/metadata)
df = pd.read_csv('Hangar_devc (5).csv', skiprows=[0])

# ==========================================
#               CONTROL PANEL
# ==========================================
# Change these values to customize your graph:
GRAPH_COLOR = 'red'        # Options: 'blue', 'green', 'orange', etc.
FONT_SIZE_LABELS = 20      # Font size for X and Y axis labels
FONT_SIZE_TICKS = 18       # Font size for the numbers on the axes
LINE_WIDTH = 2             # Thickness of the line
# ==========================================

# 1. Identify the columns (Time and the very last column)
time_col = 'Time'
last_col = df.columns[1]

# 2. Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df[time_col], df[last_col], color=GRAPH_COLOR, linewidth=LINE_WIDTH)
plt.xlim(0, 1000)


# Sağ ve üst çerçeve çizgilerini (spines) kaldır
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3. Apply customizations from the Control Panel
plt.xlabel('Time (s)', fontsize=FONT_SIZE_LABELS)
plt.ylabel('Temperature (°C)', fontsize=FONT_SIZE_LABELS)

# Set the font size for the axis numbers (ticks)
plt.xticks(fontsize=FONT_SIZE_TICKS)
plt.yticks(fontsize=FONT_SIZE_TICKS)

# 4. Add a grid for easier reading
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Final adjustment and save
plt.tight_layout()
plt.savefig('last_column_graph.png')
plt.show()