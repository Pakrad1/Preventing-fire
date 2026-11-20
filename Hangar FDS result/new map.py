import pandas as pd
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

# ==========================================
# 0. GLOBAL SETTINGS (Adjust these values)
# ==========================================
FS_TITLE = 28      # Font size for the Title
FS_LABEL = 22      # Font size for X and Y labels
FS_TICK  = 18      # Font size for axis numbers
FS_CBAR  = 18      # Font size for colorbar label/numbers
Y_ELEVATION = 4.93 # Height elevation for Ansys export

# ==========================================
# 1. READ AND PREPARE DATA
# ==========================================
# Ensure "Top tempreture.xlsx" is in the same folder
df = pd.read_excel("Top tempreture.xlsx") 

# Group by x and z to average out duplicate coordinates
df_agg = df.groupby(['x', 'z'], as_index=False)['C'].mean()

x = df_agg['x'].values
z = df_agg['z'].values
c = df_agg['C'].values

ambient_temp = c.min() # Get the coolest temperature

# --- ADD A FAR-AWAY AMBIENT BOUNDARY RING ---
# Forces the heat to decay to ambient at the edges
angles = np.linspace(0, 2 * np.pi, 60)
x_fake = 60 * np.cos(angles)
z_fake = 60 * np.sin(angles)
c_fake = np.full_like(x_fake, ambient_temp)

# Combine real sensors with the fake boundary sensors
x_all = np.concatenate((x, x_fake))
z_all = np.concatenate((z, z_fake))
c_all = np.concatenate((c, c_fake))

# ==========================================
# 2. DEFINE GEOMETRY AND INTERPOLATE
# ==========================================
X_MIN, X_MAX = -50.0, 30.0
Z_MIN, Z_MAX = -11.0, 11.0

# Create the dense 200x200 grid
grid_x, grid_z = np.mgrid[X_MIN:X_MAX:200j, Z_MIN:Z_MAX:200j]

# Apply Radial Basis Function (Linear for stability)
rbf_interpolator = Rbf(x_all, z_all, c_all, function='linear')
grid_c = rbf_interpolator(grid_x, grid_z)

# Clip temperatures to stay within physical bounds
grid_c = np.clip(grid_c, ambient_temp, c.max())

# ==========================================
# 3. EXPORT TO CSV FOR ANSYS
# ==========================================
x_flat = grid_x.flatten()
z_flat = grid_z.flatten()
c_flat = grid_c.flatten()
y_flat = np.full_like(x_flat, Y_ELEVATION)

export_df = pd.DataFrame({'x': x_flat, 'y': y_flat, 'z': z_flat, 'C': c_flat})
filename = 'ANSYS_Dense_Top_Temperature_Fixed.csv'
export_df.to_csv(filename, index=False)
print(f"Success! Saved {len(export_df)} points to {filename}")

# ==========================================
# 4. PLOT THE HEAT MAP
# ==========================================
print("Generating heat map plot...")
plt.figure(figsize=(14, 10))

# Plot contours with a smooth natural gradient (inferno)
contour = plt.contourf(grid_x, grid_z, grid_c, levels=np.linspace(22, 1300, 100), cmap='Reds', extend='max')

plt.gca().set_aspect('equal', adjustable='box')

# SET TITLES AND AXIS LABELS WITH ADJUSTABLE FONTS
plt.title('Heat map of ceiling', fontweight='bold', fontsize=FS_TITLE, pad=20)
plt.xlabel('X Coordinate (m)', fontweight='bold', fontsize=FS_LABEL, labelpad=15)
plt.ylabel('Z Coordinate (m)', fontweight='bold', fontsize=FS_LABEL, labelpad=15)

# SET TICK LABEL SIZES
plt.xticks(fontsize=FS_TICK)
plt.yticks(fontsize=FS_TICK)

# INSET COLORBAR SETTINGS
# [left, bottom, width, height] as fractions of plot
cax = plt.gca().inset_axes([0.2, 0.15, 0.75, 0.08])
cbar = plt.colorbar(contour, cax=cax, orientation='horizontal')

# COLORBAR FONT SETTINGS
cbar.ax.set_title('Temperature (°C)', color='black', fontweight='bold', fontsize=FS_CBAR, pad=10)
cbar.ax.xaxis.set_tick_params(color='black', labelcolor='black', labelsize=FS_TICK)


plt.tight_layout()

# Save and Show
plot_filename = 'ANSYS_Dense_Top_Heatmap_LargeFonts.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Success! Saved plot to {plot_filename}")
plt.show()