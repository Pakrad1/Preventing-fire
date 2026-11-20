import pandas as pd
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

# ==========================================
# 0. GLOBAL SETTINGS (Adjust these values)
# ==========================================
FS_TITLE = 28      
FS_LABEL = 22      
FS_TICK  = 18      
FS_CBAR  = 15      # Font size for colorbar title
Y_ELEVATION = 0.00 

# ==========================================
# 1. READ AND PREPARE DATA
# ==========================================
df = pd.read_excel("Bottom tempreture.xlsx") 
df_agg = df.groupby(['x', 'z'], as_index=False)['C'].mean()

x, z, c = df_agg['x'].values, df_agg['z'].values, df_agg['C'].values
ambient_temp = max(c.min(), 22.0) 

# Boundary ring for decay
angles = np.linspace(0, 2 * np.pi, 60)
x_fake, z_fake = 60 * np.cos(angles), 60 * np.sin(angles)
c_fake = np.full_like(x_fake, ambient_temp)

x_all = np.concatenate((x, x_fake))
z_all = np.concatenate((z, z_fake))
c_all = np.concatenate((c, c_fake))

# ==========================================
# 2. INTERPOLATE
# ==========================================
X_MIN, X_MAX = -50.0, 30.0
Z_MIN, Z_MAX = -11.0, 11.0
grid_x, grid_z = np.mgrid[X_MIN:X_MAX:200j, Z_MIN:Z_MAX:200j]

rbf_interpolator = Rbf(x_all, z_all, c_all, function='quintic', smooth=5)
grid_c = np.clip(rbf_interpolator(grid_x, grid_z), 22.0, c.max())

# ==========================================
# 3. PLOT THE HEAT MAP
# ==========================================
plt.figure(figsize=(14, 10))
contour = plt.contourf(grid_x, grid_z, grid_c, levels=np.linspace(22, 1300, 100), cmap='Reds', extend='max')

plt.gca().set_aspect('equal', adjustable='box')
plt.title('Heat map of floor', fontweight='bold', fontsize=FS_TITLE, pad=20)
plt.xlabel('X Coordinate (m)', fontweight='bold', fontsize=FS_LABEL, labelpad=15)
plt.ylabel('Z Coordinate (m)', fontweight='bold', fontsize=FS_LABEL, labelpad=15)
plt.xticks(fontsize=FS_TICK)
plt.yticks(fontsize=FS_TICK)

# --- RELOCATED COLORBAR LABEL ---
# Position: [left, bottom, width, height]
cax = plt.gca().inset_axes([0.2, 0.15, 0.75, 0.08])
cbar = plt.colorbar(contour, cax=cax, orientation='horizontal')

# Move the label to the TOP of the bar
cbar.ax.set_title('Temperature (°C)', color='black', fontweight='bold', fontsize=FS_CBAR, pad=10)
cbar.ax.xaxis.set_tick_params(color='black', labelcolor='black', labelsize=FS_TICK)

plt.tight_layout()
plt.savefig('Bottom_Heatmap_Relocated_Label.png', dpi=300, bbox_inches='tight')
plt.show()