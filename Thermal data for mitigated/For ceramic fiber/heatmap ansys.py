import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ==========================================
# 0. GLOBAL SETTINGS (Adjust these values)
# ==========================================
FS_TITLE = 28      
FS_LABEL = 22      
FS_TICK  = 18      
FS_CBAR  = 18      
Y_ELEVATION = 4.93 

MAX_NODE_DISTANCE = 4.0 

# ==========================================
# 1. READ AND PREPARE DATA
# ==========================================
print("Loading data...")
df = pd.read_csv("temp.txt", sep='\t', encoding='latin1') 

df = df.rename(columns={
    df.columns[1]: 'x',
    df.columns[3]: 'z',
    df.columns[4]: 'C'
})

df = df[['x', 'z', 'C']]
df_agg = df.groupby(['x', 'z'], as_index=False)['C'].mean()

x = df_agg['x'].values
z = df_agg['z'].values
c = df_agg['C'].values

# --- DYNAMIC TEMPERATURE SCALING ---
c_min = c.min() 
c_max = c.max()

# Failsafe: if the whole deck is exactly the same temperature, give the scale a 1-degree range
if c_min == c_max:
    c_max = c_min + 1.0

# --- ADD A FAR-AWAY AMBIENT BOUNDARY RING ---
angles = np.linspace(0, 2 * np.pi, 60)
x_fake = 100 * np.cos(angles) 
z_fake = 100 * np.sin(angles)
c_fake = np.full_like(x_fake, c_min)

x_all = np.concatenate((x, x_fake))
z_all = np.concatenate((z, z_fake))
c_all = np.concatenate((c, c_fake))

# ==========================================
# 2. DEFINE DYNAMIC GEOMETRY AND INTERPOLATE
# ==========================================
print("Interpolating grid (Fast Mode)...")
X_MIN, X_MAX = x.min() - 1.0, x.max() + 1.0
Z_MIN, Z_MAX = z.min() - 1.0, z.max() + 1.0

grid_x, grid_z = np.mgrid[X_MIN:X_MAX:200j, Z_MIN:Z_MAX:200j]

grid_c = griddata(
    points=np.c_[x_all, z_all], 
    values=c_all, 
    xi=(grid_x, grid_z), 
    method='linear',
    fill_value=c_min 
)

grid_c = np.clip(grid_c, c_min, c_max)

# ==========================================
# 3. MASK CAVITIES USING KD-TREE
# ==========================================
print("Punching holes for cavities...")
tree = cKDTree(np.c_[x, z])
grid_points = np.c_[grid_x.ravel(), grid_z.ravel()]

distances, _ = tree.query(grid_points)
distances_grid = distances.reshape(grid_x.shape)

grid_c[distances_grid > MAX_NODE_DISTANCE] = np.nan

# ==========================================
# 4. EXPORT TO CSV FOR ANSYS
# ==========================================
print("Exporting to ANSYS...")
x_flat = grid_x.flatten()
z_flat = grid_z.flatten()
c_flat = grid_c.flatten()

valid_mask = ~np.isnan(c_flat)

x_valid = x_flat[valid_mask]
z_valid = z_flat[valid_mask]
c_valid = c_flat[valid_mask]
y_valid = np.full_like(x_valid, Y_ELEVATION)

export_df = pd.DataFrame({'x': x_valid, 'y': y_valid, 'z': z_valid, 'C': c_valid})
filename = 'ANSYS_Dense_Top_Temperature_Fixed.csv'
export_df.to_csv(filename, index=False)
print(f"Success! Saved {len(export_df)} valid points to {filename} (Cavities removed)")

# ==========================================
# 5. PLOT THE HEAT MAP
# ==========================================
print("Generating heat map plot...")
fig, ax = plt.subplots(figsize=(14, 10))

# Ensures cavities are pure white
ax.set_facecolor('white')

if np.isnan(grid_c).all():
    print("\nERROR: All data was erased! Your MAX_NODE_DISTANCE is still too small for your sensor spacing.")
else:
    # Use dynamic min/max for 100 smooth gradient levels
    dynamic_levels = np.linspace(c_min, c_max, 100)
    
    contour = ax.contourf(grid_x, grid_z, grid_c, levels=dynamic_levels, cmap='Reds', extend='both')

    ax.set_aspect('equal', adjustable='box')

    ax.set_title('Heat map of the mitigated floor ', fontweight='bold', fontsize=FS_TITLE, pad=20)
    ax.set_xlabel('X Coordinate (m)', fontweight='bold', fontsize=FS_LABEL, labelpad=15)
    ax.set_ylabel('Z Coordinate (m)', fontweight='bold', fontsize=FS_LABEL, labelpad=15)

    ax.tick_params(axis='both', which='major', labelsize=FS_TICK)

    cax = ax.inset_axes([0.2, 0.15, 0.75, 0.08])
    
    # Generate 5 evenly spaced dynamic tick numbers based on your exact min/max data, rounded to 1 decimal
    dynamic_ticks = np.round(np.linspace(c_min, c_max, 5), 1)
    
    cbar = plt.colorbar(contour, cax=cax, orientation='horizontal', ticks=dynamic_ticks)

    cbar.ax.set_title('Temperature (°C)', color='black', fontweight='bold', fontsize=FS_CBAR, pad=10)
    cbar.ax.xaxis.set_tick_params(color='black', labelcolor='black', labelsize=FS_TICK)

    plt.tight_layout()

    plot_filename = 'ANSYS_Dense_Top_Heatmap_Cavities.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved plot to {plot_filename}")
    plt.show()