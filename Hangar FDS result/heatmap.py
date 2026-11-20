import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf

# 1. Read the Excel files
df_bottom = pd.read_excel("Bottom tempreture.xlsx")
df_top = pd.read_excel("Top tempreture.xlsx")

def plot_radial_full_surface(df, title, filename):
    # Group by x and z to average out any duplicate coordinates
    df_agg = df.groupby(['x', 'z'], as_index=False)['C'].mean()
    
    x = df_agg['x'].values
    z = df_agg['z'].values
    c = df_agg['C'].values
    
    # 2. Create the dense grid for the ENTIRE surface bounding box (Like the 2nd plot)
    # This uses the outermost sensors to define the edges of the room
    grid_x, grid_z = np.mgrid[x.min():x.max():200j, z.min():z.max():200j]
    
    # 3. Apply the Radial Basis Function (RBF)
    # This ensures the heat diffuses in a circular/radial pattern
    rbf_interpolator = Rbf(x, z, c, function='multiquadric', smooth=5)
    grid_c = rbf_interpolator(grid_x, grid_z)
    
    # 4. Clip the temperatures
    # Because we are calculating deep into the empty corners, we constrain the formula 
    # so it doesn't accidentally drop below your coolest sensor or above your hottest sensor
    grid_c = np.clip(grid_c, c.min(), c.max())
    
    # NOTE: I removed the circular masking step here, so it fills the entire box!
    
    # --- PLOT ---
    plt.figure(figsize=(10, 8))
    
    # Contour fill creates the smooth gradient rings
    contour = plt.contourf(grid_x, grid_z, grid_c, levels=100, cmap='inferno')
    
    # Show the actual sensor locations
    plt.scatter(x, z, c='white', s=8, alpha=0.6, label='Sensor Locations')
    
    # Lock the aspect ratio so the heat rings stay perfectly round circles, not ovals
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Styling
    cbar = plt.colorbar(contour)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20, fontweight='bold')
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('X Coordinate', fontweight='bold', labelpad=10)
    plt.ylabel('Z Coordinate', fontweight='bold', labelpad=10)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save & Show
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Run the function
plot_radial_full_surface(df_bottom, 'Bottom Temperature - Full Radial Surface', 'bottom_radial_full.png')
plot_radial_full_surface(df_top, 'Top Temperature - Full Radial Surface', 'top_radial_full.png')