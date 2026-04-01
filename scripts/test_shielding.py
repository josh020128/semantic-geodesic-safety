import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for server
import matplotlib.pyplot as plt

from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.interpolation import compute_directional_weights, compute_hazard_field

def create_shielding_scenario(workspace):
    X, Y, Z = workspace.X, workspace.Y, workspace.Z
    
    # 1. The Laptop (Hazard source, sitting on the "floor")
    # Height: Z goes from 0.0 to 0.05
    laptop_mask = (X > -0.15) & (X < 0.15) & (Y > -0.15) & (Y < 0.15) & (Z >= 0.0) & (Z < 0.05)
    # laptop_centroid = np.array([0.0, 0.0, 0.025])
    laptop_bbox = (-0.15, 0.15, -0.15, 0.15, 0.0, 0.05)
    
    # 2. The Table (Lowered to sit just 5cm above the laptop!)
    # Height: Z goes from 0.10 to 0.15
    table_mask = (X > -0.35) & (X < 0.35) & (Y > -0.3) & (Y < 0.3) & (Z >= 0.15) & (Z < 0.20)
    
    return laptop_mask, laptop_bbox, table_mask

def run_shielding_test():
    print("Initializing 3D Workspace...")
    workspace = WorkspaceGrid(bounds=(-0.5, 0.5, -0.5, 0.5, 0.0, 0.8), resolution=0.01)
    
    laptop_mask, laptop_bbox, table_mask = create_shielding_scenario(workspace)
    
    # Both the laptop and table are solid physical objects
    occupancy_grid = ~(laptop_mask | table_mask)

    print("Computing Geodesic FMM (Obstacle-Aware)...")
    laptop_weights = {'w_+x': 0.0, 'w_-x': 0.0, 'w_+y': 0.0, 'w_-y': 0.0, 'w_+z': 1.0, 'w_-z': 0.0}
    
    dist_geodesic = workspace.compute_geodesic_distance(laptop_mask, occupancy_grid)
    W_laptop = compute_directional_weights(workspace.X, workspace.Y, workspace.Z, laptop_bbox, laptop_weights)    
    # Calculate the risk field (using a moderate alpha so we can see the distance decay)
    V_geodesic = compute_hazard_field(W_laptop, dist_geodesic, gamma=1.0, alpha=10.0)

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    print("Plotting results...")
    y_center_idx = workspace.shape[1] // 2
    
    X_slice = workspace.X[:, y_center_idx, :]
    Z_slice = workspace.Z[:, y_center_idx, :]
    V_slice = V_geodesic[:, y_center_idx, :]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the risk heatmap
    contour = plt.contourf(X_slice, Z_slice, V_slice, levels=50, cmap='inferno')
    plt.colorbar(contour, label='Risk Cost V(x)')
    
    # Draw the physical objects
    plt.contour(X_slice, Z_slice, laptop_mask[:, y_center_idx, :], levels=[0.5], colors='white', linewidths=2)
    plt.contour(X_slice, Z_slice, table_mask[:, y_center_idx, :], levels=[0.5], colors='cyan', linewidths=2)
    
    plt.title("Environmental Shielding: Geodesic Risk Field (XZ Plane)", fontsize=14)
    plt.xlabel("X-axis (meters)")
    plt.ylabel("Z-axis (meters, Height)")
    plt.text(-0.06, 0.01, "Laptop", color='black', fontweight='bold', backgroundcolor='white')
    plt.text(-0.05, 0.17, "Table", color='black', fontweight='bold', backgroundcolor='cyan')
    
    plt.tight_layout()
    output_path = "shielding_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Open '{output_path}' in Cursor to view the shielding effect.")

if __name__ == "__main__":
    run_shielding_test()