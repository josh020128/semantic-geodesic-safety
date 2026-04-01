import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for servers
import matplotlib.pyplot as plt

from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.interpolation import compute_directional_weights, compute_hazard_field

def create_scene(workspace):
    """Creates boolean mask and bounding box for the Scene Object (Laptop)."""
    X, Y, Z = workspace.X, workspace.Y, workspace.Z
    
    # A flat Laptop in the center
    laptop_mask = (X > -0.2) & (X < 0.2) & (Y > -0.15) & (Y < 0.15) & (Z >= 0.0) & (Z < 0.05)
    
    # Bounding Box: (xmin, xmax, ymin, ymax, zmin, zmax)
    laptop_bbox = (-0.2, 0.2, -0.15, 0.15, 0.0, 0.05)
    
    return laptop_mask, laptop_bbox

def run_demo():
    print("Initializing 3D Workspace...")
    workspace = WorkspaceGrid(bounds=(-0.5, 0.5, -0.5, 0.5, 0.0, 1.0), resolution=0.01)
    
    laptop_mask, laptop_bbox = create_scene(workspace)
    
    # The laptop is the only solid physical obstacle in the scene
    occupancy_grid = ~laptop_mask

    # ---------------------------------------------------------
    # HAZARD: THE LAPTOP (Vulnerable to water from above: +z)
    # ---------------------------------------------------------
    print("Computing Laptop Risk Field (Upward Vertical Cone)...")
    laptop_weights = {'w_+x': 0.0, 'w_-x': 0.0, 'w_+y': 0.0, 'w_-y': 0.0, 'w_+z': 1.0, 'w_-z': 0.0}
    
    dist_laptop = workspace.compute_geodesic_distance(laptop_mask, occupancy_grid)
    W_laptop = compute_directional_weights(workspace.X, workspace.Y, workspace.Z, laptop_bbox, laptop_weights)
    
    # Calculate the final risk field for the laptop
    V_laptop = compute_hazard_field(W_laptop, dist_laptop, gamma=1.0, alpha=15.0)

    # ---------------------------------------------------------
    # VISUALIZATION: Slice the room down the middle (Y=0)
    # ---------------------------------------------------------
    print("Plotting results...")
    y_center_idx = workspace.shape[1] // 2
    
    X_slice = workspace.X[:, y_center_idx, :]
    Z_slice = workspace.Z[:, y_center_idx, :]
    V_slice = V_laptop[:, y_center_idx, :]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the risk heatmap
    contour = plt.contourf(X_slice, Z_slice, V_slice, levels=50, cmap='inferno')
    plt.colorbar(contour, label='Risk Cost V(x)')
    
    # Draw the physical laptop boundary
    plt.contour(X_slice, Z_slice, laptop_mask[:, y_center_idx, :], levels=[0.5], colors='white', linewidths=2)
    
    # Add titles and context descriptions
    plt.title("Semantic Geodesic Risk Field (XZ Plane)", fontsize=14)
    plt.xlabel("X-axis (meters)")
    plt.ylabel("Z-axis (meters, Height)")
    plt.text(-0.05, 0.02, "Laptop", color='black', fontweight='bold', backgroundcolor='white')
    
    # Add a descriptive box explaining the manipulated object
    context_text = "Manipulated Object: Cup of Water (Not Shown)\nScene Object: Laptop\nTopology: Upward Vertical Cone"
    plt.text(-0.45, 0.85, context_text, color='white', fontsize=10, 
             bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    output_path = "risk_field_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Math engine visualized. Open '{output_path}' in Cursor to view.")

if __name__ == "__main__":
    run_demo()