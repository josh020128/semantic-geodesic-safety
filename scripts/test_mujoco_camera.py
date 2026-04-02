import mujoco
import numpy as np
import cv2  # OpenCV for saving the images
import os

# Import the headless camera wrapper we wrote
from semantic_safety.perception_2d3d.mujoco_camera import MujocoCamera

def run_camera_test():
    xml_path = "tabletop.xml"
    
    if not os.path.exists(xml_path):
        print(f"Error: Could not find '{xml_path}'. Make sure it is saved in the root directory.")
        return

    print("Loading MuJoCo Scene...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Step the physics simulation forward 50 frames 
    # This allows gravity to pull the drill and bowl down so they rest naturally on the table
    print("Stepping physics simulation...")
    for _ in range(50):
        mujoco.mj_step(model, data)

    print("Initializing Headless Camera...")
    # Make sure cam_name="main_cam" matches the <camera> tag in your tabletop.xml
    try:
        camera = MujocoCamera(model, data, cam_name="main_cam", width=640, height=480)
    except ValueError as e:
        print(f"Camera Error: {e}")
        return

    print("Capturing RGB-D Frames...")
    color_image, depth_metric, intrinsics = camera.get_frames()

    if color_image is None:
        print("Failed to capture frames.")
        return

    # ---------------------------------------------------------
    # VISUALIZATION & SAVING
    # ---------------------------------------------------------
    # 1. Save the standard RGB image
    # OpenCV expects BGR color channels, but MuJoCo outputs RGB. We must convert it.
    bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_rgb.png", bgr_image)

    # 2. Save the Depth map
    # Depth is currently in pure meters (e.g., 0.5m to 1.5m). 
    # To save it as a visible image, we normalize it to a 0-255 grayscale range.
    depth_normalized = cv2.normalize(depth_metric, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = depth_normalized.astype(np.uint8)
    
    # Apply a JET colormap so closer objects appear red/yellow and far objects appear blue
    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    cv2.imwrite("test_depth.png", depth_colored)

    print("\nSuccess! Render complete. Saved to your root directory:")
    print(" - test_rgb.png")
    print(" - test_depth.png")
    print(f"\nCamera Focal Length (fx, fy): {intrinsics['fx']:.2f}, {intrinsics['fy']:.2f}")

if __name__ == "__main__":
    run_camera_test()