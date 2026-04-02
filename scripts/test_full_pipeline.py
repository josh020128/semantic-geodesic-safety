import cv2
import numpy as np
import mujoco

# Import all the modules we've built!
from semantic_safety.perception_2d3d.mujoco_camera import MujocoCamera
from semantic_safety.perception_2d3d.lang_sam_wrapper import SemanticPerception
from semantic_safety.perception_2d3d.deproject_3d import CameraGeometry
from semantic_safety.perception_2d3d.transform import WorldTransform

def run_pipeline():
    print("--- STEP 1: RENDERING PHYSICS SCENE ---")
    model = mujoco.MjModel.from_xml_path("tabletop.xml")
    data = mujoco.MjData(model)
    
    # Let gravity settle the objects
    for _ in range(50):
        mujoco.mj_step(model, data)

    # Initialize our headless camera wrapper
    camera = MujocoCamera(model, data, cam_name="main_cam", width=640, height=480)
    color_image, depth_metric, intrinsics = camera.get_frames()
    
    # Save the image for the AI to read
    bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_rgb.png", bgr_image)
    print("Render complete. Depth map and image saved to memory.")

    print("\n--- STEP 2: SEMANTIC AI DETECTION ---")
    ai_detector = SemanticPerception()
    target = ["power drill"]
    
    results = ai_detector.detect_objects("test_rgb.png", target)
    
    if not results:
        print("Pipeline Failed: AI could not locate the power drill.")
        return
        
    # Grab the highest confidence result
    best_result = results[0]
    box = best_result['box']
    
    # Calculate the center pixel (u, v) of the bounding box
    center_u = int((box['xmin'] + box['xmax']) / 2)
    center_v = int((box['ymin'] + box['ymax']) / 2)
    
    print(f"Found '{best_result['label']}'!")
    print(f"2D Pixel Coordinate: u={center_u}, v={center_v}")

    print("\n--- STEP 3: 3D DEPROJECTION ---")
    geom_engine = CameraGeometry(intrinsics)
    
    # Convert that 2D pixel back into 3D space!
    point_3d = geom_engine.get_3d_coordinate(center_u, center_v, depth_metric)
    
    if point_3d is not None:
        print(f"SUCCESS! The power drill is located at Camera Coordinates:")
        print(f"  X: {point_3d[0]:.3f} meters (Left/Right)")
        print(f"  Y: {point_3d[1]:.3f} meters (Up/Down)")
        print(f"  Z: {point_3d[2]:.3f} meters (Distance from Lens)")
    else:
        print("Failed to deproject point (hit skybox).")

    print("\n--- STEP 4: WORLD TRANSFORMATION ---")
    # Get the camera's ID so we can pull its extrinsics from the MuJoCo data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
    
    # Initialize the transform engine with the camera's global position and rotation matrix
    world_engine = WorldTransform(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    
    # Execute the transformation!
    world_pt = world_engine.to_world(point_3d)
    
    print(f"GLOBAL COORDINATES (Robot's Map):")
    print(f"  X: {world_pt[0]:.3f} meters")
    print(f"  Y: {world_pt[1]:.3f} meters")
    print(f"  Z: {world_pt[2]:.3f} meters (Height off the floor)")

if __name__ == "__main__":
    run_pipeline()