import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        """
        Initializes the RealSense hardware pipeline.
        Defaulting to 640x480 @ 30fps ensures USB bandwidth stability
        and keeps the 2D->3D matrix math blazingly fast.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable Color and Depth streams
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Start streaming
        self.profile = self.pipeline.start(config)
        
        # CRITICAL HARDWARE MATH: Aligning the sensors
        # The depth sensor and RGB sensor are physically offset by a few centimeters.
        # We must mathematically warp the depth map to match the RGB lens's perspective.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # Extract the camera's intrinsic matrix (focal length, optical center)
        # We will need this later in deproject_3d.py to convert pixels to meters.
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        """
        Pulls the latest aligned frame pair from the hardware buffer.
        Returns:
            color_image (np.ndarray): The 2D RGB matrix for Lang-SAM.
            depth_image (np.ndarray): The 2D Depth matrix (in millimeters).
            depth_frame (rs.frame): The raw PyRealSense object for internal deprojection.
        """
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        
        # Mathematically align the infrared depth map to the RGB image
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
            
        # Convert images to standard NumPy arrays for our pipeline
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image, depth_frame

    def get_depth_scale(self):
        """Returns the conversion factor from depth units to meters (usually 0.001)."""
        depth_sensor = self.profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def stop(self):
        """Safely closes the hardware connection."""
        self.pipeline.stop()