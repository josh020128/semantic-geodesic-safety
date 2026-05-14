import numpy as np

from semantic_safety.ur5_experiment.risk_volume_query import RiskVolumeQuery
from semantic_safety.ur5_experiment.trajectory import (
    TrajectoryProcessor,
    TrajectoryProcessingConfig,
    rotation_matrix_from_rpy,
)

rv = RiskVolumeQuery.from_npz("loop1_risk_field.npz")

raw_path = np.load("out/ur5_planning/risk_aware_path.npy")

processor = TrajectoryProcessor(
    TrajectoryProcessingConfig(
        enable_shortcut=True,
        shortcut_step_m=0.01,
        resample_spacing_m=0.03,
        enable_smoothing=True,
        smoothing_window=3,
    )
)

def point_is_valid(p):
    return rv.is_free_with_margin(p, margin_m=0.02)

# 예시 fixed orientation.
# 나중에 UR5 XML에서 실제 downward tool orientation 기준으로 바꾸면 됨.
fixed_R = rotation_matrix_from_rpy(
    roll=0.0,
    pitch=np.pi,
    yaw=0.0,
)

processed = processor.process(
    raw_path,
    point_is_valid=point_is_valid,
    fixed_rotation=fixed_R,
)

print("raw points:", len(processed.raw_path))
print("simplified points:", len(processed.simplified_path))
print("resampled points:", len(processed.resampled_path))
print("final points:", len(processed.final_path))
print("raw length:", processed.path_length_raw)
print("final length:", processed.path_length_final)

np.save("out/ur5_planning/risk_aware_path_processed.npy", processed.final_path)
np.save("out/ur5_planning/risk_aware_poses.npy", processed.pose_matrices)