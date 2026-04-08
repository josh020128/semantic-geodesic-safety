import numpy as np

data = np.load("loop1_risk_field.npz")

V = data["risk_field"]
x = data["x"]
y = data["y"]
z = data["z"]
bbox = data["bbox"]

print("shape:", V.shape)
print("min/max:", float(V.min()), float(V.max()))
print("mean:", float(V.mean()))
print("p90:", float(np.percentile(V, 90)))
print("p95:", float(np.percentile(V, 95)))
print("p99:", float(np.percentile(V, 99)))
print("p99.5:", float(np.percentile(V, 99.5)))
print("fraction > 2:", float(np.mean(V > 2.0)))
print("fraction > 5:", float(np.mean(V > 5.0)))
print("fraction > 10:", float(np.mean(V > 10.0)))

z_top = float(bbox[5])

above_idx = z >= z_top
below_idx = z < z_top

print("mean above object:", float(V[:, :, above_idx].mean()))
print("mean below object:", float(V[:, :, below_idx].mean()))

# Optional: center-column diagnostics
xc = 0.5 * (bbox[0] + bbox[1])
yc = 0.5 * (bbox[2] + bbox[3])
ix = int(np.argmin(np.abs(x - xc)))
iy = int(np.argmin(np.abs(y - yc)))

print("center column max:", float(V[ix, iy, :].max()))
print("center column mean above:", float(V[ix, iy, above_idx].mean()))
print("center column mean below:", float(V[ix, iy, below_idx].mean()))