import numpy as np

R = np.random.randn(3,3)
T = np.random.randn(3,1)
fx, fy = 800, 800
cx, cy = 640, 360
I = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])

points = np.random.randn(10,3)

def project_points(R, T, I, points):
    # automatically broadcast
    camera_projection = R @ points.T + T
    camera_projection = camera_projection / camera_projection[2,:]
    camera_projection = I @ camera_projection
    return camera_projection[:2,:]

camera_projection = project_points(R, T, I, points)
print(camera_projection.T)

def project_points_fast(R, T, I, points):
    Transform = np.concatenate([R, T], axis=1)
    pad = np.ones((points.shape[0],1))
    points_4d = np.concatenate([points, pad], axis=1)
    camera_projection = Transform @ points_4d.T
    camera_projection = camera_projection / camera_projection[2,:]
    camera_projection = I @ camera_projection
    return camera_projection[:2,:]

camera_projection = project_points_fast(R, T, I, points)
print(camera_projection.T)