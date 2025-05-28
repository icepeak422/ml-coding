## Given a numpy array of 2d points, and a rectangular box in a global coordinate frame, 
## write a function that returns all points that lie within the box.

## The rectangle is made up of 4 vertices and ordered clockwise from top left to bottom left.
## All Points are ordered by [x, y], where the positive x axis points up, and the positive y axis points right from the origin.

import numpy as np

def get_yaw_from_rectangle(vertices: np.ndarray) -> float:
    return np.arctan2(vertices[0, 1] - vertices[3, 1], vertices[0, 0] - vertices[3, 0])


def get_points_in_axis_aligned_rectangle(points: np.ndarray, rectangle: np.ndarray) -> np.ndarray:
    x_min = np.min(rectangle[:,0])
    x_max = np.max(rectangle[:,0])
    y_min = np.min(rectangle[:,1])
    y_max = np.max(rectangle[:,1])
    points = points[points[:,0]>=x_min]
    points = points[points[:,0]<=x_max]
    points = points[points[:,1]>=y_min]
    points = points[points[:,1]<=y_max]
    return points

##### Implement this function ######
def get_points_in_rectangle(points: np.ndarray, rectangle: np.ndarray) -> np.ndarray:
    ## find the rotate matrix
    yaw = get_yaw_from_rectangle(rectangle)
    # The rotation matrix for a 2D point (x, y) is given by:
    # [cos(yaw) -sin(yaw)]
    # [sin(yaw)  cos(yaw)]
    # where yaw is the angle of rotation in radians.
    rotation_matrix = np.eye(2)
    rotation_matrix[0, 0] = np.cos(yaw)
    rotation_matrix[0, 1] = -np.sin(yaw)
    rotation_matrix[1, 0] = np.sin(yaw)
    rotation_matrix[1, 1] = np.cos(yaw)
    ## rotate the points
    rotated_pts = rotation_matrix @ points.T
    rotated_pts = rotated_pts.T
    print(rotated_pts)
    rotated_pts_in_rec = get_points_in_axis_aligned_rectangle(rotated_pts,rectangle)
    rotation_matrix2 = np.eye(2)
    rotation_matrix2[0, 0] = np.cos(yaw)
    rotation_matrix2[0, 1] = np.sin(yaw)
    rotation_matrix2[1, 0] = -np.sin(yaw)
    rotation_matrix2[1, 1] = np.cos(yaw)
    rotated_pts_in_rec = rotation_matrix2 @ rotated_pts_in_rec.T
    rotated_pts_in_rec = rotated_pts_in_rec.T 
    return rotated_pts_in_rec


######### Testing #########
input_points = np.ndarray([100, 2])
count = 0
## Input points, uniform sampling between [0,0] and [10, 10] with all points separated by 1 unit along x and y axes.
for x_idx in range(0,10):
    for y_idx in range(0,10):
        input_points[count][0] = float(x_idx)
        input_points[count][1] = float(y_idx)
        count+=1




##### Test Case 1: Axis Oriented Rectangle #######
#
#       |      (10, 5)    (10, 10) 
#       |           __________
#       |           | .   . .|. . 
#       |           | . .  . |  . 
# x axis|  ..  .    |. .   . |.    
#       |           |__._._._|..  
#       |       (5, 5).   (5, 10)
#       |
#       |
#       |_________________________________
#                       y axis

# rectangle = np.array([[10,5], [10,10], [5,10], [5,5]])
# output_points = get_points_in_rectangle(input_points, rectangle)
# print("Number of points in the rectangle: ", output_points.shape)
# print("Points within the rectangle: ", output_points)



##### Test Case 2: Non Axis Oriented Rectangle #######

#       |      (10, 5)     
#       |       /^-_ 
#       | .    /. .    ^-_   (5, 10)
#       |     /   .  .    / 
#       |    / .   .     /
# x axis|   / .      .  /. . 
#       |  /  .  . .   /  . 
#  (5,0)| ^-_    ..   /
#       |     ^- _   /  (0, 5)
#       |___________/_____________________
#                   y axis
rectangle = np.array([[10,5], [5,10], [0,5], [5,0]])
output_points = get_points_in_rectangle(input_points, rectangle)
print("Number of points in the rectangle: ", output_points.shape)
print("Points within the rectangle: ", output_points)
