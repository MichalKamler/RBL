import math
import numpy as np


class LloydBasedAlgorithm:

    def __init__(self, robot_pos, radius, step_size, k, encumbrance, size_neighbors, dt, v_max):
        self.radius = radius                        # cell dimension (half the sensing radius)
        self.step_size = step_size                  # spece discretization step
        self.k  = k                                 # parameter k_p
        self.encumbrance = encumbrance              # robot encumbrance
        self.time_step = dt                         # time step
        self.size_neighbors_unfiltered = size_neighbors       # neighbours' encumbrance
        self.robot_pos = robot_pos                  # initial robot position
        self.v_max = v_max                          # max velocity of robots

    def aggregate(self, neighbors, beta, destination):
        self.neighbors = neighbors
        self.beta = beta
        self.destination = destination
        self.filter_neighbors()

    def filter_neighbors(self):
        neighbor_positions = np.array(self.neighbors)
        robot_position = np.array(self.robot_pos)

        distances = np.linalg.norm(neighbor_positions - robot_position, axis=1)
        valid_indices = np.where(distances <= 2 * self.radius)[0]

        self.neighbors = neighbor_positions[valid_indices].tolist()
        self.size_neighbors = [self.size_neighbors_unfiltered[i] for i in valid_indices]
    
    
    def points_inside_sphere(self):
        x_center, y_center, z_center = self.robot_pos

        x_min = int((x_center - self.radius) / self.step_size)
        x_max = int((x_center + self.radius) / self.step_size)
        y_min = int((y_center - self.radius) / self.step_size)
        y_max = int((y_center + self.radius) / self.step_size)
        z_min = int((z_center - self.radius) / self.step_size)
        z_max = int((z_center + self.radius) / self.step_size)

        x_coords = np.arange(x_min, x_max + 1) * self.step_size
        y_coords = np.arange(y_min, y_max + 1) * self.step_size
        z_coords = np.arange(z_min, z_max + 1) * self.step_size

        x, y, z = np.meshgrid(x_coords, y_coords, z_coords)
        distances = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2 + (z - z_center) ** 2)
        
        valid_indices = np.where(distances <= self.radius)
        points = list(zip(x[valid_indices], y[valid_indices], z[valid_indices]))

        return points

    def find_closest_points(self, points):
        points = np.array(points)
        robot_pos = np.array(self.robot_pos)
        neighbors = np.array(self.neighbors)

        distances_to_robot = np.linalg.norm(points - robot_pos, axis=1)
        distances_to_neighbors = np.linalg.norm(points[:, np.newaxis] - neighbors, axis=2)

        closer_points_mask = np.all(distances_to_robot[:, np.newaxis] < distances_to_neighbors, axis=1)
        closer_points = points[closer_points_mask]

        return closer_points.tolist()

    def compute_scalar_value(self, x_test, y_test, z_test):
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        z_test = np.array(z_test)

        tmp = np.column_stack((x_test - self.destination[0], y_test - self.destination[1], z_test - self.destination[2]))
        distances = np.linalg.norm(tmp, axis=1)       
        scalar_values = np.exp(-distances / self.beta)
        return scalar_values.tolist()
    
    def account_encumbrance(self, points):
        index = []
        robot_x, robot_y, robot_z = self.robot_pos

        for j, neighbor in enumerate(self.neighbors):
            delta_x = robot_x - neighbor[0]
            delta_y = robot_y - neighbor[1]
            delta_z = robot_z - neighbor[2]

            # avoid division by 0
            delta_x = delta_x if abs(delta_x) >= 0.001 else 0.001
            delta_y = delta_y if abs(delta_y) >= 0.001 else 0.001
            delta_z = delta_z if abs(delta_z) >= 0.001 else 0.001

            #midpoint and distance from midpoint
            xm = 0.5 * (robot_x + neighbor[0])
            ym = 0.5 * (robot_y + neighbor[1])
            zm = 0.5 * (robot_z + neighbor[2])
            dm = np.linalg.norm([xm - robot_x, ym - robot_y, zm-robot_z])

            if dm < self.size_neighbors[j] + self.encumbrance:
                uvec = np.array([delta_x, delta_y, delta_z]) / np.linalg.norm([delta_x, delta_y, delta_z])
                solx = xm + (self.size_neighbors[j] + self.encumbrance - dm) * uvec[0]
                soly = ym + (self.size_neighbors[j] + self.encumbrance - dm) * uvec[1]
                solz = zm + (self.size_neighbors[j] + self.encumbrance - dm) * uvec[2]

                # Compute signed distance for the robot
                robot_point = np.array([robot_x, robot_y, robot_z])
                plane_point = np.array([solx, soly, solz])
                robot_distance = np.dot(uvec, robot_point - plane_point)

                for i, point in enumerate(points):
                    p = np.array([point[0], point[1], point[2]])
                    point_distance = np.dot(uvec, p - plane_point)

                    if robot_distance > 0:
                        # If the robot is on the positive side, exclude points on the negative side
                        if point_distance < 0:
                            index.append(i)
                    else:
                        # If the robot is on the negative side, exclude points on the positive side
                        if point_distance > 0:
                            index.append(i)

                # for i, point in enumerate(points):
                #     p = np.array([point[0], point[1], point[2]])
                #     p0 = np.array([solx, soly, solz])
                #     n = uvec

                #     if np.dot(n, p-p0)>0:
                #         index.append(i)

        new_points = [point for i, point in enumerate(points) if i not in index]
        return new_points
       
    def get_centroid(self):
        # Get points inside the sphere
        sphere_points = self.points_inside_sphere()

        if len(self.neighbors) > 0:
            # Compute the Voronoi cell
            voronoi_sphere_intersection = self.find_closest_points(sphere_points)
            # Account encumbrance
            voronoi_circle_intersection_and_encumbrance = self.account_encumbrance(voronoi_sphere_intersection)
            if not voronoi_circle_intersection_and_encumbrance:
                voronoi_circle_intersection_and_encumbrance = [self.robot_pos]
            x_in, y_in, z_in = zip(*voronoi_circle_intersection_and_encumbrance)
        else:
            x_in, y_in, z_in = zip(*sphere_points)
        
        x_in_no_neigh, y_in_no_neigh, z_in_no_neigh = zip(*sphere_points)

        # Compute scalar values
        scalar_values = self.compute_scalar_value(x_in, y_in, z_in)
        scalar_values_no_neigh = self.compute_scalar_value(x_in_no_neigh, y_in_no_neigh, z_in_no_neigh)

        # Convert x_in, y_in, scalar_values to NumPy arrays
        x_in = np.array(x_in)
        y_in = np.array(y_in)
        z_in = np.array(z_in)
        scalar_values = np.array(scalar_values)

        # Compute the weighted centroid
        centroid = np.array([np.sum(x_in * scalar_values) / np.sum(scalar_values), 
                             np.sum(y_in * scalar_values) / np.sum(scalar_values), 
                             np.sum(z_in * scalar_values) / np.sum(scalar_values)])

        # Convert x_in_no_neigh, y_in_no_neigh, scalar_values_no_neigh to NumPy arrays
        x_in_no_neigh = np.array(x_in_no_neigh)
        y_in_no_neigh = np.array(y_in_no_neigh)
        z_in_no_neigh = np.array(z_in_no_neigh)
        scalar_values_no_neigh = np.array(scalar_values_no_neigh)

        # Compute the centroid without neighbors
        centroid_no_neighbors = np.array([np.sum(x_in_no_neigh * scalar_values_no_neigh) / np.sum(scalar_values_no_neigh), 
                                          np.sum(y_in_no_neigh * scalar_values_no_neigh) / np.sum(scalar_values_no_neigh), 
                                          np.sum(z_in_no_neigh * scalar_values_no_neigh) / np.sum(scalar_values_no_neigh)])

        return centroid, centroid_no_neighbors
       
    def compute_control(self):
        centroid, _ = self.get_centroid()
        u = -self.k * (np.array(self.robot_pos) - np.array(centroid))
        return u if np.linalg.norm(u) <= self.v_max else u / np.linalg.norm(u) * self.v_max
      
    def move(self):
        x, y, z = self.robot_pos  
        velocity = self.compute_control()
        velocity_x, velocity_y, velocity_z = velocity
        next_x =  x +  velocity_x * self.time_step
        next_y =  y +  velocity_y * self.time_step
        next_z =  z +  velocity_z * self.time_step
        self.robot_pos = next_x, next_y, next_z
        return next_x, next_y, next_z



# def compute_centroid(x, y, scalar_values):
#     total_weight = np.sum(scalar_values)
#     centroid_x = np.sum(x * scalar_values) / total_weight
#     centroid_y = np.sum(y * scalar_values) / total_weight
#     return centroid_x, centroid_y

def applyrules(j, P, beta, current_position, c1, c2, th, goal, Robots, c1_no_rotation, d2, d4):
    c1_j = np.array(c1[j])
    current_j = np.array(current_position[j])

    # first condition
    dist_c1_c2 = np.linalg.norm(c1_j - np.array(c2[j]))
    if dist_c1_c2 > d2 and np.linalg.norm(current_j - c1_j) < P["d1"]:
        beta[j] = max(beta[j] - P["dt"], P["beta_min"])
    else:
        beta[j] = beta[j] - P["dt"] * (beta[j] - P["betaD"][j])

    # second condition
    dist_c1_c2_d4 = dist_c1_c2 > d4
    if dist_c1_c2_d4 and np.linalg.norm(current_j - c1_j) < P["d3"]:
        th[j] = min(th[j] + P["dt"], math.pi / 2)
    else:
        th[j] = max(0, th[j] - P["dt"])

    # third condition
    if th[j] == math.pi / 2 and np.linalg.norm(current_j - np.array(c1_no_rotation[j])) > np.linalg.norm(current_j - c1_j):
        th[j] = 0

    # Compute the direction vector and distance in 3D
    delta_x = goal[j][0] - current_j[0]
    delta_y = goal[j][1] - current_j[1]
    delta_z = goal[j][2] - current_j[2]
    distance = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Compute the spherical angles
    theta = math.atan2(delta_y, delta_x)  # Azimuthal angle in the xy-plane
    phi = math.acos(delta_z / distance)  # Polar angle from the z-axis

    # Adjust angles based on the robot's heading
    new_theta = theta - th[j]

    # Compute the new destination in 3D
    Robots.destinations[j][0] = current_j[0] + distance * math.sin(phi) * math.cos(new_theta)
    Robots.destinations[j][1] = current_j[1] + distance * math.sin(phi) * math.sin(new_theta)
    Robots.destinations[j][2] = current_j[2] + distance * math.cos(phi)


    # angle = math.atan2(goal[j][1] - current_j[1], goal[j][0] - current_j[0])
    # new_angle = angle - th[j]
    # distance = math.sqrt((goal[j][0] - current_j[0]) ** 2 + (goal[j][1] - current_j[1]) ** 2)
    # Robots.destinations[j][0] = current_j[0] + distance * math.cos(new_angle)
    # Robots.destinations[j][1] = current_j[1] + distance * math.sin(new_angle)

