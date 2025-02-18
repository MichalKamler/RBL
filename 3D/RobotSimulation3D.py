import math
import numpy as np
# import matplotlib.pyplot as plt
from RobotInit import RobotsInit
# from RobotInit1 import RobotsInit1
from Lloydbasedalgorithm import LloydBasedAlgorithm, applyrules
import time
import copy
import pybullet as p
import pybullet_data


# refresh_rate = 0.01

# while True:
#     p.stepSimulation()

#     for sphere_id in sphere_ids:
#         pos, _ = p.getBasePositionAndOrientation(sphere_id)
#         velocity = p.getBaseVelocity(sphere_id)

#         new_velocity = [velocity[0][0] + 1, velocity[0][1], velocity[0][2]]

#         p.resetBaseVelocity(sphere_id, linearVelocity=new_velocity)

#     time.sleep(refresh_rate)


class RobotSimulation3D:
    def __init__(self, parameters):
        self.P = parameters
        self.tmp = 0
        self.c1 = np.zeros((self.P["N"], 3))  # actual centroids
        self.c2 = np.zeros((self.P["N"], 3))  # virtual centroids (without neighbors)
        self.c1_no_rotation = np.zeros((self.P["N"], 3))  # \bar p = e, centroids
        self.c2_no_rotation = np.zeros((self.P["N"], 3))  # \bar p = e, virtual centroids (without neighbors)
        self.c1_no_humans = np.zeros((self.P["N"], 3))
        self.c2_no_humans = np.zeros((self.P["N"], 3))
        self.step = 0
        self.flag = np.zeros(self.P["N"])
        self.file_path = 'test' + str(self.P["h"]) + '.txt'
        self.current_position_x = np.zeros(self.P["N"])
        self.current_position_y = np.zeros(self.P["N"])
        self.current_position_z = np.zeros(self.P["N"])
        self.th = np.zeros(self.P["N"])
        self.flag_convergence = 0
        self.current_position = None
        self.goal = None
        self.Lloyd = None
        self.Lloyd_virtual = None
        self.Robots = None
        self.beta = self.P["betaD"].copy() #[0.5] * self.P["N"]

        self.spheres = []
        self.goal_spheres = []
        self.region_goals = []

        self.record = self.P["record"]
        self.replay_csv = self.P["replay_csv"]

        self.recorded_data = []
        self.sim_running = None


    def initialize_simulation(self):
        self.Robots = RobotsInit(self.P)

        self.current_position = self.Robots.positions
        self.goal = copy.deepcopy(self.Robots.destinations)
        self.Lloyd = [LloydBasedAlgorithm(self.Robots.positions[j], self.P["radius"], self.P["dx"],
                                          self.P["k"][j], self.P["size"][j],
                                          np.delete(self.P["size"], j, axis=0),
                                          self.P["dt"], self.P["v_max"][j]) for j in range(self.P["N"])]

        self.Lloyd_virtual = [LloydBasedAlgorithm(self.Robots.positions[j], self.P["radius"], self.P["dx"],
                                                  self.P["k"][j], self.P["size"][j],
                                                  np.delete(self.P["size"], j, axis=0),
                                                  self.P["dt"], self.P["v_max"][j]) for j in range(self.P["N"])]
        
        if self.replay_csv:
            return
        
        if self.record: 
            p.connect(p.DIRECT)
        else: 
            p.connect(p.GUI)
        self.sim_running = True
        p.setGravity(0, 0, -9.8)

        for j in range(self.P["N"]):
            # Create spheres for current positions

            sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.P["size"][j])
            visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.P["size"][j],
                                                  rgbaColor=[self.beta[j] / max(self.beta), 0.7, 0.7, 1])
            sphere_body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_id,
                                               baseVisualShapeIndex=visual_shape_id,
                                               basePosition=(self.current_position[j][0], self.current_position[j][1], self.current_position[j][2]))
            self.spheres.append(sphere_body_id)
            if self.record:
                self.recorded_data.append(("current", j, self.current_position[j], self.P["size"][j]))

            # Create goal spheres
            goal_sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
            goal_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                                       rgbaColor=[(j + 1) / (self.P["N"] + 1), 0.7, 0.7, 1])
            goal_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=goal_sphere_id,
                                              baseVisualShapeIndex=goal_visual_shape_id,
                                              basePosition=(self.goal[j][0], self.goal[j][1], self.goal[j][2]))
            self.goal_spheres.append(goal_body_id)
            if self.record:
                self.recorded_data.append(("goal", j, self.goal[j], 0.05))



            # Create region goal spheres
            region_goal_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.P["radius"])
            region_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.P["radius"],
                                                         rgbaColor=[(j + 1) / (self.P["N"] + 1), 0.7, 0.7, 0.1])
            region_goal_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=region_goal_id,
                                                    baseVisualShapeIndex=region_visual_shape_id,
                                                    basePosition=(self.goal[j][0], self.goal[j][1], self.goal[j][2]))
            self.region_goals.append(region_goal_body_id)
            if self.record:
                self.recorded_data.append(("region", j, self.goal[j], self.P["radius"]))

    def replay(self):
        p.connect(p.GUI)
        time.sleep(5)
        for i, data in enumerate(self.recorded_data):
            sphere_type, j, position, size = data
            if sphere_type == "current":
                color = [self.beta[j] / max(self.beta), 0.7, 0.7, 1]
            elif sphere_type == "goal":
                color = [(j + 1) / (self.P["N"] + 1), 0.7, 0.7, 1]
            elif sphere_type == "region":
                color = [(j + 1) / (self.P["N"] + 1), 0.7, 0.7, 0.1]
            
            sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
            visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
            sphere_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_id,
                            baseVisualShapeIndex=visual_shape_id, basePosition=position)
            
            if sphere_type == "current" and self.replay_csv:
                self.spheres.append(sphere_body_id)

            
            if i >=3*self.P["N"]-1:
                break

            # time.sleep(0.1)  # Control playback speed
        
        for data in self.recorded_data:
            sphere_type, j, position, size = data
            if sphere_type == "current":
                p.resetBasePositionAndOrientation(self.spheres[j], position, [0, 0, 0, 1])
            time.sleep(self.P["dt"])

        p.disconnect()

    def save_recorded_data(self, filename="recorded_data.csv"):
        data_to_save = []
        for data in self.recorded_data:
            sphere_type, j, position, size = data
            
            # Handle different position formats (numpy array, tuple of np.float64, or tuple of floats)
            if isinstance(position, np.ndarray):
                x, y, z = [float(val) for val in position]
            elif isinstance(position, tuple):
                # Handle tuple of np.float64 or regular floats
                x, y, z = [float(val) for val in position]
            else:
                raise TypeError(f"Unexpected position type: {type(position)}")
            
            # Convert size and j to proper types
            size = float(size)
            j = int(j)
            
            # Create a tuple with the correct types
            data_to_save.append((sphere_type, j, x, y, z, size))
        
        # Convert to structured numpy array with explicit dtypes
        dtype = [('sphere_type', 'U10'), ('j', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('size', 'f8')]
        data_array = np.array(data_to_save, dtype=dtype)
        
        # Save as CSV
        np.savetxt(filename, data_array, fmt='%s,%d,%.6f,%.6f,%.6f,%.6f',
                header="sphere_type,j,x,y,z,size", comments="")
        print(f"Recorded data saved to {filename}")
    
    def load_recorded_data(self, filename="recorded_data.csv"):
        try:
            # Load the CSV file
            dtype = [('sphere_type', 'U10'), ('j', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('size', 'f8')]
            loaded_data = np.loadtxt(filename, dtype=dtype, delimiter=",", skiprows=1)
            self.recorded_data = []
            
            for row in loaded_data:
                sphere_type = str(row['sphere_type'])
                j = int(row['j'])
                position = np.array([row['x'], row['y'], row['z']])
                size = float(row['size'])
                
                # Store data in the same format as it was originally
                if sphere_type in ['region', 'goal']:
                    # These types use numpy arrays for position
                    self.recorded_data.append((sphere_type, j, position, size))
                else:
                    # 'current' type uses tuple of floats
                    self.recorded_data.append((sphere_type, j, tuple(float(x) for x in position), size))
                    
            print(f"Recorded data loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
        except Exception as e:
            print(f"Error loading data: {str(e)}")

    def simulate_step(self):
        self.step += 1
        # if self.P["flag_plot"] == 1:
        #     maxX,maxY = max(self.goal, key=lambda x: x[0])
        #     minX,minY = min(self.goal, key=lambda x: x[0])
        #     maxX,maxY= maxX+1,maxY+1
        #     minX,minY= maxX-1,maxY-1
        #     for j in range(self.P["N"]):
        #         if self.current_position[j][0] < minX:
        #             minX = self.current_position[j][0]
        #         if self.current_position[j][0] > maxX:
        #             maxX = self.current_position[j][0]
        #         if self.current_position[j][1] < minY:
        #             minY = self.current_position[j][1]
        #         if self.current_position[j][1] > maxY:
        #             maxY = self.current_position[j][1]
            
        #     self.ax1.set_xlim(minX - 2, maxX + 2)
        #     self.ax1.set_ylim(minY - 2, maxY + 2)

        for j in range(self.P["N"]):
            start = time.time()
            position_other_robots_and_humans = np.delete(self.current_position, j, axis=0)

            self.Lloyd[j].aggregate(position_other_robots_and_humans, self.beta[j], self.Robots.destinations[j])
            self.Lloyd_virtual[j].aggregate(position_other_robots_and_humans, self.beta[j], self.Robots.destinations[j])

            self.c1[j], self.c2[j] = self.Lloyd[j].get_centroid()
            self.c1_no_rotation[j], self.c2_no_rotation[j] = self.Lloyd_virtual[j].get_centroid()

            u = self.Lloyd[j].compute_control()

            if np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2) > self.tmp:
                self.tmp = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            d2 = 3 * max(self.P["size"])
            d4 = d2

            
            applyrules(j, self.P, self.beta, self.current_position, self.c1, self.c2, self.th, self.goal, self.Robots,
                       self.c1_no_rotation, d2, d4)
            if j == 0:
                print("cur pos: ", self. current_position[j])
                print("destination :", self.Robots.destinations[j])
                print("c1 :", self.c1[j])

            if math.sqrt((self.current_position[j][0] - self.goal[j][0]) ** 2 +
                         (self.current_position[j][1] - self.goal[j][1]) ** 2 + 
                         (self.current_position[j][2] - self.goal[j][2]) ** 2) <= self.P["radius"]:
                self.flag[j] = 1
            else:
                self.flag[j] = 0

            if sum(self.flag) == self.P["N"]:
                self.flag_convergence += 1

            if sum(self.flag) == self.P["N"] and self.flag_convergence == 1:
                print("travel time:", round(self.step * self.P["dt"], 3), "(s).  max velocity:", round(self.tmp, 3),
                      "(m/s)")

            if self.flag_convergence == self.P["waiting_time"] - 1:
                # plt.close()
                p.disconnect()
                self.sim_running = False

            # if self.P["write_file"] == 1:
            #     with open(self.file_path, 'a') as file:
            #         size = self.P["size"]
            #         dt = self.P["dt"]
            #         k = self.P["k"]
            #         data = f"{self.step},{j},{self.current_position[j][0]},{self.current_position[j][1]},{self.goal[j][0]},{self.goal[j][1]},{self.beta[j]},{size[j]},{self.c1[j][0]},{self.c1[j][1]},{k[j]},{dt}\n"
            #         file.write(data)

            self.current_position_x[j], self.current_position_y[j], self.current_position_z[j] = self.Lloyd[j].move()
            if j == 0:
                self.Lloyd[j].printVel()
            self.current_position[j] = self.current_position_x[j], self.current_position_y[j], self.current_position_z[j]

        if self.P["flag_plot"] == 1 and self.sim_running:
            for j in range(self.P["N"]):
                new_position = (self.current_position[j][0], 
                                self.current_position[j][1],  
                                self.current_position[j][2])  
                
                # Update the sphere's position using resetBasePositionAndOrientation
                p.resetBasePositionAndOrientation(self.spheres[j], new_position, [0, 0, 0, 1])  # No rotation
                if self.record:
                    self.recorded_data.append(("current", j, new_position, self.P["size"][j]))
                
                # circle = plt.Circle((self.current_position[j][0], self.current_position[j][1]), self.P["size"][j],
                #                     fill=True, color=(self.beta[j] / max(self.beta), 0.7, 0.7))
                # circlegoals = plt.Circle((self.goal[j][0], self.goal[j][1]), 0.05, fill=True,
                #                          color=((j + 1) / (self.P["N"] + 1), 0.7, 0.7))
                # regiongoals = plt.Circle((self.goal[j][0], self.goal[j][1]), self.P["radius"], fill=True, alpha=0.1,
                #                          color=((j + 1) / (self.P["N"] + 1), 0.7, 0.7))
            #     self.ax1.add_patch(circle)
            #     self.ax1.add_patch(circlegoals)
            #     self.ax1.add_patch(regiongoals)
            # plt.draw()
            # plt.pause(0.001)
            # self.ax1.clear()
