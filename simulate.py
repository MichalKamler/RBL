import math
import numpy as np
import matplotlib.pyplot as plt
from RobotInit import RobotsInit
from RobotInit1 import RobotsInit1
from Lloydbasedalgorithm import LloydBasedAlgorithm
from plot_utils import plot_circle, plot_line  
import time
import matplotlib.patches as patches

def check_parameters(P):
    for j in range(P["N"]):
        if (((P["radius"]-P["R_gaussianD"][j])*math.exp(2*P["radius"]/P["R_gaussianD"][j])+P["radius"]+P["R_gaussianD"][j])/(math.exp(2*P["radius"]/P["R_gaussianD"][j])-1)) < 2*P["size"][j]:
            print("Error type 1. parameters not selected correctly. Safety and convergence may be compromized ")
        if P["d3"] >  (((P["radius"]-P["R_gaussianD"][j])*math.exp(2*P["radius"]/P["R_gaussianD"][j])+P["radius"]+P["R_gaussianD"][j])/(math.exp(2*P["radius"]/P["R_gaussianD"][j])-1)) - 2*P["size"][j] or P["d1"] > (((P["radius"]-P["R_gaussianD"][j])*math.exp(2*P["radius"]/P["R_gaussianD"][j])+P["radius"]+P["R_gaussianD"][j])/(math.exp(2*P["radius"]/P["R_gaussianD"][j])-1)) - 2*P["size"][j]:
            print("Error type 2. pcarameters not selected correctly. It may not converge.")
        if P["k"][j]*P["dt"]> 1:
            print("k_p is too big or dt is to big, safety issues")
        if P["k"][j] < 1:
            print("Warning, of kp may be too small")
#def check_collision(x1, y1, radius1, x2, y2, radius2,in1,in2):
#    distances = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#    collision_indices = distances <= (radius1 + radius2)
#    if collision_indices > 0:
    #    return in1,in2
 #   else:
#        return -1,-1

def set_robot_positions_and_goals(P):
    positionandgoals = RobotsInit1(P)
    plt.show()
    return positionandgoals.get_positions_and_goals()


def simulate( h, P ):
    #Initialize variables
    collisions = 0
    tmp  = 0
    tmp_count = [0] * P["N"]
    c1 = [0,0] * P["N"]  #actual centroids 
    c2 = [0,0] * P["N"]  #virtual centroids (without neighbours)
    c1_no_rotation = [0,0] * P["N"]  # \bar p = e, centroids  
    c2_no_rotation = [0,0] * P["N"]  # \bar p = e, virtual centroids (without neighbours)
    c1_no_humans = [0,0] * P["N"] 
    c2_no_humans = [0,0] * P["N"] 
    step = 0
    flag = [0]*P["N"]
    file_path = 'test' + str(h) + '.txt'
    current_position_x = np.zeros(P["N"])
    current_position_y = np.zeros(P["N"])
    th                 = [0] * P["N"]  
    Robots = RobotsInit(P)
    if P["manual"] == 1:
        #Robots = RobotsInit1(P)
        Robots.positions, Robots.destinations = set_robot_positions_and_goals(P)
        current_position = Robots.positions
        goal = Robots.destinations.copy()
    else:
        current_position = Robots.positions
        goal             = Robots.destinations.copy()  #goal positions   

    maxX,maxY = max(goal, key=lambda x: x[0])
    minX,minY = min(goal, key=lambda x: x[0])
    maxX,maxY= maxX+2,maxY+2
    minX,minY= maxX-2,maxY-2


    goal_copy        = goal.copy()
    R_gaussian       = P["R_gaussianD"].copy()     #spreding factor
    Lloyd            = [0] * P["N"]
    Lloyd_virtual    = [0] * P["N"]
    #Lloyd_virtual1   = [0] * P["N"]




    for j in range(P["N"]):  
        Lloyd[j]  = LloydBasedAlgorithm(Robots.positions[j], P["radius"], P["dx"], P["k"][j], P["size"][j], np.delete(P["size"], j, axis=0), P["dt"],P["v_max"][j])
        Lloyd_virtual[j] = LloydBasedAlgorithm(Robots.positions[j], P["radius"], P["dx"], P["k"][j], P["size"][j], np.delete(P["size"], j, axis=0), P["dt"],P["v_max"][j])
      
        #Lloyd_virtual1[j] = LloydBasedAlgorithm(Robots.positions[j], P["radius"], P["dx"], P["k"][j], P["size"][j], np.delete(P["size"], j, axis=0), P["dt"],P["v_max"][j])
    plt.ion()
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid()


    while sum(flag) < P["N"] and step <P["num_steps"]: # until all robots reach their goal or the number of steps is reached            
        if P["flag_plot"] == 1:


            for j in range(P["N"]): 
                if Robots.positions[j][0] < minX:
                    minX = Robots.positions[j][0]
                if Robots.positions[j][0] > maxX:
                    maxX = Robots.positions[j][0]
                if Robots.positions[j][1] < minY:
                    minY = Robots.positions[j][1]
                if Robots.positions[j][1] > maxY:
                    maxY = Robots.positions[j][1]
            ax1.set_xlim(minX-1,maxX+1)
            ax1.set_ylim(minY-1,maxY+1)
            ax1.axis('equal')

        step= step+1
        for j in range(P["N"]):   #for each robot
            start = time.time()
            #if flag[j] == 1:
            #    tmp_count[j] = tmp_count[j] + 1
            #    goal[j] =   random.choice(goal_copy)
            #    flag[j] == 0
            #    if j > P["N_h"]-1:
            #        if tmp_count[j] > 5:
            #            goal[j] = goal_copy[j]
            #            flag[j] == 1
            #    else:
            #        if tmp_count[j] > 5:
            #            goal[j] = goal_copy[j]
            #            flag[j] == 1



            #position_other_robots=np.delete(Robots.positions, [j] + list(range(P["N_h"]-1)), axis=0)
            position_other_robots_and_humans=np.delete(Robots.positions, j , axis=0)

            if j < P["N_h"]:
                Lloyd[j].aggregate([] , R_gaussian[j], Robots.destinations[j])   
                Lloyd_virtual[j].aggregate([], R_gaussian[j], goal[j])
                #Lloyd_virtual1[j].aggregate([], R_gaussian[j], goal[j])

            else:
                Lloyd[j].aggregate(position_other_robots_and_humans , R_gaussian[j], Robots.destinations[j])   
                Lloyd_virtual[j].aggregate(position_other_robots_and_humans, R_gaussian[j], goal[j])
                #Lloyd_virtual1[j].aggregate(position_other_robots, R_gaussian[j], goal[j])
            
                    
            c1[j],c2[j] = Lloyd[j].get_centroid()
            c1_no_rotation[j],c2_no_rotation[j] =  Lloyd_virtual[j].get_centroid()
            #c1_no_humans[j],c2_no_humans[j] = Lloyd_virtual1[j].get_centroid()

            #u = Lloyd[j].compute_control()
            #if np.sqrt(u[0]**2+u[1]**2) >tmp:
            #    tmp = np.sqrt(u[0]**2+u[1]**2)
            
            #Apply the Heuristic inputs to modify Rgaussian and Robots.destinations on the basis of c1 and c2          
            #equation (8)
            d2 = 2*max(P["size"])
            d4 = d2
            #if abs(np.linalg.norm(np.array(c1[j]) - np.array(c2[j]))) > d2 and np.linalg.norm(np.array(current_position[j]) - np.array(c1[j])) < P["d1"]:
            if abs(np.linalg.norm(np.array(c1[j]) - np.array(c2[j]))) > d2 and np.linalg.norm(np.array(current_position[j]) - np.array(c1[j])) < P["d1"]:
                R_gaussian[j] = R_gaussian[j] - 1*P["dt"]
                R_gaussian[j] = max(R_gaussian[j], P["R_gauss_min"])
            else:
                R_gaussian[j]= R_gaussian[j] - 1*P["dt"]*(R_gaussian[j]-P["R_gaussianD"][j])
                 #equation (9)
            if abs(np.linalg.norm(np.array(c1[j]) - np.array(c2[j]))) > d4 and np.linalg.norm(np.array(current_position[j]) - np.array(c1[j])) < P["d3"]:# and distancemin[j] < size[j]+max(size)+1:
                th[j] = min(th[j] + 1*P["dt"], math.pi/2.1)
            else:
                th[j] = max(0, th[j] - 1*P["dt"] )
            if th[j] == math.pi/2.1 and abs(np.linalg.norm(np.array(current_position[j]) - np.array(c1_no_rotation[j]))  > np.linalg.norm(np.array(current_position[j]) - np.array(c1[j]))):
                th[j] = 0 
            angle     = math.atan2(goal[j][1] - current_position[j][1], goal[j][0] - current_position[j][0])
            new_angle = angle - th[j]
            distance  = math.sqrt((goal[j][0] - current_position[j][0])**2 + (goal[j][1] - current_position[j][1])**2)
            Robots.destinations[j][0] = current_position[j][0] + distance * math.cos(new_angle)
            Robots.destinations[j][1] = current_position[j][1] + distance * math.sin(new_angle)
            end = time.time()
            #print(end-start)
            #condition used for stop the simulation

            if  0:#math.sqrt((current_position[j][0]-goal[j][0])**2 + (current_position[j][1]-goal[j][1])**2) <= d2+P["dx"]:
                flag[j] = 1
            else:
                flag[j] = 0

            if sum(flag) == P["N"] and j ==P["N"]-1:
                 print("travel time:", round(step*P["dt"],3), "(s).  max velocity:", round(tmp,3), "(m/s). max computational time")

            
            if P["write_file"] == 1:
                with open(file_path, 'a') as file:
                    size = P["size"]
                    dt = P["dt"]
                    k = P["k"]
                    data = f"{step},{j},{current_position[j][0]},{current_position[j][1]},{goal[j][0]},{goal[j][1]},{R_gaussian[j]},{size[j]},{c1[j][0]},{c1[j][1]},{k[j]},{dt}\n"
                    file.write(data)
            #ind = []
            #for ii in range(P["N"]):
            #    for i in range((P["N"])):
            #        if i!=ii:
            #            ind1,ind2 = check_collision(current_position_x[ii],current_position_y[ii],P["size"][ii]+0.2,current_position_x[i],current_position_y[i],P["size"][i],i,ii)

           #             if ind1 < P["N_h"] and ind1 != -1 and ind2 > P["N_h"]-1 :
           #                 ind.append(ind1)
           #             if ind2 < P["N_h"] and ind2 != -1 and ind1 > P["N_h"]-1 :
           #                 ind.append(ind2)

                        #if ((ind1>P["N_h"]-1 and  ind2<P["N_h"]) or (ind2>P["N_h"]-1 and ind1<P["N_h"]))  and step>1:
                        #    print("collision",ind1,ind2)
                        #    if ind1<P["N_h"]:   
                        #        ind =ind1
                        ##    else:
                         #       ind = ind2
                         #   if ind1 == -1: 
                         #       ind = ind1
            current_position_x[j], current_position_y[j] = Lloyd[j].move()
            current_position[j] = current_position_x[j], current_position_y[j]    
                    
            #if j not in ind :            
           
            #else:
            #    current_position[j] = current_position_x[j], current_position_y[j]   
        if P["flag_plot"] == 1:
            circles = []

                #ax1.plot_circle(current_position[j],P["size"][j],'blue')
                #if j < P["N_h"]:
                   #plot_circle(current_position[j],P["size"][j],'red')
                #ax1.plot_line((current_position[j][0],current_position[j][1]),(goal[j][0],goal[j][1]))
            for j in range(P["N"]):
                circle = patches.Circle((current_position[j][0], current_position[j][1]), P["size"][j], fill=False, color='blue')
                ax1.add_patch(circle)
            fig1.canvas.flush_events()

            plt.pause(0.001)
            for circle in ax1.patches:
                circle.remove()
            fig1.canvas.flush_events()




