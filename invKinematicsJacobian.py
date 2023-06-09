#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math as mth
import matplotlib.pyplot as plt
import os
import time 
from vedo import *
from os import path


def drawRobotArm(Phi, goal, obstacles):
    l1 = 5
    l2 = 7
    l3 = 6
    l4 = 5
    p1 = np.array([[0],[0],[0]])
    p2 = np.array([[0],[0],[l1]])
    p3 = np.array([[0],[0],[l2]])
    p4 = np.array([[0],[0],[l3]])
    p5 = np.array([[0],[0],[l4]])
    
    
    # Local transformation matrices 
    
     
    T_01 = getTransformation(p1, Phi[0])
    T_12 = getTransformation(p2, Phi[1])
    T_23 = getTransformation(p3, Phi[2])
    T_34 = getTransformation(p4, Phi[3])

    # Local-to-global transformation matrices
    T_02 = T_01 @ T_12
    T_03 = T_01 @ T_12 @ T_23
    T_04 = T_01@T_12 @T_23 @T_34 


    # Define the coordinates of first joint and create sphere
    p1 = np.array([0,0,0])
    s1=Sphere(p1, r=0.5, c='black') 
    
     # Calculate the global coordinates of joint 2 and create sphere
    j2_local = np.array([0.0, 0.0, l1, 1])
    p2 = T_01 @ j2_local
    p2 = p2[:3]
    s2=Sphere(p2, r=5, c='purple') 

    # Calculate the global coordinates of joint 3 and create sphere
    j3_local = np.array([0.0, 0.0, l2, 1])
    p3 = T_02 @ j3_local
    p3 = p3[:3]
    s3=Sphere(p3, r=3, c='purple') 

    # Calculate the global coordinates of joint 4 and create sphere
    j4_local = np.array([0.0, 0.0, l3, 1])
    p4 = T_03 @ j4_local
    p4 = p4[:3]
    s4=Sphere(p4, r=2, c='black') 

    # Calculate the global coordinates of joint 5 and create sphere
    j5_local = np.array([0.0, 0.0, l4, 1])
    p5 = T_04 @ j5_local
    p5 = p5[:3]
    s5=Sphere(p5, r=2, c='purple') 
    
    
    g1=Sphere(goal[0], r=1, c='red') 
    g2=Sphere(goal[1], r=1, c='red')


    so1= Sphere(obstacles[0], r=1, c="blue")
    so2= Sphere(obstacles[1], r=1, c="blue")
    so3= Sphere(obstacles[2], r=1, c="blue")

    #Create cylinder arms
    c1 = Cylinder(pos=[p1,p2],r=5, axis=(0, 0, 1), cap=True, res=24, c='purple',alpha=1)

    c2 = Cylinder(pos=[p2,p3],r=3, axis=(0, 0, 1), cap=True, res=24, c='purple',alpha=1)

    c3 = Cylinder(pos=[p3,p4],r=2, axis=(0, 0, 1), cap=True, res=24, c='black',alpha=1)

    c4 = Cylinder(pos=[p4,p5],r=1, axis=(0, 0, 1), cap=True, res=24, c='light grey',alpha=1)
    

    return s1, s2, s3, s4, s5, c1, c2, c3, c4, g1, g2, so1, so2, so3


def F_R(d: float) -> float:
    
    # obstacle:1 and end_effector: 2
    R = 3

    # Assign the cost to 0 by default
    # This covers the case where the distance is greater than the radius
    cost = 0 

    # If the distance is less than or equal to the radius, then we assign a non-zero cost 
    if d > 0 and d <= R:
      cost = np.log(R/d)

    if d <= 0: 
       print("Error: Distance is 0 or negative!")
    
    return cost   # float 

def L(phi: float, joint: int) -> float:
    L1 = 5
    L2 = 7
    L3 = 6
    L4 = 5
    # Define the limits for each joint
    phi_min = [90, 0, 0, 0]  # Joint's minimum angle limit
    phi_max = [90, 90, 180,180]  # Joint's maximum angle limit
    delta = [mth.degrees(2 * mth.asin(5 / (2 * 5))),mth.degrees(2 * mth.asin(3 / (2 * 7))),mth.degrees(2 * mth.asin(2 / (2 * 6))),mth.degrees(2 * mth.asin(2 / (2 * 5)))]
      # The angular distance from each of the limits after which the limit function vanishes
    
    cost = 0
    # Compute the cost based on the provided formula for the given joint
    if phi > phi_min[joint] and phi <= phi_min[joint] + delta[joint]:
        cost = np.log(delta[joint] / (phi - phi_min[joint]))
    elif phi > phi_min[joint] + delta[joint] and phi < phi_max[joint] - delta[joint]:
        cost = 0
    elif phi >= phi_max[joint] - delta[joint] and phi < phi_max[joint]:
        cost = np.log(delta[joint] / (phi_max[joint] - phi))
    return cost

def getTransformation(t, theta):

    if theta == -1:  # no rotation
        R = np.eye(3)

    else:
        c = np.cos(theta * np.pi / 180)
        s = np.sin(theta * np.pi / 180)
        

        R = np.array(
              [[c, 0, s],
              [0, 1, 0],
              [-s, 0, c]]
          )

    T = np.block([[R, t],
                  [np.zeros((1, 3)), 1]])

    return T

# Find (x,y,z) location of ee in global coordinates
def end_effector(Phi):
    l1 = 5
    l2 = 7
    l3 = 6
    l4 = 5
    p1 = np.array([[0],[0],[0]])
    p2 = np.array([[0],[0],[l1]])
    p3 = np.array([[0],[0],[l2]])
    p4 = np.array([[0],[0],[l3]])
    p5 = np.array([[0],[0],[l4]])
    
    
    # Local transformation matrices 
    
     
    T_01 = getTransformation(p1, Phi[0])
    T_12 = getTransformation(p2, Phi[1])
    T_23 = getTransformation(p3, Phi[2])
    T_34 = getTransformation(p4, Phi[3])

    # Local-to-global transformation matrices
    T_04 = T_01@T_12@T_23@T_34
    

    e= T_04[:3,3]
    
    return e 

def gradient_descent(phi, goal,obstacles, d=0.1, learn_rate=0.5, max_iterations=10000):
    et = end_effector(phi)
    g = goal
    ob = obstacles
    draw = []
    counter = 0
    for i in range(len(goal)):
        while np.linalg.norm(et - g[i]) > d and counter < max_iterations:
            counter += 1
            c = C(phi,g,ob)
            
            delta_phi1 = np.array([ 0.05, 0, 0, 0])
            delta_phi2 = np.array([ 0, 0.05, 0, 0])
            delta_phi3 = np.array([ 0, 0, 0.05, 0])
            delta_phi4 = np.array([ 0, 0, 0, 0.05])

            c1 = phi + delta_phi1
            c1 = C(c1,g,ob)-c 
            c1= c1/0.05

            c2 = phi + delta_phi2
            c2 = C(c2,g,ob)-c 
            c2= c2/0.05

            c3 = phi + delta_phi3
            c3 = C(c3,g,ob)-c 
            c3= c3/0.05

            c4 = phi + delta_phi4
            c4 = C(c4,g,ob)-c 
            c4= c4/0.05

            delta_c_phi = np.array([ c1, c2, c3, c4])

            new_phi = phi - learn_rate * delta_c_phi
            #Draw robot arm at the new configuration
            draw.append(new_phi)

            # Updated end-effector location
            et = end_effector(new_phi)

            # Set current configuration to new configuration
            phi = new_phi
        print(et - g[i])
    return draw


# Cost function 
def C(Phi,goal, obstacles):  
    # End-effector location 
    e_Phi = end_effector(Phi)

    ob = obstacles
    
    # Goal (target) location 
    g = goal

    cost = 0

    cost = np.linalg.norm(e_Phi - g)
    cost += sum(F_R(np.linalg.norm(e_Phi - ob[j])) for j in range(len(ob)))
    i=0
    for phi in Phi:
        cost += L(phi,i)
        i= i+1
 
    return cost

# function to make the arm move
def loop_func(event):
    global s1, s2, s3, s4, s5, c1, c2, c3, c4, g1, g2, so1, so2, so3, phi,draw, i

    sa,sb,sc, sd, se, ca, cb, cc, cd, ga, gb, soa, sob, soc = drawRobotArm(phi, [[10,6,10],[15,10,6]], [[20,10,11], 
                            [31,11,30],
                            [21,21, 11]])

    s1.points(sa.points())
    s2.points(sb.points())
    s3.points(sc.points())
    s4.points(sd.points())
    s5.points(se.points())
    c1.points(ca.points())
    c2.points(cb.points())
    c3.points(cc.points())
    c4.points(cd.points())
    g1.points(ga.points())
    g2.points(gb.points())
    so1.points(soa.points())
    so2.points(sob.points())
    so3.points(soc.points())
    
    arm = s1 + s2 + s3 + s4+ s5 + c1 + c2 + c3 + c4
    g = g1+ g2
    ob = so1 + so2 + so3
    

    #Update scene
    plt.render()
    time.sleep(0.1)
    video.add_frame()
    
    #Update phi
    phi = draw[i]
    i= i+1


#Video file
video = Video("tmp.mp4", 
                backend='ffmpeg', 
                fps = 24
                ) 
phi = np.array([ 0, np.pi/20, np.pi/20, np.pi/20])
goal = np.array([[10,6,10],[15,10,6]])
# Location of obstacles 
obstacles = np.array([[20,10,11], 
                            [31,11,61],
                            [21,21, 11]])
s1,s2,s3, s4,s5, c1, c2, c3, c4, g1, g2, so1, so2, so3 = drawRobotArm(phi, goal, obstacles)
arm = s1 + s2 + s3 + s4+ s5 + c1 + c2 + c3 + c4
g = g1 + g2
ob = so1 + so2 + so3
plt = Plotter(size=(1050, 600))
plt += [ arm,g,ob, __doc__]
plt.background("black", "w").add_global_axes(axtype=1).look_at(plane='xy')
    
i=0
draw = gradient_descent(phi,goal,obstacles)
plt.add_callback("timer", loop_func)
plt.timer_callback("create", dt=50)
plt.show().close()
# merge all the recorded frames
video.close()                        

# Convert the video file spider.mp4 to play on a wider range of video players 
if path.exists("./animation.mp4"):
    os.system("rm animation.mp4")
    
os.system("ffmpeg -i tmp.mp4 -pix_fmt yuv420p animation.mp4")
os.system("rm tmp.mp4")






