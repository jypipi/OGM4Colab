import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import cv2
from IPython.display import clear_output
from math import *
from time import time, sleep
from Path_Sim import Simulation
from util import World2Grid, Grid2World, Body2World
from utilities.timings import Timings


class Map():
    def __init__(self, map_file: str, realMapSize, res=0.1):
        self.map = np.loadtxt(map_file, dtype=float)
        self.landmarks = [(2.5/2., 2.5/2.),   (-2.5/2., 2.5/2.),
                          (-2.5/2., -2.5/2.), (2.5/2., -2.5/2.)]
        scale_idx = 3
        self.map = cv2.resize(self.map, dsize=(50*scale_idx, 50*scale_idx), interpolation=cv2.INTER_NEAREST)
        res /= scale_idx
        self.realMapSize = realMapSize
        self.gridMapSize = int(self.realMapSize/res)
        self.res = res

    def World2Grid(self, point):
        return World2Grid(point, self.realMapSize, self.gridMapSize, self.res)

class ParticleFilter():
    class Particle():
        def __init__(self, x, y, yaw, weight=0):
            self.pos = np.array([x, y])
            self.yaw = yaw
            self.weight = weight

    class Robot():
        def __init__(self, robot_pose):
            x, y, yaw = robot_pose
            self.pos = np.array([x, y])
            self.yaw = yaw

    def __init__(self, input_map: Map, num_particles, robot_pose):
        ''' 
        The constructor initializes certain parameters and the robot on the map, 
        while updating the robot and particle states and weights after a key event.

        Initializes:
             - robot position 
             - motion noise 
             - measurement noise 
             - landmark positions 
             - number of particles
        '''

        self.map = input_map

        # Initialize robot
        self.robot = self.Robot(robot_pose)

        # motion and measurement noise
        self.motion_noise = 0.02
        self.measurement_noise = 0.5

        # position of landmarks
        self.landmarks = self.map.landmarks

        # Initialize particles
        self.num_particles = num_particles
        self.particles = []
        self.generate_particles()

    def generate_particles(self):
        '''
        This method creates the particles object which is a num_particles x 3 numpy array.
        '''
        weight = 1 / self.num_particles
        # Xs = np.random.normal(self.robot.pos[0], 1, self.num_particles)
        # Ys = np.random.normal(self.robot.pos[1], 1, self.num_particles)
        # yaws = np.random.uniform(0, 2*pi, self.num_particles)
        # for i in range(self.num_particles):
        #     x, y = Xs[i], Ys[i]
        #     while np.abs(x) > self.map.realMapSize/2:
        #         x = np.random.normal(self.robot.pos[0], 1, 1)
        #     while np.abs(y) > self.map.realMapSize/2:
        #         y = np.random.normal(self.robot.pos[1], 1, 1)
        #     self.particles.append(self.Particle(x, y, yaws[i], weight))
        # self.particles = np.array(self.particles)
        
        Xs = np.random.uniform(-2.45, 2.45, self.num_particles)
        Ys = np.random.uniform(-2.45, 2.45, self.num_particles)
        yaws = np.random.uniform(0, 2*pi, self.num_particles)
        for i in range(self.num_particles):
            self.particles.append(self.Particle(Xs[i], Ys[i], yaws[i], weight))
        self.particles = np.array(self.particles)
        
    def prediction_step(self, new_robot_pose):
        '''
        This method performs the prediction step and changes the state of the particles. 
        '''
        ### Goal: use orientation (yaw angle)
        new_robot_pos = np.array(new_robot_pose[0:2])
        dx, dy = new_robot_pos - self.robot.pos
        self.robot.pos = new_robot_pos
        for particle in self.particles:
            particle.pos[0] += dx + np.random.normal(0, self.motion_noise, 1)
            particle.pos[1] += dy + np.random.normal(0, self.motion_noise, 1)
            
            # # Unicycle dynamic model prediction: [world_x, world_y, yaw]
            # x_old = np.array([particle.pose[0], particle.pose[1], particle.pose[2]])
            # A_mat = np.identity(3, dtype=float)
            # B_mat = np.array([[np.cos(particle.pose[2]), 0.],
            #                   [np.sin(particle.pose[2]), 0.],
            #                   [0., 1.]])
            # x_new = np.matmul(A_mat, x_old) + np.matmul(B_mat, velocity.T)*delta_time
            # particle.pose = x_new.flatten()

    def update_step(self):
        ''' 
        This method performs the update step and changes the weights of the particles.
        '''
        # weights is the difference of particle distance and robot distance with respect to the 2 landmarks
        normalized_weights = self.calculate_weight() # size: num_particles X 4

        # Assign final weights to the particles
        for i in range(self.num_particles):
            self.particles[i].weight = normalized_weights[i]

    def calculate_weight(self):
        '''
        This method finds varying distances, and returns the distances to be used in the update_step
        Measures:
            - Distance between robot and each landmark
            - Distance between each particle and each landmark
            - Distance between each particle and each landmark with measurement noise (Gaussian distribution)
        
        Returns:
            - Normalized Weights
        '''
        # distance between robot and all landmarks
        num_landmarks = len(self.landmarks)
        dists_r2l = np.zeros((num_landmarks))
        for i in range(num_landmarks):
            dists_r2l[i] = np.linalg.norm(self.robot.pos - np.array(self.landmarks[i])) + np.random.normal(0, self.measurement_noise, 1)
        Z_r2l = np.tile(dists_r2l, (self.num_particles, 1))

        # distance between each particle and all landmarks
        Y_p2l = np.zeros((self.num_particles, num_landmarks), dtype=float)
        for i in range(self.num_particles):
            for j in range(num_landmarks):
                Y_p2l[i][j] = np.linalg.norm(self.particles[i].pos-np.array(self.landmarks[j])) + np.random.normal(0, self.measurement_noise, 1)

        # calculate the weights by finding the difference of particles' and robot's distances to all landmarks
        weights = 1 / np.abs(Y_p2l - Z_r2l)
        normalized_weights = weights / np.sum(weights, axis=0)
        normalized_weights = np.sum(normalized_weights, axis=1)
        normalized_weights /= np.sum(normalized_weights)

        return normalized_weights
    
    def resampling(self):
        '''
        This method performs stratified resampling of particles with the latest weights.

        Reference: Probabilistic Robotics, Ch.4, page 86
        '''
        new_particles = []
        new_weight = 1 / self.num_particles
        r = np.random.uniform(0, new_weight, 1)
        c = self.particles[0].weight
        i = 0
        count = 0

        for m in range(self.num_particles):
            u = r + (m-1)*new_weight
            while u > c:
                i += 1
                c += self.particles[i].weight

            x, y = self.particles[i].pos
            if abs(x) > self.map.realMapSize/2 or abs(y) > self.map.realMapSize/2:
                ### Goal: resample based on the particle with the largest weight (replace self.robot.pos)
                while np.abs(x) > self.map.realMapSize/2:
                    x = np.random.normal(self.robot.pos[0], 1, 1)
                while np.abs(y) > self.map.realMapSize/2:
                    y = np.random.normal(self.robot.pos[1], 1, 1)
                count += 1
                
            new_particles.append(self.Particle(x[0], y[0], self.particles[i].yaw, new_weight))

        self.particles = np.array(new_particles)

        if count != 0:
            print("outliers resampled:", count)
            
    def visualize(self, image, end=False):
        # if image is None:
        #     return

        plt.clf()
        clear_output(wait=True)

        # # self.figure.add_subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.title('Physical World')

        # cmap = plt.get_cmap('rainbow', self.particles)
        # cNorm  = colors.Normalize(vmin=np.min(At[t,:,:]), vmax=np.max(At[t,:,:]))
        # scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
        
        # self.figure.add_subplot(1, 2, 2)
        plt.imshow(self.map.map, cmap='gray', vmin=0, vmax=1, origin='lower')
        robot_pos = self.map.World2Grid((self.robot.pos[0], self.robot.pos[1]))
        plt.scatter(robot_pos[0], robot_pos[1], s=180, marker='o', c='b', edgecolors='r')
        for particle in self.particles:
            ### Goal: adjust the range of color bar with max and min weights
            # colorVal = scalarMap.to_rgba(At[t,i,j])
            particle_pos = self.map.World2Grid((particle.pos[0], particle.pos[1]))
            plt.scatter(particle_pos[0], particle_pos[1], marker='o', s=3, c=particle.weight, cmap='rainbow')
        plt.title('Real-time Map with Particles')
        plt.colorbar(label='Particle Weight', orientation='vertical', shrink=0.9)

        #### For running on Colab
        plt.show(block=True)
        #### For running on local computer
        # if end:
        #     plt.show(block=True)
        # else:
        #     plt.show(block=False)
        #     plt.pause(0.0001)

def main():
    t0 = time()
    num_of_particles = 75

    sim_FPS = 1
    path_sim_time = Timings(sim_FPS)

    sim = Simulation()
    realMapSize = sim.sim.get_env_info()["map_size"]
    input_map = Map('/content/OGM4Colab/Particle_Filter_V1/probGridMap_perfect.txt', realMapSize, res=0.1)
    image, dataset, status, vel, steering = sim.collectData(True, begin=True)

    pf = ParticleFilter(input_map, num_of_particles, dataset[0])
    pf.visualize(image) # display the initial particles

    time_to_plot = False

    while True:
        image, dataset, status, vel, steering = sim.collectData(True)

        if status == -1:
            print('Total run time:', floor((time()-t0)/60), 'min',
                round((time()-t0)%60, 1), 'sec.')
            pf.visualize(image, end=True)
            break

        if path_sim_time.update_time():
            time_to_plot = True

        # Perform update once a movement is completed
        pf.prediction_step(dataset[0])
        if time_to_plot:
            pf.visualize(image) # particles move based on control u
            time_to_plot = False
        pf.update_step()
        pf.resampling()


if __name__ == "__main__":
    main()