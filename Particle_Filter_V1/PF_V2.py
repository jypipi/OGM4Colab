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
        self.turn_noise = 0.02
        self.measurement_noise = 0.5

        # position of landmarks
        self.landmarks = self.map.landmarks

        # Initialize particles
        self.num_particles = num_particles
        self.particles = []
        self.weights = np.ones(self.num_particles)
        self.generate_particles()

    def generate_particles(self):
        '''
        This method creates the particles object which is a num_particles x 3 numpy array.
        '''
        weight = 1 / self.num_particles
        # Xs = np.random.normal(self.robot.pos[0], 1, self.num_particles)
        # Ys = np.random.normal(self.robot.pos[1], 1, self.num_particles)
        Xs = np.random.uniform(-2.45, 2.45, self.num_particles)
        Ys = np.random.uniform(-2.45, 2.45, self.num_particles)
        yaws = np.random.uniform(0, 2*pi, self.num_particles)
        for i in range(self.num_particles):
            self.particles.append(self.Particle(Xs[i], Ys[i], yaws[i], weight))
        self.particles = np.array(self.particles)
        self.weights *= weight
        
    def prediction_step(self, new_robot_pose):
        '''
        This method performs the prediction step and changes the state of the particles. 
        '''
        new_robot_pos = np.array(new_robot_pose[0:2])
        dx, dy = new_robot_pos - self.robot.pos
        d_yaw = new_robot_pose[2] - self.robot.yaw
        self.robot.pos, self.robot.yaw = new_robot_pos, new_robot_pose[2]
        for particle in self.particles:
            motion_noise = np.random.normal(0, self.motion_noise, 2)
            turn_noise = np.random.normal(0, self.turn_noise, 1)
            particle.pos[0] += dx + motion_noise[0]
            particle.pos[1] += dy + motion_noise[1]
            particle.yaw = (particle.yaw + pi + d_yaw + turn_noise[0]) % (2*pi) - pi

    def update_step(self):
        '''
        This method performs the update step and updates the weights of all particles.

        Measures:
            - Distance between robot and each landmark with measurement noise (Gaussian distribution)
            - Distance between each particle and each landmark
            - Joint probability of each particle based on the above distances
        '''
        # Calculate robot's distance measurements to all landmarks
        num_landmarks = len(self.landmarks)
        dists_r2l = np.zeros((num_landmarks))
        for i in range(num_landmarks):
            dists_r2l[i] = np.linalg.norm(self.robot.pos - np.array(self.landmarks[i])) \
                         + np.random.normal(0, self.measurement_noise, 1) # introduce the measurement noise

        # Calculate weights based on the distance between each particle and all landmarks
        p_x = lambda x, dist_meas: np.exp(-((x-dist_meas)**2)/(2*(self.measurement_noise**2))) / np.sqrt(2*pi*(self.measurement_noise**2))
        weights = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            probability = 1
            for j in range(num_landmarks):
                dist_meas = np.linalg.norm(self.particles[i].pos-np.array(self.landmarks[j]))
                # Calculate the likelihood of this particle based on the robot's measurements under 1D Gaussian distribution
                probability *= p_x(dists_r2l[j], dist_meas)
            weights[i] = probability
        
        normalized_weights = weights / np.sum(weights)
        self.weights = normalized_weights
    
    def resampling(self):
        '''
        This method performs stratified resampling of particles with the latest weights.

        Reference: Probabilistic Robotics, Ch.4, page 86
        '''
        new_particles = []
        new_weight = 1 / self.num_particles
        r = np.random.uniform(0, new_weight, 1)
        c = self.weights[0]
        i = 0
        count = 0

        for m in range(self.num_particles):
            u = r + (m-1)*new_weight
            while u > c:
                i += 1
                c += self.weights[i]

            x, y = self.particles[i].pos
            if abs(x) > self.map.realMapSize/2 or abs(y) > self.map.realMapSize/2:
                ### Goal: resample based on the particle with the largest weight (replace self.robot.pos)
                while np.abs(x) > self.map.realMapSize/2:
                    x = np.random.normal(self.robot.pos[0], 1, 1)
                    x = x[0]
                while np.abs(y) > self.map.realMapSize/2:
                    y = np.random.normal(self.robot.pos[1], 1, 1)
                    y = y[0]
                count += 1
                
            new_particles.append(self.Particle(x, y, self.particles[i].yaw, new_weight))

        self.particles = np.array(new_particles)
        self.weights = new_weight * np.ones(self.num_particles)

        # if count != 0:
        #     print("outliers resampled:", count)
            
    def visualize(self, image, end=False):
        # if image is None:
        #     return

        plt.clf()
        clear_output(wait=True)

        # # self.figure.add_subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.title('Physical World')

        # self.figure.add_subplot(1, 2, 2)
        cmap = plt.get_cmap('rainbow', self.num_particles)
        cNorm  = colors.Normalize(vmin=0, vmax=0.1)
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

        plt.imshow(self.map.map, cmap='gray', vmin=0, vmax=1, origin='lower')
        robot_pos = self.map.World2Grid((self.robot.pos[0], self.robot.pos[1]))
        plt.scatter(robot_pos[0], robot_pos[1], s=180, marker='o', color='b', edgecolors='r')
        plt.arrow(robot_pos[0], robot_pos[1],
                  18*np.cos(self.robot.yaw), 18*np.sin(self.robot.yaw),
                  head_width=3, head_length=7, length_includes_head=True,
                  color='b', linewidth=2)
        
        max_weight, max_point_idx, checked = np.max(self.weights), None, False
        for i in range(self.num_particles+1):
            if i == self.num_particles:
                particle = self.particles[max_point_idx]
                curr_weight = max_weight
            else:
                particle = self.particles[i]
                curr_weight = self.weights[i]

            if curr_weight == max_weight and not checked:
                max_point_idx = i
                checked = True
                continue

            colorVal = scalarMap.to_rgba(curr_weight)
            particle_pos = self.map.World2Grid((particle.pos[0], particle.pos[1]))
            plt.scatter(particle_pos[0], particle_pos[1], marker='o', s=3, color=colorVal, cmap=scalarMap)
            plt.arrow(particle_pos[0], particle_pos[1],
                      7*np.cos(particle.yaw), 7*np.sin(particle.yaw),
                      head_width=1.5, head_length=3, length_includes_head=True,
                      color=colorVal, linewidth=0.75)
            
        plt.title('Real-time Map')
        plt.colorbar(scalarMap, label='Particle Weight', orientation='vertical', shrink=0.9)
        x, y = self.map.map.shape
        plt.xlim([-0.5, x-0.5])
        plt.ylim([0, y])

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

    sim_FPS = 0.5
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
        pf.update_step()
        if time_to_plot:
            pf.visualize(image) # particles move based on control u
            time_to_plot = False
        pf.resampling()


if __name__ == "__main__":
    main()