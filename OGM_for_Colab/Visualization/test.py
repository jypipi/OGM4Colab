from vedo import *
import numpy as np

x = np.arange(0, 1000, 0.1)
y = np.sin(x)

line = Line(x,y)
pointcloud = load('PointCloud/testingpointcloud.ply')

rgb = load('PointCloud/rgb.png')
show(pointcloud, rgb, line)