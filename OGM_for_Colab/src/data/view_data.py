import numpy as np

poses = np.load('poses.npy')
np.savetxt('poses.csv', poses, delimiter=',')

meas = np.load('meas.npy')
ranges = meas[0]
bearings = meas[1]
np.savetxt('ranges.csv', ranges, delimiter=',')
np.savetxt('bearings.csv', bearings, delimiter=',')
