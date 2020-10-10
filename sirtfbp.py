from __future__ import print_function
import numpy as np
import astra
import pylab as pl
import cv2

import h5py
import glob
import os
# Compute SIRT-FILTER for 100 iterations
# (will save computed filters to 'filter_dir', and load from disk if
# precomputed filter is found)
import sirtfilter

####change HERE#####
PATH = '../data/'
##########

nd = 513 # Number of detector pixels
ang = np.linspace(0, np.pi, 1000, False) # Projection angles

# Create simple phantom
x = np.zeros((nd, nd))
x[nd//3:2*nd//3,nd//3:2*nd//3] = 1

# Create ASTRA projector and OpTomo object
pg = astra.create_proj_geom('parallel', 1, nd, ang)
vg = astra.create_vol_geom(362)
pid = astra.create_projector('cuda', pg, vg) # Use 'cuda' for CUDA projector
w = astra.OpTomo(pid)
filter_file = sirtfilter.getfilterfile(nd, ang, 100, filter_dir='./')
# Register SIRT-FILTER plugin with ASTRA
astra.plugin.register(sirtfilter.astra_plugin)

for step in ['train','test','validation']:
    print(step)
    try:
        os.mkdir(step)
    except OSError:
        print("Creation of the directory %s failed" % step)
    else:
        print("Successfully created the directory %s " % step)
    index = 0
    for step_path in sorted(glob.glob(PATH + 'observation_'+step+'_*.hdf5')):
        print(step_path)
        with h5py.File(step_path, "r") as f:
            # List all groups
            a_group_key = list(f.keys())[0]

            # Get the data
            data = list(f[a_group_key])

            # Get sinogram
            for sino in data:
                # Compute SIRT-FBP reconstruction
                r = w.reconstruct('SIRT-FBP', sino, extraOptions ={'filter_file': filter_file})
                #normalize data
                r = r / np.max(r)
                cv2.imwrite(step+'/' +str(index)+ '.png', cv2.flip(cv2.flip(r, 1),-1)*255)
                index +=1

