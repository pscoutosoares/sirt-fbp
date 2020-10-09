from __future__ import print_function
import numpy as np
import astra
import pylab as pl
import cv2
pl.gray()

nd = 513 # Number of detector pixels
ang = np.linspace(0, np.pi, 1000, False) # Projection angles


# Create simple phantom
x = np.zeros((nd, nd))
x[nd//3:2*nd//3,nd//3:2*nd//3] = 1



# Create ASTRA projector and OpTomo object
pg = astra.create_proj_geom('parallel', 1.0, nd, ang)
vg = astra.create_vol_geom(362)
pid = astra.create_projector('cuda', pg, vg) # Use 'cuda' for CUDA projector
w = astra.OpTomo(pid)

# Simulate sinogram
#sino = (w*x).reshape(w.sshape)
#sino += np.random.normal(scale=sino.max()/20, size=sino.shape) # add noise
sino = cv2.imread('sinTest01.png',0)

# Compute SIRT-FILTER for 100 iterations
# (will save computed filters to 'filter_dir', and load from disk if
# precomputed filter is found)
import sirtfilter
filter_file = sirtfilter.getfilterfile(nd, ang, 100, filter_dir='./')

# Register SIRT-FILTER plugin with ASTRA
astra.plugin.register(sirtfilter.astra_plugin)

# Print help message to screen
print(astra.plugin.get_help('SIRT-FBP'))

# Compute SIRT-FBP reconstruction
r = w.reconstruct('SIRT-FBP', sino, extraOptions ={'filter_file': filter_file})

# Compare with ASTRA FBP reconstruction
rfbp = w.reconstruct('FBP_CUDA', sino) # use 'FBP_CUDA' for CUDA FBP algorithm
pl.subplot(121)
pl.axis('off')
pl.imshow(rfbp, vmin=0, vmax=1)
pl.title('ASTRA FBP')
pl.subplot(122)
pl.imshow(r, vmin=0, vmax=1)
pl.title('SIRT-FBP')
pl.show()

# Multiple filters can be requested by supplying a list of iteration numbers
# This returns a dictionary with a filter file for each chosen iteration number
filter_dict = sirtfilter.getfilterfile(nd, ang, [50, 100,200])
r50 = w.reconstruct('SIRT-FBP', sino, extraOptions ={'filter_file': filter_dict[50]})
r100 = w.reconstruct('SIRT-FBP', sino, extraOptions ={'filter_file': filter_dict[100]})
r200 = w.reconstruct('SIRT-FBP', sino, extraOptions ={'filter_file': filter_dict[200]})
pl.subplot(131)
pl.axis('off')
pl.imshow(r50, vmin=0, vmax=1)
pl.title('50 iterations')
pl.subplot(132)
pl.axis('off')
pl.imshow(r100, vmin=0, vmax=1)
pl.title('100 iterations')
pl.subplot(133)
pl.axis('off')
pl.imshow(r200, vmin=0, vmax=1)
pl.title('200 iterations')
pl.show()