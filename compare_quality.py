from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import cv2
import h5py
import glob
import pylab as pl
import os
PATH = '../data/
ssim = {'train': [], 'test': [], 'validation': []}
psnr = {'train': [], 'test': [], 'validation': []}
for step in ['train']:
    print(step)
    index = 0
    for step_path in sorted(glob.glob(PATH + 'ground_truth_'+step+'_*.hdf5')):
        print(step_path)
        with h5py.File(step_path, "r") as f:
            # List all groups
            a_group_key = list(f.keys())[0]

            # Get the data
            data = list(f[a_group_key])

            # Get sinogram
            for gt_sample in data:
                if (np.count_nonzero(gt_sample)==0):
                    continue
                #rotate,flip and normalize image between 0-255
                gt_sample = cv2.rotate(cv2.flip((gt_sample/np.max(gt_sample)*255), 1),cv2.ROTATE_90_COUNTERCLOCKWISE)
                ld_sample = cv2.imread('./'+step+'/'+str(index)+'.png',0)

                ssim[step].append(structural_similarity(gt_sample, ld_sample, data_range=255))
                psnr[step].append(peak_signal_noise_ratio(gt_sample, ld_sample, data_range=255))
                index +=1
    print('Media ssim para '+step+':' +str(np.mean(ssim[step])))
    print('Media psnr para '+step+':' +str(np.mean(psnr[step])))
