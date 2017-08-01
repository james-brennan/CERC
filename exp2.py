"""
Exp2.py

    Full signal, with
    BRDF plus with additive noise 
    of low, medium and high values
""" 

import os
import numpy as np 
import h5py 
from utils import *


if __name__ == "__main__":


    brdfDict = {
    'medium':
         {
            'sza_off' : 0.5*np.pi, # offset -- currently N hemi summer
            'szaMAX' : 60 ,    #  degrees
            'szaMIN' : 10  ,   #  degrees
            'vzaMAX' : 45  ,   #  degrees
            'vzaMIN' : 0   ,   #  degrees
            'raaMAX' : 360 ,   #  degrees
            'raaMIN' : 0    ,  #  degrees
            'raa_cycle' : 32 , #  in days
            'vza_cycle' : 6 ,  #  in days
         }
    }


    if not os.path.isdir("exp2"):
        os.makedirs("exp2")
    os.chdir("exp2")

    uncs = np.array([0.004, 0.015, 0.003, 0.004, 0.013, 0.010, 0.006])
    uncs = np.array([0.003 ,  0.015 ,  0.004 ,  0.004  , 0.015 ,  0.008 ,  0.003])**2


    N = 35

    """
    Load stuff
    """ 


    for site in ['h24v03', "h11v10", "h20v09"]:
        """
        Load data
        """ 
        datadir = "/group_workspaces/cems2/nceo_generic/users/jbrennan01/RRLandsat/models/"
        filein = "ref_%s_2008.hdf5" % site
        fi = h5py.File("%s/%s" % (datadir, filein))
        iso = fi['isotropic'][90:-90]
        geo = fi['geometric'][90:-90]
        vol = fi['volumetric'][90:-90]
        sh = iso.shape
        shape = (sh[0], sh[-1], sh[-2])

        for sn in [0.5, 1.0, 1.5]:

            errors = uncs * sn 
            cov = np.diag(errors)
            for i in xrange(N):
                
                iso_copy = np.copy(iso)
                # add BRDF
                kerns, VZA, SZA, RAA = angular1(brdfDict['medium'])
                # simulate it
                angg = geo * kerns.Li[:-1][:, None, None, None] + vol * kerns.Ross[:-1][:, None, None, None]
                # add BRDF
                iso_copy += 1.5*angg 
                # make noise 
                noise = np.random.multivariate_normal(np.zeros(7), cov, size=shape)
                noise = np.swapaxes(noise, 3, 1)
                # add noise
                iso_copy += noise
                # save
                np.save("Exp2_%s_sn_%2f_%i" % (site, sn, i), (iso_copy*255).astype(np.uint8))

