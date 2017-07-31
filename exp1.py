"""
Exp1.py

Generate data for experiment 1
"""
import os
import numpy as np 
import h5py  
import scipy.stats 
from utils import *
 

if not os.path.isdir("exp1"):
    os.makedirs("exp1")
os.chdir("exp1")

uncs = np.array([0.004, 0.015, 0.003, 0.004, 0.013, 0.010, 0.006])
uncs = np.array([0.003 ,  0.015 ,  0.004 ,  0.004  , 0.015 ,  0.008 ,  0.003])**2


N = 35


for site in ['h24v03', "h11v10", "h20v09"]:
    """
    Load data
    """ 
    datadir = "/group_workspaces/cems2/nceo_generic/users/jbrennan01/RRLandsat/models/"
    filein = "ref_%s_2008.hdf5" % site
    fi = h5py.File("%s/%s" % (datadir, filein))
    iso = fi['isotropic'][90:-90]
    sh = iso.shape
    shape = (sh[0], sh[-1], sh[-2])

    for sn in [0.5, 1.0, 1.5]:

        errors = uncs * sn 
        cov = np.diag(errors)
        for i in xrange(N):
            # make noise 
            noise = np.random.multivariate_normal(np.zeros(7), cov, size=shape)
            noise = np.swapaxes(noise, 3, 1)
            iso_copy = np.copy(iso)
            iso_copy += noise
            np.save("Exp1_%s_sn_%2f_%i" % (site, sn, i), (iso_copy*255).astype(np.uint8))


