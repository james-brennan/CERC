"""
Exp2.py

    Partial Clean signal (BRDF corrected) 
    with additive noise of low, medium and high values;
""" 

import os
import numpy as np 
import h5py  
import scipy.stats 


def cloudModel(PsPs, PcPc):
    """
    simple markov chain model 
    for qa field
    """
    PsPc = 1 - PsPs
    PcPs = 1 - PcPc
    """
    transition matrix -- don't really need...
    """
    P = np.array([[PsPs, PsPc],
                  [PcPs, PcPc]])
    """
    initial condintion
    """
    sunny = True
    t0 = np.random.choice([False, True])
    """
    run realisation
    """
    clear = []
    t = t0
    for k in xrange(365):
        # predict tomorrow
        if t == sunny:
            t1 = bool(scipy.stats.bernoulli.rvs(PsPs))
        else:
            t1 = not bool(scipy.stats.bernoulli.rvs(PcPc))
        clear.append(t1)
        t = t1
    clear = np.array(clear)
    return clear  


if __name__ == "__main__":
    
    if not os.path.isdir("exp3"):
        os.makedirs("exp3")
    os.chdir("exp3")

    uncs = np.array([0.004, 0.015, 0.003, 0.004, 0.013, 0.010, 0.006])
    uncs = np.array([0.003 ,  0.015 ,  0.004 ,  0.004  , 0.015 ,  0.008 ,  0.003])**2


    N = 50

    shape = (365, 300, 200)

    """
    Load stuff
    """ 


    for site in ['h24v03']:#, "h11v10", "h20v09"]:
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
            # add clouds for partial
            qa  = cloudModel(0.8, 0.8)
            # set to -999 where qa is 0
            iso_copy[np.where(qa)[0], :, :, :]=-999
            # make noise 
            noise = np.random.multivariate_normal(np.zeros(7), cov, size=shape)
            noise = np.swapaxes(noise, 3, 1)
            # add noise
            iso_copy += noise

            """
            convert to int8 to compress files

            *255
            """
            # save
            np.save("Exp3_%s_%i.npz", (iso_copy*255).astype(np.uint8))

