"""
Exp2.py

    Full signal, with
    BRDF plus with additive noise 
    of low, medium and high values
""" 

import os
import numpy as np 
import h5py 


def angular1(brdf_settings):
    """
    simulate simple BRDF
    BRDF is not random so we enforce a
    certain structure onto the BRDF
    First the SZA is modelled as a simple
    sin across the year
    One approach is a climatology from
    the MODIS data?
    For this i think sin waves of
    diff freq will give some characteristic
    covering
    can vary max and min angles to sensors
    """
    # const
    scaleconst = 2*np.pi/366

    locals().update(brdf_settings)

    def scale(x, a=5, b=10, xmin=-1, xmax=1):
        """
        rescale the sin
        a new min
        b = new max
        xmin = min of x
        xmax = max of x
        """
        return (b - a)*(x - xmin)/(xmax - xmin) + a

    t = np.linspace(0, 2*np.pi, 366)


    noise = np.random.normal(0, 2*np.pi/100.0, size=366)

    szaMAX = 60
    szaMIN = 10
    sza_off = 0.5*np.pi # in pi

    sza_t = np.sin(noise + t + sza_off)
    SZA = scale(sza_t, a=szaMIN, b=szaMAX)


    # noisy it a bit?

    """
    vza cycle
    """
    vzaMAX = 45
    vzaMIN = 0
    vza_cycle = 6 # in days

    vza_t = np.sin(noise + t/(vza_cycle/366.0))
    VZA = scale(vza_t, a=vzaMIN, b=vzaMAX)

    """
    raa cycle
    """
    raaMAX = 360
    raaMIN = 0
    raa_cycle = 32 # in days

    raa_t = np.sin(t/(raa_cycle/366.0))
    RAA = scale(noise + vza_t, a=raaMAX, b=raaMIN)


    """
    only need to return kernels really
    """
    kerns = Kernels(VZA, SZA, RAA,
                LiType='Sparse', doIntegrals=False,
                normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                RossType='Thick',nbar=0.0)
    return kerns, VZA, SZA, RAA








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


    N = 100

    shape = (365, 300, 200)

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
                iso_copy += noise
                # add BRDF
                kerns, VZA, SZA, RAA = angular1(brdfDict['medium'])
                # simulate it
                angg = geo * kerns.Li + vol * kerns.Ross 
                # add BRDF
                iso_copy += angg 
                # make noise 
                noise = np.random.multivariate_normal(np.zeros(7), cov, size=shape)
                noise = np.swapaxes(noise, 3, 1)
                # add noise
                iso_copy += noise
                # save
                np.save("Exp2_%s_%i.npz", iso_copy)

