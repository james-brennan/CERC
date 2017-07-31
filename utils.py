"""
utils.py
code for brdf and clouds
"""
import scipy.stats
import numpy as np 
from kernels import *


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



