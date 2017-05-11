from scipy.integrate import simps,quad
import numpy as np

""" COSMOLOGY MODULE
27/08/2013  - Define Angular Diameter Distance and Cosmological parameters
            - Compare function against http://en.wikipedia.org/wiki/File:Angular_diameter_distance.gif
"""
#### PLACNK COSMOLOGY
Obar=0.049                 #Baryon Density Parameter
Omat=0.3#0.3175                    #Matter Density Parameter
Ok=0.0                      #Curvature Density Parameter
Orad=0.0                    #Radiation Density Parameter
Ow=0.7#0.6825                      #Dark Energy Density Parameter
w=-1.0                      #DE equation of state parameter p/rho=w
H0=70.0#67.11                      #km/s/Mpc
Msun=1.989e30               #kg
Mpc=3.0857e19               #km
c=2.9979e5                  #km/s

def Hubble(z,pars=None):
    "Returns the value for the standard Hubble parameter at a redshift z"
    P={'h':H0/100.,'r':Orad,'m':Omat,'k':Ok,'l':Ow,'w':w}
    if not (pars==None):
        for p in pars:
            P[p] = pars[p]
    return 100*P['h']*np.sqrt(P['r']*(1+z)**4.+P['m']*(1+z)**3.+P['k']*(1+z)**2.+P['l']*(1+z)**(3*(1.+P['w'])))


def invHubble(z,pars=None):
    return 1.0/Hubble(z,pars=pars)

def comov_rad(z,pars=None,npoints=10000):
    """Returns the comoving radial distance corresponding to the redshift z in Mpc
    If nedeed, multiply by h to get result in units of (h**-1 Mpc)"""
    radius,err=quad(invHubble,0,z)
    return c*radius

def lookback_time(z,pars=None,npoints=10000):
    "Computes the lookback time (in Gyr) at a redshift z"
    if z==0:
        z=1e-4 
    z_points=np.linspace(1e-5,z,npoints)
    I = 1./((1+z_points)*Hubble(z_points,pars=pars))
    tl = simps(I,z_points)*Mpc/(365.25*24*60*60*1e9)
    return tl

def find_z_tL(time,pars=None):
    from scipy.optimize import newton
    def minim(z):
        return lookback_time(z,pars=None)-time
    res=newton(minim,0.5)
    return res
    
def angular_distance(z,pars=None):
    "Computes the angular diameter distance (Mpc) in a standard LCDM cosmology"
    return comov_rad(z,pars=pars)/(1+z)


def luminosity_distance(z,pars=None):
    "Computes the luminosity distance (Mpc) in a standard LCDM cosmology"
    return comov_rad(z,pars=pars)*(1+z)

def plot_lookback():
    import matplotlib.pyplot as pl
    Zs = np.linspace(1e-5,10,1000)   
    times=np.zeros(len(Zs))
    for i in range(len(Zs)):
        times[i]=lookback_time(Zs[i])
    fig,ax=pl.subplots()
    ax.plot(Zs,times,'k',lw=3)
    ax.hlines(13.8,0,10,'k',':')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$t_L\ [\mathrm{Gyr}]$')
    aux = fig.add_axes([0.5,0.2,0.3,0.3])
    aux.plot(Zs,times,'k',lw=3)
    aux.set_xlim(2,6)
    aux.set_ylim(10,13)
    aux.grid(True)
    pl.show()
     
def test_distances():
    import matplotlib.pyplot as pl
    Zs = np.linspace(1e-5,10,1000)
    distsA=np.zeros(len(Zs))
    distsA2=np.zeros(len(Zs))
    distsL=np.zeros(len(Zs))
    distsL2=np.zeros(len(Zs))
    for i in range(len(Zs)):
        distsA[i]=angular_distance(Zs[i])
        distsA2[i]=angular_distance(Zs[i],pars={'l':0.0,'m':1.0})
        distsL[i]=luminosity_distance(Zs[i])
        distsL2[i]=luminosity_distance(Zs[i],pars={'l':0.0,'m':1.0})
    
    fig,(ax1,ax2)=pl.subplots(1,2,figsize=(15,8))
    ax1.plot(Zs,distsA,'b-',Zs,distsA2,'r-')
#    ax1.set_xlim(2,5)
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$d_A$')
    ax2.plot(Zs,distsL,'b-',Zs,distsL2,'r-')
#    ax2.set_xlim(2,5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$d_L$')    
    pl.show()
    return

def plot_conversion_arcsec_to_kpc():
    import matplotlib.pyplot as mpl
    Zs = np.linspace(0.1,6.9,1000)

    dists_Planck=np.array([angular_distance(z) for z in Zs])

    kpc_per_arcsec = 1*dists_Planck/(180/np.pi*3600)*1000

    fig,ax=mpl.subplots(figsize=(15,8))

    ax.plot(Zs,1.0/kpc_per_arcsec,'-',color='RoyalBlue',lw=3)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$S\ [\mathrm{arcsec/kpc}]$')
    ax.minorticks_on()
    ax.set_xlim(0.1,6.5)
    fig.savefig('Scale_arcsec_per_kpc.png')
    mpl.show()

def compare_WMAP_Planck_cosmologies():
    import matplotlib.pyplot as pl
    Zs = np.linspace(0.1,6.9,1000)

    dists_Planck=np.array([angular_distance(z) for z in Zs])
    dists_WMAP=np.array([angular_distance(z,pars={'h':0.7,'m':0.3,'l':0.7}) for z in Zs])


    diff = dists_Planck/dists_WMAP

    fig,ax=pl.subplots(2,1,sharex=True,figsize=(15,13))
    pl.subplots_adjust(hspace=0.0)
    ax[1].plot(Zs,diff,'k-',lw=2)
    ax[1].set_xlabel(r'$z$')
    ax[1].set_ylabel(r'$D_A^\mathrm{Planck}/D_A^\mathrm{STD}$')
    ax[1].hlines(1,0,7,'r','--',alpha=0.5,lw=2)
    ax[1].set_xlim(0.1,6.9)
    ax[1].set_ylim(0.985,1.045)

    ax[0].plot(Zs,dists_WMAP,'r-',lw=2,label=r'STD: $H_0=70\mathrm{km\ s^{-1}Mpc^{-1}}; \Omega_m=0.3; \Omega_\Lambda=0.7$')
    ax[0].plot(Zs,dists_Planck,'g-',lw=2,label='Planck')
    ax[0].set_ylabel(r'$D_A\ [\mathrm{Mpc}]$')
    ax[0].legend(loc='lower right')
    ax[0].set_ylim(250,1900)
    fig.savefig('Impact_Cosmology_Choice.png')
    pl.show()

    
if __name__=='__main__':
#    test_distances()
#    plot_lookback()
#    compare_WMAP_Planck_cosmologies()
    plot_conversion_arcsec_to_kpc()
