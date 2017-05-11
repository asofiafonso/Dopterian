import numpy as np
import numpy.random as npr
import scipy.integrate as scint 
import scipy.optimize as scopt
import scipy.ndimage as scndi
import astropy.io.fits as pyfits
import matplotlib.pyplot as mpl
import .cosmology as cosmos
import astropy.modeling as apmodel
import astropy.convolution as apcon
import warnings

#==============================================================================
#  CONSTANTS
#==============================================================================
version = '1.0.0'

c = 299792458. ## speed of light


## SDSS maggies to lupton
magToLup = {'u':1.4e-10,'g':0.9e-10,'r':1.2e-10,'i':1.8e-10,'z':7.4e-10}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def nu2lam(nu):
    "From Hz to Angstrom"
    return c/nu*1e-10
    
def lam2nu(lam):
    "From Angstrom to Hz"
    return c/lam*1e10
    
def maggies2mags(maggies):
    return -2.5*np.log10(maggies)

def mags2maggies(mags):
    return 10**(-0.4*mags)

def maggies2fnu(maggies):
    "From maggies to units of [erg s-1 Hz-1 cm-2]"
    return 3531e-23*maggies 

def fnu2maggies(fnu):
    "From [erg s-1 Hz-1 cm-2] to maggies"
    return 3631e23*fnu

def fnu2flam(fnu,lam):
    return c*1e10/lam**2*fnu
    
def flam2fnu(flam,lam):
    return flam/c/1.0e10*lam**2

def lambda_eff(lam,trans):
    "Calculate the mean wavelength of a filter"
    indexs = np.where(lam != 0)[0]
    if len(indexs)==0:
        raise ValueError('ERROR: no non-zero wavelengths')
    else:
        Lambda=np.squeeze(lam[indexs])
        Transmission=np.squeeze(trans[indexs])
        return scint.simps(Lambda*Transmission,Lambda)/scint.simps(Transmission,Lambda)

def cts2maggies(cts,exptime,zp):
    return cts/exptime*10**(-0.4*zp)
    
def cts2mags(cts,exptime,zp):
    return maggies2mags(cts2maggies(cts,exptime,zp))

def maggies2cts(maggies,exptime,zp):
    return maggies*exptime/10**(-0.4*zp)
    
def mags2cts(mags,exptime,zp):
    return maggies2cts(mags2maggies(mags),exptime,zp)


def maggies2lup(maggies,filtro):
    b = magToLup[filtro]
    return -2.5/np.log(10)*(np.arcsinh(maggies/b*0.5)+np.log(b))
    
def lup2maggies(lup,filtro):
#    maggies=lup
    b = magToLup[filtro]
    return 2*b*np.sinh(-0.4*np.log(10)*lup-np.log(b))
    
def random_indices(size,indexs):
    "Returns an array of a set of indices from indexs with a size with no duplicates."
    return npr.choice(indexs,size=size,replace=False)

def edge_index(a,rx,ry):
    "The routine creates an index list of a ring with width 1 around the centre at radius rx and ry"
    N,M=a.shape
    XX,YY=np.meshgrid(np.arange(N),np.arange(M))
    
    Y = np.abs(XX-N/2.0).astype(np.int64)
    X = np.abs(YY-M/2.0).astype(np.int64)
    
    idx = np.where(((X==rx) * (Y<=ry)) + ((Y==ry) * (X<=rx)))
####    CHECK
    return idx


def dist_ellipse(img,xc,yc,q,ang):
    "Compute distance to the center xc,yc in elliptical apertures. Angle in degrees."
    ang=np.radians(ang)

    X,Y = np.meshgrid(range(img.shape[1]),range(int(img.shape[0])))
    rX=(X-xc)*np.cos(ang)-(Y-yc)*np.sin(ang)
    rY=(X-xc)*np.sin(ang)+(Y-yc)*np.cos(ang)
    dmat = np.sqrt(rX*rX+(1/(q*q))*rY*rY)
    return dmat

def resistent_mean(a,k):
    """Compute the mean value of an array using a k-sigma clipping method
    """
    
    media=np.nanmean(a)
    dev=np.nanstd(a)

    back=a.copy()
    back=back[back!=0]
    thresh=media+k*dev
    npix = len(a[a>=thresh])
    while npix>0:
        back = back[back<thresh]
        media=np.mean(back)
        dev=np.std(back)
        thresh=media+k*dev
        npix = len(back[back>=thresh])
        
    nrej = np.size(a[a>=thresh])
    return media,dev,nrej
  
def ring_sky(image,width0,nap,x=None,y=None,q=1,pa=0,rstart=None,nw=None):
    """ For an image measure the flux around postion x,y in rings with
    axis ratio q and position angle pa with nap apertures (for sky slope). rstart indicates 
    the starting radius and nw (if set) limits the width to the number of apertures
    and not calculated in pixels.
    """
    
    if nap<=3:
        raise ValueError('Number of apertures must be greater than 3.')

    if type(image)==str:
        image=pyfits.getdata(image)
    
    N,M=image.shape
    if rstart is None:
        rstart=0.05*min(N,M)
    
    if x is None and y is None:
        x=N*0.5
        y=M*0.5
    elif x is None or y is None:
        raise ValueError('X and Y must both be set to a value')
    else:
        pass
    
    rad = dist_ellipse(image,x,y,q,pa)
    max_rad=0.95*np.amax(rad)
    
    if nw is None:
        width=width0
    else:
        width=max_rad/float(width0)
    
    media,sig,nrej=resistent_mean(image,3)
    sig*=np.sqrt(np.size(image)-1-nrej)
    
    if rstart is None:
        rhi=width
    else:
        rhi=rstart
    
    nmeasures=2
    
    r=np.array([])
    flux=np.array([])
    i=0
    while rhi<=max_rad:
        extra=0
        ct=0
        while ct<10:
            idx = (rad<=rhi+extra)*(rad>=rhi-extra)*(np.abs(image)<3*sig)
            ct=np.size(image[idx])
            
            extra+=1
            if extra>max(N,M)*2:
                break
            
        if ct<5:
            sky = flux[len(flux)-1]
        else:
            sky = resistent_mean(image[idx],3)[0]
        
        r=np.append(r,rhi-0.5*width)
        flux=np.append(flux,sky)
        
        i+=1
        if np.size(flux) > nap:
            pars,err = scopt.curve_fit(lambda x,a,b:a*x+b,r[i-nap+1:i],flux[i-nap+1:i])
            slope=pars[0]
            if slope>0 and nmeasures==0:
                break
            elif slope>0:
                nmeasures-=1
            else:
                pass

        rhi += width
    sky = resistent_mean(flux[i-nap+1:i],3)[0]        
    return sky

def ferengi_make_psf_same(psf1,psf2):
    "Compares the size of both psf images and zero-pads the smallest one so that they have the same size"
    
    if np.size(psf1)>np.size(psf2):
        case=True
        big=psf1
        small=psf2
    else:
        big=psf2
        small=psf1
        case=False

    Nb,Mb=big.shape
    Ns,Ms=small.shape
    
    center = Nb/2
    small_side = Ns/2
    
    new_small=np.zeros(big.shape)
    new_small[center-small_side:center+small_side+1,center-small_side:center+small_side+1]=small
    
    if case==True:
        return psf1,new_small
    else:
        return new_small,psf2

def barycenter(img,segmap):
    """ Compute the barycenter of a galaxy from the image and the segemntation map.
    """
    N,M=img.shape
    XX,YY=np.meshgrid(range(M),range(N))
    gal=abs(img*segmap)
    Y = np.average(XX,weights=gal)
    X = np.average(YY,weights=gal)     
    return X,Y


def ferengi_psf_centre(psf,debug=False):
    "Center the psf image using its light barycenter and not a 2D gaussian fit."
    N,M=psf.shape
    
    assert N==M,'PSF image must be square'
    
    if N%2==0:
        center_psf=np.zeros([N+1,M+1])
    else:
        center_psf=np.zeros([N,M])

##    if debug:
##        print np.amax(psf),np.amin(psf)
##        mpl.imshow(psf);mpl.show()
        

    center_psf[0:N,0:M]=psf
    N1,M1=center_psf.shape

    X,Y=barycenter(center_psf,np.ones(center_psf.shape))
    G2D_model = apmodel.models.Gaussian2D(np.amax(psf),X,Y,3,3)

    fit_data = apmodel.fitting.LevMarLSQFitter()
    X,Y=np.meshgrid(np.arange(N1),np.arange(M1))
    with warnings.catch_warnings(record=True) as w:
        pars = fit_data(G2D_model, X, Y, center_psf)

    if len(w)==0:
        cenY = pars.x_mean.value
        cenX = pars.y_mean.value
    else:
        for warn in w:
            print warn
        cenX = center_psf.shape[0]/2+N%2
        cenY = center_psf.shape[1]/2+N%2
    
    dx =(cenX-center_psf.shape[0]/2)
    dy =(cenY-center_psf.shape[1]/2)
    
    center_psf= scndi.shift(center_psf,[-dx,-dy])
    
    
    return center_psf#,pars
    

def ferengi_deconvolve(wide,narrow):#TBD
    "Images should have the same size. PSFs must be centered (odd pixel numbers) and normalized."

    Nn,Mn=narrow.shape #Assumes narrow and wide have the same shape
    
    
    smax = max(Nn,Mn) 
    bigsz=2    
    while bigsz<smax:
        bigsz*=2

    if bigsz>2048:
        print 'Requested PSF array is larger than 2x2k!'
    
    psf_n_2k = np.zeros([bigsz,bigsz],dtype=np.double)
    psf_w_2k = np.zeros([bigsz,bigsz],dtype=np.double)
    
    psf_n_2k[0:Nn,0:Mn]=narrow
    psf_w_2k[0:Nn,0:Mn]=wide
    
#    fig,ax=mpl.subplots(1,2,sharex=True,sharey=True)
#    ax[0].imshow(psf_n_2k)
#    ax[1].imshow(psf_w_2k)
#    mpl.show()

    psf_n_2k=psf_n_2k.astype(np.complex_)
    psf_w_2k=psf_w_2k.astype(np.complex_)
    fft_n = np.fft.fft2(psf_n_2k)
    fft_w = np.fft.fft2(psf_w_2k)
    
    del psf_n_2k,psf_w_2k
    fft_n = np.absolute(fft_n)/(np.absolute(fft_n)+0.000000001)*fft_n
    fft_w = np.absolute(fft_w)/(np.absolute(fft_w)+0.000000001)*fft_w
    
    psf_ratio = fft_w/fft_n

    del fft_n,fft_w
    
#    Create Transformation PSF
    psf_intermed = np.real(np.fft.fft2(psf_ratio))
    psf_corr = np.zeros(narrow.shape,dtype=np.double)
    lo = bigsz-Nn/2
    hi=Nn/2
    psf_corr[0:hi,0:hi]=psf_intermed[lo:bigsz,lo:bigsz]    
    psf_corr[hi:Nn-1,0:hi]=psf_intermed[0:hi,lo:bigsz]    
    psf_corr[hi:Nn-1,hi:Nn-1]=psf_intermed[0:hi,0:hi]    
    psf_corr[0:hi,hi:Nn-1]=psf_intermed[lo:bigsz,0:hi]        
    del psf_intermed
    
    psf_corr = np.rot90(psf_corr,2)
    return psf_corr/np.sum(psf_corr)
    
def ferengi_clip_edge(image,auto_frac=2,clip_also=None,norm=False):#TBD
    N,M=image.shape
    rx = int(N/2/auto_frac)
    ry = int(M/2/auto_frac)
    
    sig=np.array([])
    r=np.array([])
    while True:
        idx = edge_index(image,rx,ry)
        if np.size(idx[0])==0:
            break
        med,sigma,nrej=resistent_mean(image,3)
        sigma*=np.sqrt(np.size(image)-1-nrej)
        sig=np.append(sig,sigma)
        r=np.append(r,rx)
        rx+=1
        ry+=1
    
    new_med,new_sig,new_nrej=resistent_mean(sig,3)
    new_sig*=np.sqrt(np.size(sig)-1-new_nrej)
    
    i=np.where(sig>=new_med*10*new_sig)
    if np.size(i)>0:
        lim = np.min(r[i])
        if np.size(i)>new_nrej*3:
            print 'Large gap?'
        npix = round(N/2.0-lim)
        
        if clip_also is not None:
            clip_also = clip_also[npix:N-1-npix,npix:M-1-npix]
        image=image[npix:N-1-npix,npix:M-1-npix]
    
    if norm==True:
        image/=np.sum(image)
        if clip_also is not None:
            clip_also/=np.sum(clip_also)
    
    if clip_also is not None:
        return npix,image,clip_also
    else:
        return npix,image

def rebin2d(img,Nout,Mout,flux_scale=False):
    """Special case of non-integer magnification for 2D arrays
    from FREBIN of IDL Astrolib.
    """

    N,M = img.shape

    xbox = N/float(Nout)
    ybox = M/float(Mout)

    temp_y = np.zeros([N,Mout])

    for i in range(Mout):
        rstart = i*ybox
        istart = int(rstart)

        rstop = rstart + ybox
        if int(rstop) > M-1:
            istop = M-1
        else:
            istop = int(rstop)

        frac1 = rstart-istart
        frac2 = 1.0 - (rstop-istop)       
        if istart == istop:
            temp_y[:,i] = (1.0-frac1-frac2)*img[:,istart]
        else:
            temp_y[:,i] = np.sum(img[:,istart:istop+1],1) - frac1 * img[:,istart] - frac2 * img[:,istop]

    temp_y = temp_y.transpose()
    img_bin = np.zeros([Mout,Nout])

    for i in range(Nout):
        rstart = i*xbox
        istart = int(rstart)

        rstop = rstart + xbox
        if int(rstop) > N-1:
            istop = N-1
        else:
            istop = int(rstop)

        frac1 = rstart-istart
        frac2 = 1.0 - (rstop-istop)

        if istart == istop:
            img_bin[:,i] = (1.0-frac1-frac2)*temp_y[:,istart]
        else:
            img_bin[:,i] = np.sum(temp_y[:,istart:istop+1],1) - frac1 * temp_y[:,istart]- frac2 * temp_y[:,istop]        

    if flux_scale:
        return img_bin.transpose()
    else:
        return img_bin.transpose()/(xbox*ybox)
        
        
def lum_evolution(zlow,zhigh):
    "Defined Luminosity evolution from L* of Sobral et al. 2013."
    def luminosity(z):
        logL = 0.45*z+41.87
        return 10**(logL)
    return luminosity(zhigh)/luminosity(zlow)

def ferengi_downscale(image_low,z_low,z_high,pix_low,pix_hi,upscale=False,nofluxscale=False,evo=None):

    da_in = cosmos.angular_distance(z_low)
    da_out = cosmos.angular_distance(z_high)
    
    dl_in=da_in*(1+z_low)**2#cosmos.luminosity_distance(z_low)
    dl_out=da_out*(1+z_high)**2#cosmos.luminosity_distance(z_high)
    
    if evo is not None:
##        evo_fact = evo(z_low,z_high)
        evo_fact = evo(0.0,z_high) ### UPDATED TO MATCH FERENGI ALGORITHM
    else:
        evo_fact=1.0
        
    mag_factor = (da_in/da_out)*(pix_low/pix_hi)
    if upscale == True:
        mag_factor=1./mag_factor
    
##    lum_factor = (dl_in/dl_out)**2
    lum_factor = (dl_in/dl_out)**2*(1.+z_high)/(1.+z_low) ### UPDATED TO MATCH FERENGI ALGORITHM
        
    if nofluxscale==True:
        lum_factor=1.0
        
    N,M = image_low.shape

    N_out = int(round(N*mag_factor))
    M_out = int(round(M*mag_factor))

    img_out = rebin2d(image_low,N_out,M_out,flux_scale=True)*lum_factor*evo_fact

    return img_out


def ferengi_odd_n_square():
    #TBD : in principle avoidable if PSF already square image
    #feregi_psf_centre already includes number of odd pixels
    raise NotImplementedError('In principle avoidable if PSF already square image')
    return


def ferengi_transformation_psf(psf_low,psf_high,z_low,z_high,pix_low,pix_high,same_size=None):
    """ Compute the transformation psf. Psf_low and psf_high are the low and high redshift PSFs respectively.
    Also needed as input paramenters the redshifts (low and high) and pixelscales (low and high).
    """    
    
    psf_l = ferengi_psf_centre(psf_low)
    psf_h = ferengi_psf_centre(psf_high)

    da_in = cosmos.angular_distance(z_low)
    da_out = cosmos.angular_distance(z_high)   
    
    N,M=psf_l.shape
    add=0
    
    out_size = round((da_in/da_out)*(pix_low/pix_high)*(N+add))
    
    while out_size%2==0:
        add+=2
        psf_l=np.pad(psf_l,1,mode='constant')
        out_size = round((da_in/da_out)*(pix_low/pix_high)*(N+add))
        if add>N*3:
            return -99
##            raise ValueError('Enlarging PSF failed!')
    
    psf_l = ferengi_downscale(psf_l,z_low,z_high,pix_low,pix_high,nofluxscale=True)
    psf_l = ferengi_psf_centre(psf_l)
    

# Make the psfs the same size (then center)
    psf_l,psf_h=ferengi_make_psf_same(psf_l,psf_h)  


    psf_l = ferengi_psf_centre(psf_l)
    psf_h = ferengi_psf_centre(psf_h)
    

# NORMALIZATION    
    psf_l/=np.sum(psf_l)
    psf_h/=np.sum(psf_h)

    return psf_l,psf_h,ferengi_psf_centre(ferengi_deconvolve(psf_h,psf_l))

def ferengi_convolve_plus_noise(image,psf,sky,exptime,nonoise=False,border_clip=None,extend=False):
    
    if border_clip is not None: ## CLIP THE PSF BORDERS (CHANGE IN INPUT PSF)
        Npsf,Mpsf=psf.shape
        psf = psf[border_clip:Npsf-border_clip,border_clip:Mpsf-border_clip]
    
    Npsf,Mpsf=psf.shape
    Nimg,Mimg=image.shape

    out = np.pad(image,Npsf,mode='constant') #enlarging the image for convolution by zero-padding the psf image size
    
    out = apcon.convolve_fft(out,psf/np.sum(psf))
    
    Nout,Mout=out.shape
    if extend==False:
        out=out[Npsf:Nout-Npsf,Mpsf:Mout-Mpsf]
        
    Nout,Mout=out.shape # grab new dimensions, if escaped
    
    Nsky,Msky=sky.shape

    if nonoise==False:
        ef= Nout%2
        try:
            out+=sky[Nsky/2-Nout/2:Nsky/2+Nout/2+ef,Msky/2-Mout/2:Msky/2+Mout/2+ef]+\
                 np.sqrt(np.abs(out*exptime))*npr.normal(size=out.shape)/exptime
        except ValueError:
##            raise ValueError('Sky Image not big enough!')
            return -99*np.ones(out.shape)
        
    return out

def dump_results(image,psf,imgname_in,bgimage_in,names_out,lowz_info,highz_info):
    name_imout,name_psfout = names_out

    Hprim = pyfits.PrimaryHDU(data=image)
    hdu = pyfits.HDUList([Hprim])
    hdr_img = hdu[0].header
    hdr_img['INPUT']=imgname_in
    hdr_img['SKY_IMG']=bgimage_in
    for key in lowz_info.keys():
        hdr_img['%s_i'%key[:4]]=(lowz_info[key],'%s value for input lowz object'%(key))
    hdr_img['comment']='Using ferengi.py version %s'%version
    for key in highz_info.keys():
        hdr_img['%s_o'%key[:4]]=(highz_info[key],'%s value for input highz object'%(key))
    hdu.writeto(name_imout,clobber=True)
    pyfits.writeto(name_psfout,psf,clobber=True)
    return
    
def ferengi(imgname,background,lowz_info,highz_info,namesout,imerr=None,noflux=False,evo=None,noconv=False,kcorrect=False,extend=False,nonoise=False,border_clip=3):

    Pl=pyfits.getdata(lowz_info['psf'])
    Ph=pyfits.getdata(highz_info['psf'])
    sky=pyfits.getdata(background)
    image=pyfits.getdata(imgname)
    if imerr is None:
        imerr=1/np.sqrt(np.abs(image))
    else:
        imerr=pyfits.getdata(imerr)
    
    if kcorrect:
        raise NotImplementedError('K-corrections are not implemented yet')
    else:
        img_nok = maggies2cts(cts2maggies(image,lowz_info['exptime'],lowz_info['zp']),highz_info['exptime'],highz_info['zp'])#*1000.
        img_downscale = ferengi_downscale(img_nok,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux)

        psf_low = Pl
        psf_hi = Ph

    
    median = scndi.median_filter(img_downscale,3)
    idx = np.where(np.isfinite(img_downscale)==False)
    img_downscale[idx]=median[idx]
    
    idx = np.where(img_downscale==0.0)
    img_downscale[idx]=median[idx]
    
    X=img_downscale.shape[0]*0.5
    Y=img_downscale.shape[1]*0.5 ## To be improved
    
    img_downscale-=ring_sky(img_downscale, 50,15,x=X,y=Y,nw=True)
    
    if noconv==True:
        dump_results(img_downscale/highz_info['exptime'],psf_low/np.sum(psf_low),imgname,background,namesout,lowz_info,highz_info)
        return img_downscale/highz_info['exptime'],psf_low/np.sum(psf_low)

    try:
        psf_low,psf_high,psf_t = ferengi_transformation_psf(psf_low,psf_hi,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'])
    except TypeError as err:
        print('Enlarging PSF failed! Skipping Galaxy.')
        return -99,-99

    try:
        recon_psf = ferengi_psf_centre(apcon.convolve_fft(psf_low,psf_t))
    except ZeroDivisionError as err:
        print('Reconstrution PSF failed!')
        return -99,-99
##    pyfits.writeto('transform_psf_dopterian.fits',psf_t,clobber=True)
    
    recon_psf/=np.sum(recon_psf)
    
    img_downscale = ferengi_convolve_plus_noise(img_downscale/highz_info['exptime'],psf_t,sky,highz_info['exptime'],nonoise=nonoise,border_clip=border_clip,extend=extend)
    if np.amax(img_downscale) == -99:
        print 'Sky Image not big enough!'
        return -99,-99

##    import matplotlib.pyplot as mpl
##    mpl.imshow(img_downscale);mpl.show()

    dump_results(img_downscale,recon_psf,imgname,background,namesout,lowz_info,highz_info)
    return img_downscale,recon_psf



if __name__=='__main__':
    PlowName='psf_sdss.fits'
    PhighName='psf_acs.fits'
    BgName='sky_ACSTILE_40x40.fits'
    InputImName='galaxy.fits'
    
    lowz_info = {'redshift':0.017,'psf':PlowName,'zp':28.235952,'exptime':53.907456,'filter':'r','lam_eff':6185.0,'pixscale':0.396}
    highz_info = {'redshift':0.06,'psf':PhighName,'zp':25.947,'exptime':6900.,'filter':'f814w','lam_eff':8140.0,'pixscale':0.03}
    
    import time as t
    t0=t.time()
#    imOUT,psfOUT = ferengi(InputImName,BgName,lowz_info,highz_info,['smooth_galpy_evo.fits','smooth_psfpy_evo.fits'],noconv=False,evo=lum_evolution)
    imOUT,psfOUT = ferengi(InputImName,BgName,lowz_info,highz_info,['smooth_galpy.fits','smooth_psfpy.fits'],noconv=False,evo=None)
    print 'elapsed %.6f secs'%(t.time()-t0)

#    fig,ax=mpl.subplots(1,2)
#    ax[0].imshow(imOUT)
#    ax[1].imshow(psfOUT)
#    mpl.show()
