import numpy as np
import subprocess as sp
import sys
import os
from scipy.ndimage import binary_dilation,median_filter

### astronomy packages
from pyraf import iraf
from astropysics import coords
import pywcs
import pyfits
import cosmology as cosmos

def rescale_rexc(r_exc,z_low,z_high):
    d_low = cosmos.angular_distance(z_low)
    d_high = cosmos.angular_distance(z_high)    
    return r_exc*(d_low/d_high)
    
def dist2(x1,y1,x2,y2):
    "Computes the distance between two points with coordinates (x1,y1) and (x2,y2)"
    return ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))


def sex_to_degrees(ra,dec):
    ra=coords.AngularCoordinate(ra,sghms=True)
    dec=coords.AngularCoordinate(dec,sghms=False)
    return ra.degrees,dec.degrees

#==============================================================================
#  IMAGE Handling
#==============================================================================

def get_center_coords(imgname,ra,dec):
    """ Function ot convert from sky coordinates (RA,DEC) into image coordinates
    (XC,YC).
    """
    hdu=pyfits.open(imgname)
    wcs=pywcs.WCS(hdu[0].header)

    ctype=hdu[0].header["ctype1"]
    
    if 'RA' in ctype:
        sky=np.array([[ra,dec]],np.float_)
    else:
        sky=np.array([[dec,ra]],np.float_)

    pixcrd=wcs.wcs_sky2pix(sky,1)

    xc=pixcrd[0,0]
    yc=pixcrd[0,1]

    return xc,yc

def imagecut(imgname,ra,dec,path,exptime,hsize):

    hdu=pyfits.open(imgname)
    xc,yc=get_center_coords(imgname,ra,dec)

    nmgy=hdu[0].header['nmgy']
    xmax=hdu[0].header["naxis1"]
    ymax=hdu[0].header["naxis2"]
    
    
    if xc < 0 or xc > xmax:
        raise ValueError("Ra or Dec out of image range")
    if yc < 0 or yc > ymax:
        raise ValueError("Ra or Dec out of image range")
    
    if xc-hsize<=0:
        xc=hsize+1
    if yc-hsize<=0:
        yc=hsize+1
    
    if xc+hsize>xmax:
        xc=xmax-hsize-1
    if yc+hsize>ymax:
        yc=ymax-hsize-1

    xl=int(xc-hsize)
    xr=int(xc+hsize)
    yd=int(yc-hsize)
    yu=int(yc+hsize)

    if os.path.isfile(path+'galaxy.fits'):
        print "galaxy.fits exists, removing old file\n"
        os.system('rm '+path+'galaxy.fits')

    if os.path.isfile(path+'intermed.fits'):
        os.system('rm '+path+'intermed.fits')

    iraf.imcopy("%s[0]"%imgname,path+"intermed.fits")
    iraf.imarith("%sintermed.fits[%i:%i,%i:%i]"%(path,xl,xr,yd,yu),'*',(exptime/nmgy),path+"galaxy.fits")
    os.system('rm %sintermed.fits'%path)


    return xc,yc


#==============================================================================
# HEADER handling
#==============================================================================

def get_ids(imgfile):
    fname=imgfile.split('/')[-1]                            #split string by / to select the frame-******-*-****.fits full name
    parts=fname.split('-')                                  #split string by - to separete the numbers of the name of the file
    run=int(parts[2])                                       #set run to the 3rd number of the previous split 
    camcol=int(parts[3])                                    #set camcol to the 4th number of the previous split 
    field=int(parts[-1].split('.')[0])                      #set field to the last number of previous split (additional splitting is required to separate the .fits termination)
    return run,camcol,field
    
def header_keys(path,imgfile,band):
    run,camcol,field=get_ids(imgfile)                                       #get the numbers run, camcol and field for the frame image
    hdu=pyfits.open(path+'/photoField-%06i-%i.fits'%(run,camcol))            #open the photoField image using the retrieved numbers
    table=hdu[1].data                                                       #define the data table as the second extension of the fits file
    FIELDS=table.field('field')                                             #define the subtable containg the field numbers
    gain=table.field('gain')[FIELDS==field][0,int(band)]                  
    rdnoise=table.field('dark_variance')[FIELDS==field][0,int(band)]      #select subtables containg keywords of interest and selecting the values
    airmass=table.field('airmass')[FIELDS==field][0,int(band)]            # for the field of the frame image and the band ('ugriz'<->'12345')
    sky_nmgy=table.field('sky_frames_sub')[FIELDS==field][0,int(band)]    #check http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/photoField.html
#    print gain,rdnoise,airmass,sky_nmgy,'r'
    return gain,rdnoise,airmass,sky_nmgy                                    # for more details on all the keywords available

def update_header(path,imgfile,imgname,exptime=0.0,band=2):
    gain,rdnoise,airmass,sky_nmgy=header_keys(path,imgfile,band=band)       #get keywords values from photoField
    filters=['u','g','r','i','z']                                           #set the filters list of the SDSS
    hdu=pyfits.open(imgname,mode='update')                                  #open frame image to update
    hdr=hdu[0].header                                                       #set header variable
    if exptime!=0:
#    hdr.update('OLD_EXPT',hdr['EXPTIME'],'Old Exposure Time Value',after='EXPTIME') #copy EXPTIME keyword to OLD_EXPT
        hdr.update('EXPTIME',exptime,'Exposure Time of the High Redshift Field')   #set new EXPTIME = 1.0 to avoid miscalculated magnitudes
    hdr.update('GAIN',gain,'Gain value from photoField')                            #create gain keyword
    hdr.update('RDNOISE',rdnoise,'Dark_Variance value from photoField')             #create rdnoise keyword
    hdr.update('AIRMASS',airmass,'Airmass value from photoField')                   #create airmass keyword
    hdr.update('FILTERS',filters[band],'Filter of this image')                    #create filters keyword
    hdr.update('SKY_NMGY',sky_nmgy,'Sky value (in nmgy) obtained from photoField')  #create sky_nmgy keyword
    hdu.close()                                                                     #close file to save changes
    return None


#==============================================================================
#  Create PSF from SDSS psField
#==============================================================================

def prep_psf(psfield,xc,yc,path,band='g'):
    band_dict={'u':'1','g':'2','r':'3','i':'4','z':'5'}    
    
    bas=open(path+'psf.sh','w')
    idlcommand='psf_creator,"'+psfield+'",%i,%i,%s\n'%(xc,yc,band_dict[band])
    bas.write('#!/bin/bash\n')
    bas.write('idl << EOF\n')
    bas.write('\t.r psf\n')
    bas.write('\t'+idlcommand)
    bas.write('EOF')
    bas.close()
    os.system('chmod +x '+path+'psf.sh')
    os.chdir(path)
    os.popen('./psf.sh')
    os.chdir('..')
    return 'psf%s.fits'%(band_dict[band])

#==============================================================================
#  SExtractor
#==============================================================================

def gen_segmap_sex(fname,zeropoint,thresh=3.0,pix_scale=0.03,seeing=0.09,weights='none'):
    root=os.getcwd()
    "Returns the segmentation map generated from SExtractor"
    
    if 'linux' in sys.platform:
        sexcommand='sextractor'
    elif 'darwin' in sys.platform:
        sexcommand='sex'
    else:
        sexcommand='sextractor'
        
    if 'none' in weights:
        sp.call("%s %s -MAG_ZEROPOINT %.8f -CATALOG_NAME temp.cat -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME %s/map.fits -DETECT_MINAREA 5 -FILTER Y -DETECT_THRESH %f -ANALYSIS_THRESH %f -PIXEL_SCALE %.2f -SEEING_FWHM %.2f -WEIGHT_TYPE NONE "%(sexcommand,fname,zeropoint,root,thresh,thresh,pix_scale,seeing),shell=True)
        print("%s %s -MAG_ZEROPOINT %.8f -CATALOG_NAME temp.cat -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME %s/map.fits -DETECT_MINAREA 5 -FILTER Y -DETECT_THRESH %f -ANALYSIS_THRESH %f -PIXEL_SCALE %.2f -SEEING_FWHM %.2f -WEIGHT_TYPE NONE "%(sexcommand,fname,zeropoint,root,thresh,thresh,pix_scale,seeing))
    else:
        sp.call("%s %s -MAG_ZEROPOINT %.8f -CATALOG_NAME temp.cat -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME %s/map.fits -DETECT_MINAREA 5 -FILTER Y -DETECT_THRESH %f -ANALYSIS_THRESH %f -PIXEL_SCALE %.2f -SEEING_FWHM %.2f -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE %s"%(sexcommand,fname,zeropoint,root,thresh,thresh,pix_scale,seeing,weights),shell=True)
    segmap=pyfits.getdata('%s/map.fits'%root)
    sp.call('rm %s/map.fits'%(root),shell=True)
    return segmap

def get_sex_pars(xc,yc,rmax,catfile='test.cat',psf=False):
    """Returns the SExtractor parameters from the catalog associated with the
    segmentation map for the source closest to the VUDS catalog coordinates."""

    f=open(catfile,'r');txt=f.readlines();f.close()
    last_line=txt[-1]
    if last_line[0]=='#':        
        return -99,-99,-99,-99,-99,-99,-99,-99,-99,-99

    if psf:
        xs,ys,mag,mum,re,a,kr,e,t,cs,xp,yp=np.loadtxt(catfile,unpack=True,ndmin=2)
    else:
        xs,ys,mag,mum,re,a,kr,e,t,cs,xp,yp,isoarea=np.loadtxt(catfile,unpack=True,ndmin=2)

    n=a*kr/re
    e=1-e
    t=t
##    if np.size(xs)==1:
##        separation=np.sqrt(dist2(xs,ys,xc,yc))
##        if psf:
##            return xp,yp
##        else:
##            return [xs],[ys],[mag],[re],[n],[e],[t],0,separation,[isoarea]
##    else:
    dists=(dist2(xs,ys,xc,yc))
    obj_num=np.where(dists == min(dists))
    if min(dists)>rmax*rmax:
        return -99
    if psf:
        return xp,yp,obj_num
    else:
        return xs,ys,mag,re,n,e,t,obj_num,np.sqrt(min(dists)),isoarea

#==============================================================================
# MASK preparation
#==============================================================================

def sky_value(img,k=3):
    """Compute the value of sky background and respective standard
    deviation using a sigma clipping method."""
    media=np.nanmean(img)
    dev=np.nanstd(img)

    back=img.copy()
    back=back[back!=0]
    thresh=media+k*dev
    npix = len(img[abs(img)>=thresh])
    while npix>0:
        back = back[abs(back)<thresh]
        media=np.nanmean(back)
        dev=np.nanstd(back)
        thresh=media+k*dev
        npix = len(back[abs(back)>=thresh])
    return media,dev

def outlier_detection(img,k=15,s=1,mask=None):
    """ Flags the pixels that are recognized as outliers from a median_filter passage.
    """
    N,M=img.shape
    out_map=np.zeros(img.shape)
    blurred = median_filter(img,size=s+2)
    diff=img-blurred

    out_map=np.zeros(img.shape)
    ks=np.where(abs(diff)>np.mean(diff)+k*np.std(diff))  

    if mask is None:
        out_map[ks]=1        
    else:
        out_map[ks]=1-mask[ks]
        
#    fig,ax=subplots(1,4)
#    ax[0].imshow(img)
#    ax[1].imshow(blurred)
#    ax[2].imshow(diff)
#    ax[3].imshow(out_map)
#    show()
    return out_map

def distance_matrix(xc,yc,img):
    """Compute a matrix with distances and a sorted array with unique distance
    values from each pixel position to xc,yc"""
    N,M=img.shape
    Dmat=np.zeros([N,M])
    for i in range(N):
        for j in range(M):
            Dmat[i,j] = dist2(xc,yc,i,j)
    Dmat=np.sqrt(Dmat)
    dists=np.sort(np.array(list(set(Dmat.reshape(np.size(Dmat))))))
    return Dmat, dists
    
def select_object_map(xc,yc,segmap,pixscale,rexclusive):
    """Select the segmentation map of the object by settin all regions within
    approximantely 1 arcsecond to be assigned to the same object.
    """
    Dmat,d=distance_matrix(yc,xc,segmap)    
    

    
    s_values = set(segmap[Dmat<rexclusive/pixscale])
    new_map = segmap.copy()    
    
    for s in s_values:
        if s==0:
            continue
        new_map[new_map==s]=-1
    new_map[new_map>0]=0
    
#    import matplotlib.pyplot as mpl
#    from matplotlib.patches import Rectangle,Ellipse,Circle,Polygon
#    print rexclusive,rexclusive/pixscale
#    fig,ax=mpl.subplots(1,2)
#    ax[0].imshow(segmap)
#    C = Circle((xc,yc),radius=rexclusive/pixscale,color='red',fill=False)
#    ax[0].add_artist(C)
#    ax[1].imshow(-new_map)
#    mpl.show()
    
    return -new_map

def map_to_mask(img,segmap,xc,yc,maskname,kout,pixscale,rexc,s=1):
    """Converts the segmentation map from Sextractor into a mask file for GALFIT.
    Adds outlier pixels to the map and dilates the map to cover faint surface brightness
    regions."""
    segmap=segmap.copy()
    omap=outlier_detection(img,kout,s=s,mask=segmap)
#    imshow(omap);show()
    
    segmap[omap==1]=-1
    new_map=select_object_map(xc,yc,segmap,pixscale=pixscale,rexclusive=rexc)

#    imshow(new_map);show()

#    print len(segmap[segmap==-1])

    segmap[new_map==1]=0        ## deselecting object
    segmap[segmap!=0]=1         ## setting all segmap values to one
    segmap[img==0]=1            ## adding pixels where image as zero value
    segmap[segmap==-1]=1        ## adding pixel outliers that fall outside de fitting region
     
    SIZE=6
    STRUCTURE = np.zeros([SIZE,SIZE])
    dmat,d= distance_matrix(SIZE/2.-0.5,SIZE/2.-0.5,STRUCTURE)
    STRUCTURE[dmat<SIZE/2]=1
    segmap = binary_dilation(segmap,structure=STRUCTURE).astype(int)
     
    pyfits.writeto(maskname,segmap,clobber=True)
    return segmap

#==============================================================================
# GALFIT file creation
#==============================================================================

def write_object(f,model,x,y,m,re,n,ba,pa,num,fixpars=None):
    if fixpars==None:
        fixpars={'x':1,'y':1,'m':1,'re':1,'n':1,'q':1,'pa':1}
    f.write("#Object number: %i\n"%num)    
    f.write(' 0) %s             # Object type\n'%model)
    f.write(' 1) %6.4f %6.4f  %i %i    # position x, y        [pixel]\n'%(x,y,fixpars['x'],fixpars['y']))
    f.write(' 3) %4.4f      %i       # total magnitude\n' %(m,fixpars['m']))
    f.write(' 4) %4.4f       %i       #     R_e              [Pixels]\n'%(re,fixpars['re']))
    f.write(' 5) %4.4f       %i       # Sersic exponent (deVauc=4, expdisk=1)\n'%(n,fixpars['n']))  
    f.write(' 9) %4.4f       %i       # axis ratio (b/a)   \n'%(ba,fixpars['q']))
    f.write('10) %4.4f       %i       # position angle (PA)  [Degrees: Up=0, Left=90]\n'%(pa,fixpars['pa']))
    f.write(' Z) 0                  #  Skip this model in output image?  (yes=1, no=0)\n')
    f.write(' \n')
    return

def galfit_input_file(f,magzpt,sky,xsize,ysize,sconvbox,pixscale,imgname='galaxy.fits',outname="results.fits",psfname='psf.fits',maskname="none",signame='none',constname='none',fixpars=None):
    if fixpars==None:
        fixpars={'sky':1}
    f.write("================================================================================\n")
    f.write("# IMAGE and GALFIT CONTROL PARAMETERS\n")
    f.write("A) %s         # Input data image (FITS file)\n"%imgname)
    f.write("B) %s        # Output data image block\n"%outname)
    f.write("C) %s                # Sigma image name (made from data if blank or 'none' \n"%signame)
    f.write("D) %s         # Input PSF image and (optional) diffusion kernel\n"%psfname)
    f.write("E) 1                   # PSF fine sampling factor relative to data \n")
    f.write("F) %s                # Bad pixel mask (FITS image or ASCII coord list)\n"%maskname)
    f.write("G) %s                # File with parameter constraints (ASCII file) \n"%constname)
    f.write("H) 1    %i   1    %i # Image region to fit (xmin xmax ymin ymax)\n"%(xsize+1,ysize+1))
    f.write("I) %i    %i          # Size of the convolution box (x y)\n"%(sconvbox,sconvbox))
    f.write("J) %7.5f             # Magnitude photometric zeropoint \n"%magzpt)
    f.write("K) %.3f %.3f        # Plate scale (dx dy)   [arcsec per pixel]\n"%(pixscale,pixscale))
    f.write("O) regular             # Display type (regular, curses, both)\n")
    f.write("P) 0                   # Options: 0=normal run; 1,2=make model/imgblock and quit\n")
    f.write("\n")
    f.write("# INITIAL FITTING PARAMETERS\n")
    f.write("#\n")
    f.write("#For object type, the allowed functions are:\n")
    f.write("#nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat,\n")
    f.write("#ferrer, and sky.\n")
    f.write("#\n")
    f.write("#Hidden parameters will only appear when theyre specified:\n")
    f.write("#C0 (diskyness/boxyness),\n")
    f.write("#Fn (n=integer, Azimuthal Fourier Modes).\n")
    f.write("#R0-R10 (PA rotation, for creating spiral structures).\n")
    f.write("#\n")
    f.write("# ------------------------------------------------------------------------------\n")
    f.write("#  par)    par value(s)    fit toggle(s)   parameter description\n")
    f.write("# ------------------------------------------------------------------------------\n")
    f.write("\n")

    obj=open('galfit_object.temp','r')
    objects=obj.readlines()
    for line in objects:
        f.write(line)
    obj.close()
    
    f.write("# Object: Sky\n")
    f.write(" 0) sky                    #  object type\n")
    f.write(" 1) %7.4f      %i          #  sky background at center of fitting region [ADUs]\n"%(sky,fixpars['sky']))
    f.write(" 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n")
    f.write(" 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n")
    f.write(" Z) 0                      #  output option (0 = resid., 1 = Dont subtract)")    
    f.close()
    return

#==============================================================================
#  READ GALFIT results
#==============================================================================

def read_results_file(fname):
    try:
        hdu=pyfits.open(fname)
        chi=hdu[2].header['CHI2NU']
        xc=hdu[2].header['1_XC'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        yc=hdu[2].header['1_YC'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        M=hdu[2].header['1_MAG'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        R=hdu[2].header['1_RE'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        N=hdu[2].header['1_N'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        Q=hdu[2].header['1_AR'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        T=hdu[2].header['1_PA'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        F= hdu[2].header['FLAGS'].replace(' ',',')
        if ('1' in F.split(',')):
            F=1
        elif ('2' in F.split(',')):
            F=2
        else:
            F=0
    except IOError:
        xc,yc,M,R,N,Q,T,chi,F="-99 -99","-99 -99","-99 -99","-99 -99","-99 -99","-99 -99","-99 -99","-99",-9
    return xc,yc,M,R,N,Q,T,chi,F
