import .auxiliary_functions_sdss as afs
import warnings
import argparse
import .dopterian

def lum_evolution(zlow,zhigh):
    "Defined Luminosity evolution from L* of Sobral et al. 2013."
    def luminosity(z):
        logL = 0.45*z+41.87
        return 10**(logL)
    return luminosity(zhigh)/luminosity(zlow)

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description="Code to download images of galaxies from catalog form SDSS DR9 server.")
parser.add_argument('catalog',metavar='NAME',type=str,help="Catalog on which to run GALFIT. Format should be ID,RA,DEC,...")
parser.add_argument('-z','--redshifts',metavar='NUM',type=str,help="Redshift of the smoothed images (Only aplicable when using --smooth).")
parser.add_argument('-i','--name',metavar='NAME',type=str,help="Name of the first galaxy to start with")
parser.add_argument('--highz_sky',metavar='fname',type=str,default="none",help="Image on which dopterian drops the high z galaxy.")
parser.add_argument('--highz_expt',metavar='pixel_scale',type=float,default=2028.,help="Pixel scale for the high redshift imaging.")
parser.add_argument('--highz_zp',metavar='zeropoint',type=float,default=25.947,help="Zeropoint for the high redshift imaging.")
parser.add_argument('--highz_psf',metavar='fname',type=str,help="PSF for the high redshift imaging.")
parser.add_argument('--ferengi',metavar='MODE',type=str,choices=('all','dimming','mag','none'),help="Options to run ferengi: with dimming and mag evolution, justo one of them or none.")
parser.add_argument('-b','--band',metavar='NAME',type=str,default='g',help="Band where to run ferengi. Default:g")
parser.add_argument('-t','--threshold',metavar='NAME',type=float,default=3.0,help="Threshold for SExtractor detections.")
parser.add_argument('-r','--rexclusive',metavar='NAME',type=float,default=20,help="Exlcusion radius for masking. Objects falling inside the given radius are not masked.")
parser.add_argument('-S','--seeing',metavar='NAME',type=float,default=0.09,help="Typical seeing of the images.")
parser.add_argument('--lowz_pixscale',metavar='pix',type=float,default=0.396,help="Pixelscale for the low redshift imaging.")
parser.add_argument('--highz_pixscale',metavar='pix',type=float,default=0.03,help="Pixelscale for the high redshift imaging.")

NOIMAGE=[]
band_dict={'u':'1','g':'2','r':'3','i':'4','z':'5'}
central_bands = {'u':3561,'g':4718,'r':6185,'i':7501,'z':8962}

if __name__=='__main__':
    args = parser.parse_args()

    SDSS_BANDS='ugriz'
    SDSS_PLATE=args.lowz_pixscale
    PS_HIGHZ=args.highz_pixscale ## pixscale high z imaging
    EXPTIME_SMOOTH=args.highz_expt # SDSS EXPTIME in seconds by default

    root=afs.os.getcwd()
    catalog_name=args.catalog
    startid=args.name
    threshold=args.threshold
    Rexclusion = args.rexclusive
    band=args.band
    SEEING=args.seeing
    MZPT=args.highz_zp
    redshifts = [float(z) for z in args.redshifts.split(',')]


    prefix = args.ferengi


    print("""
    GENERAL CONDITIONS FOR FERENGI and then GALFIT
    ----------------------------------------------------------------------------------
    Running DOPTERIAN on catalog %s and photometric bands %s
    Saving results to file %s
    
    DOPTERIAN mode = %s
    
    Pixel scale (highz) = %.3f
    Typical seeing (highz) = %.2f
    Mag. zeropoint (highz) = %.5f
    Exp. time (highz) = %.5f

    Pixel scale (lowz) = %.3f
    Mag. zeropoint (lowz) = Invidual (from image header)
    
    PSF image name: %s
    SKY image name: %s
    
    Detection threshold = %.1f sigma
    
    Start ID = %s
    ----------------------------------------------------------------------------------
    """%(catalog_name,band,'dopterian_%s_simulation_results_X.XXX.txt'%(args.ferengi),args.ferengi,PS_HIGHZ,SEEING,MZPT,EXPTIME_SMOOTH,args.lowz_pixscale,args.highz_psf,args.highz_sky,threshold,startid))

##    question=raw_input("Are these values ok? Do you want to continue? (y/n)\t: ")
##
##    if question=='n' or question=='no':
##        print "Now exiting program!"
##        afs.sys.exit()
##    else:
##        pass


    SExcatalog=[open('%s/dopterian_%s_SExcatalog_%.3f.txt'%(root,prefix,z),'w') for z in redshifts]
    [SExcatalog[k].write('# ID\tx\ty\tmag\tre\tn\tb/a\tPA\tSeparation\n') for k in range(len(redshifts))]

    table=[open('%s/dopterian_%s_simulation_results_%.3f.txt'%(root,prefix,z),'w') for z in redshifts]
    [table[k].write('# ID\tx\tx_err\ty\ty_err\tmag\tmag_err\tre\tre_err\tn\tn_err\tb/a\tb/a_err\tPA\tPA_err\tchi\tflag\tSeparation\n') for k in range(len(redshifts))]


    Info_Table=afs.np.loadtxt(catalog_name,dtype={'names':('ID','RA','DEC','Z'),'formats':('S50','S25','S25','f4')})
    ngalaxies = len(Info_Table['ID'])    
######

    start=0

    for i in range(ngalaxies):

        if (not Info_Table['ID'][i]==startid) and (start==0) and (startid!=None):
            continue
        else:
            start=1
        
        path="./%s"%(Info_Table['ID'][i])

        if not afs.os.path.isdir(path):
            print("\033[01;31m{0}\033[00m".format("Directory not found for galaxy %s"%Info_Table['ID'][i]))
            NOIMAGE.append(Info_Table['ID'][i])
            continue
        elif afs.os.path.isfile("%s/galaxy.fits"%path):
            input_name = "%s/galaxy.fits"%path
            psf_name = afs.sp.check_output('ls %s/psf%s.fits'%(path,band_dict[band]),shell=True).split()[0]
        else:
            print("\033[01;31m{0}\033[00m".format("galaxy.fits not found for galaxy %s"%Info_Table['ID'][i]))
            NOIMAGE.append(Info_Table['ID'][i])

        ra,dec=afs.sex_to_degrees(Info_Table['RA'][i],Info_Table['DEC'][i])

        hdu=afs.pyfits.open(input_name,mode='update')
        ximg,yimg=afs.get_center_coords(input_name,ra,dec)
        Nori,Mori = hdu[0].data.shape
        nmgy=hdu[0].header['nmgy']  #nanomaggies per count
        exptime=float(hdu[0].header['exptime'])
        zp_sdss=22.5-2.5*afs.np.log10(nmgy)     #mzp - 22.5 = -2.5log(nmgy/1)  calibration 1 nmgy -> 22.5 mag
        hdu.close()
        
        lowz_info = {'redshift':Info_Table['Z'][i],'psf':psf_name,'zp':zp_sdss,'exptime':exptime,'filter':band,'lam_eff':central_bands[band],'pixscale':SDSS_PLATE}
        highz_info = {'redshift':0.8,'psf':args.highz_psf,'zp':args.highz_zp,'exptime':args.highz_expt,'filter':'f814w','lam_eff':8140.0,'pixscale':PS_HIGHZ}

        for k in range(len(redshifts)):
            highz=float(redshifts[k])
            highz_info['redshift']=highz

            if args.ferengi=='all':
                dopterian.ferengi(input_name,args.highz_sky,lowz_info,highz_info,['%s/stamp_z%.3f.fits'%(path,highz),'%s/psf_z%.3f.fits'%(path,highz)],evo=lum_evolution,noflux=False)
            elif args.ferengi=='mag':
                dopterian.ferengi(input_name,args.highz_sky,lowz_info,highz_info,['%s/stamp_z%.3f.fits'%(path,highz),'%s/psf_z%.3f.fits'%(path,highz)],evo=lum_evolution,noflux=True)
            elif args.ferengi=='dimming':
                dopterian.ferengi(input_name,args.highz_sky,lowz_info,highz_info,['%s/stamp_z%.3f.fits'%(path,highz),'%s/psf_z%.3f.fits'%(path,highz)],evo=None,noflux=False)
            elif args.ferengi=='none':
                dopterian.ferengi(input_name,args.highz_sky,lowz_info,highz_info,['%s/stamp_z%.3f.fits'%(path,highz),'%s/psf_z%.3f.fits'%(path,highz)],evo=None,noflux=True)
            else:
                raise ValueError('Invalid mode selected: %s'%args.ferengi)

####################################################################################################################################################
####################################################################################################################################################
            imgname='%s/dopterian_galaxy_z%.3f_%s.fits'%(path,highz,args.ferengi)
            if afs.os.path.isfile('%s/stamp_z%.3f.fits'%(path,highz)):
                afs.iraf.imarith('%s/stamp_z%.3f.fits'%(path,highz),'*',highz_info['exptime'],imgname)
                afs.iraf.hedit(imgname,'EXPTIME',highz_info['exptime'],addonly='yes',verify='no',show='yes')
                stamp_red = afs.pyfits.getdata(imgname)
            else:
                table[k].write('%s\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\n'%(Info_Table['ID'][i]))
                SExcatalog[k].write('%s\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n'%(Info_Table['ID'][i],-99,-99,-99,-99,-99,-99,-99,-99))
                continue
                
            N,M=stamp_red.shape
            assert N==M
            xmed = ximg*(float(N)/Nori)
            ymed = yimg*(float(M)/Mori)
            
            segmap = afs.gen_segmap_sex(imgname,highz_info['zp'],pix_scale=highz_info['pixscale'],thresh=threshold,seeing=SEEING)
            afs.map_to_mask(stamp_red,segmap,xmed,ymed,'%s/mask_z%.3f.fits'%(path,highz),10,highz_info['pixscale'],afs.rescale_rexc(Rexclusion,lowz_info['redshift'],highz_info['redshift']))
            xs,ys,mag,re,n,e,t,num,sep,isoarea=afs.get_sex_pars(xmed,ymed,N,catfile='temp.cat')
            
            try:
                len(xs)
            except TypeError:
                xs,ys,mag,re,n,e,t=[xs],[ys],[mag],[re],[n],[e],[t]
                num=0
            mcorr=2.5*afs.np.log10(highz_info['exptime'])


            SExcatalog[k].write('%s\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n'%(Info_Table['ID'][i],xs[num],ys[num],mag[num]+mcorr,re[num],n[num],e[num],t[num],sep))
            print 'SEx Pars',xs[num],ys[num],mag[num]+mcorr,re[num],n[num],e[num],t[num]                                         
    
##            if sep >10:
##                print "No galaxy detected at X,Y=center,center"
##                table[k].write('%s\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\n'%(Info_Table['ID'][i]))
##                sp.call('rm %s/mask_z%.3f.fits'%(path,highz),shell=True)
##                k+=1
##                continue
    
            if sep==-99.0:
                print "Nothing detected in stamp image"
                table[k].write('%s\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\t-99.0000\n'%(Info_Table['ID'][i]))
                afs.sp.call('rm %s/mask_z%.3f.fits'%(path,highz),shell=True)
                continue
            
            
            f1=open('%s/galfit_object.temp'%(root),'w')              
            afs.write_object(f1,'sersic',xs[num],ys[num],mag[num]+mcorr,re[num],n[num],e[num],t[num],0)
            f1.close()
            afs.sp.call("rm temp.cat",shell=True,stderr=afs.sp.PIPE)

            
####################################################################################################################################################
####################################################################################################################################################
 

            f2=open('%s/galfit_file_z%.3f'%(root,highz),'w')
            afs.galfit_input_file(f2,highz_info['zp'],0.0,N,M,N,PS_HIGHZ,imgname=imgname,psfname='%s/psf_z%.3f.fits'%(path,highz),outname='%s/result_z%.3f.fits'%(path,highz),maskname='%s/mask_z%.3f.fits'%(path,highz))
            f2.close()
    
            afs.sp.call('galfit galfit_file_z%.3f >> galfit.log'%(highz),shell=True,stderr=afs.sp.PIPE)
#            sp.call('galfit galfit_file%i_z%.3f'%(runtype,highz),shell=True)
            X,Y,MAG,R,S,Q,T,CHI,F=afs.read_results_file('%s/result_z%.3f.fits'%(path,highz))
            print 'GALFIT Pars',X,Y,MAG,R,S,Q,T,CHI,F                                          

            table[k].write('%s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%i\t%3.2f\n'%(Info_Table['ID'][i],X,Y,MAG,R,S,Q,T,CHI,F,sep))
            
            if afs.os.path.isdir('%s/z%.2f'%(path,highz)):
                pass
            else:
                afs.sp.call('mkdir %s/z%.2f'%(path,highz),shell=True)
            afs.sp.call('cp %s %s/z%.2f'%(imgname,path,highz),shell=True)
            afs.sp.call('cp %s/galfit_file_z%.3f %s/z%.2f/dopterian_galfit_file_z%.3f '%(root,highz,path,highz,highz),shell=True)
            
        afs.sp.call('rm %s/dopterian*.fits %s/stamp_z*.fits %s/result_z*.fits %s/psf_z*.fits galfit_objects.temp galfit_file_z* galfit.* fit.log %s/mask_z*.fits'%(path,path,path,path,path),shell=True)

        print "--------------> Done with galaxy %s \n"%Info_Table['ID'][i]

    [SExcatalog[k1].close() for k1 in range(len(redshifts))]
    [table[k1].close() for k1 in range(len(redshifts))]
