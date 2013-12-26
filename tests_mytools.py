import mytools as my
import os

def tests_random_cata():
    path = '/data3/scratch/Planck/irsa.ipac.caltech.edu/data/Planck/release_1/all-sky-maps/maps'
   
    nside = 128 #2048
    ipath = '.'
    ifile = os.path.join(path, 'sdssgroups.fits')
    mfile = os.path.join('COM_PCCS_SZ-unionMask_2048_R1.11.fits')
    my.get_random_cata_heal(ipath, ifile, mfile, nside, ofile='rand_points_all_groups_xx.fits', mode=None, opath='./', do_plot=True)


def test_image_to_healpix():
    nside = 512
    ipath = '/home/vinu/Lensing/CFHT/Ludo'
    ifiles = ['W2_kapparecon_scale3.6.fits', 'W1_kapparecon_scale2.fits', \
              'W3_kapparecon_scale2.5.fits', 'W4_kapparecon_scale3.0.fits']
    ofile = 'CFHT_W2.fits'
    my.get_healpix_map('image', ipath, ifiles, nside, ofile=ofile, opath='./', use_wcs=True, do_plot=True, field='KAPPA', ocoord='equitorial')


if __name__=='__main__':
    #Test random catalog generator
    tests_random_cata() 

    #Testing converting image(s) or table(s) to healpix map
    test_image_to_healpix()     
