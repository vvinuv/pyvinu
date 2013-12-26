import pylab as pl 
import pyfits
import numpy as np
import os
import healpy as hp
from numpy import ma
from astropy.io import fits
import astropy.coordinates as co
from astropy import units as u
from astropy import wcs
import time

from matplotlib import ticker

def matrc1():
    MatPlotParams = {'axes.titlesize': 15, 'axes.linewidth' : 2.5, 'axes.labelsize': 22, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'xtick.major.size': 22, 'ytick.major.size' : 22, 'xtick.minor.size': 14, 'ytick.minor.size': 14, 'figure.figsize' : [6.0, 6.0], 'xtick.major.pad' : 8, 'ytick.major.pad' : 6}
    pl.rcParams.update(MatPlotParams)

def my_formatter(x, p):
    '''x and p are the ticks and the position. i.e. if the second tick is 100 th
en x=100, p=2'''
    if x == 0:
        pow = 0
        x = 0
    elif x < 0:
        pow = np.floor(np.log10(abs(x)))
    else:
        pow = np.floor(np.log10(x))
    return r'$%.1f \times 10^{%.0f}$' % (x/10**pow, pow)

def tick_format(ax, axis='x'):
    '''In the script call as 
       >>> import mytools as my
       >>> ax = pl.subplot(111)
       >>> my.tick_format(ax, axis='y')
       >>> my.tick_format(ax, axis='x')
       >>> pl.plot(a,a)
    '''

    if axis == 'x':
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(my_formatter))
    if axis == 'y':
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(my_formatter))

def whiskerplot(shear,dRA=1.,dDEC=1.,scale=5, combine=1,offset=(0,0) ):
    if combine>1:
        s = (combine*int(shear.shape[0]/combine),
             combine*int(shear.shape[1]/combine))
        shear = shear[0:s[0]:combine, 0:s[1]:combine] \
                + shear[1:s[0]:combine, 0:s[1]:combine] \
                + shear[0:s[0]:combine, 1:s[1]:combine] \
                + shear[1:s[0]:combine, 1:s[1]:combine]
        shear *= 0.25

        dRA *= combine
        dDEC *= combine


    theta = shear**0.5
    RA = offset[0] + np.arange(shear.shape[0])*dRA
    DEC = offset[1] + np.arange(shear.shape[1])*dDEC

    pl.quiver(RA,DEC,
                 theta.real.T,theta.imag.T,
                 pivot = 'middle',
                 headwidth = 0,
                 headlength = 0,
                 headaxislength = 0,
                 scale=scale)
    pl.xlim(0,shear.shape[0]*dRA)
    pl.ylim(0,shear.shape[1]*dDEC)
    #pl.xlabel('RA (arcmin)')
    #pl.ylabel('DEC (arcmin)')

def write_fits_table(outfile, keys, data, formats=None):
    '''Given keys and data it write it to fits table'''

    os.system('rm -f %s'%outfile)
    if formats is None:
        formats = ['E'] * len(keys)

    cols = []
    for key, d, format in zip(keys, data, formats):
        cols.append(pyfits.Column(name=key, format=format, array=d))

    cols = pyfits.ColDefs(cols)
    tbhdu = pyfits.new_table(cols)
    tbhdu.writeto(outfile)

def return_healang(pixel, nside=8, nest=False):
    """Given healpix pixel number returns RA and DEC """
    theta, phi = hp.pix2ang(nside, pixel, nest=False)
    dec = 90.0 - (theta * 180.0 / np.pi)
    ra = phi * 180.0 / np.pi
    return ra, dec

def return_healpixel(ra, dec, nside=8, nest=False):
    """Given RA and DEC in degrees returns array of pixel number"""
    theta = 90.0 - dec
    ra = ra * np.pi / 180.
    theta = theta * np.pi / 180.
    pixels = hp.ang2pix(nside, theta, ra, nest=False)
    return pixels


def get_random_cata_heal(ipath, ifile, mfile, nside, ofile='rand_points.fits', mode='grow', opath='./', do_plot=True):
    """Given an input catalog (ifile) and mask (mfile, in healpix format, 
       0-masked, 1-unmasked), this function generate
       a random catalog. The input catalog should contains ra and dec in 
       equitorial coordinate system and the output will in galactic coordinate.
       mode = grow -> grows the mask by one healpix pixel
       mode = shrink -> shrink the mask by one pixel (not yet implimented)
       mode = None -> Nothing will be done to the mask
    """

    ofile = os.path.join(opath, ofile)
    ifile = os.path.join(ipath, ifile)

    # Read input RA and DEC
    f = fits.open(ifile)[1].data
    ra = f['ra']
    dec = f['dec']

    # Convert RA DEC to l and b 
    coords = co.ICRS(ra=ra, dec=dec, unit=(u.degree, u.degree))
    ra = coords.galactic.l.degree
    dec = coords.galactic.b.degree

    # Generating the corresponding healpix pixels
    healpixels = return_healpixel(ra, dec, nside=nside, nest=False)
    uniquepixels = np.unique(healpixels)

    # Generating angular mask from input file
    mask = np.zeros(hp.nside2npix(nside)).astype(np.int)
    mask[uniquepixels] = 1

    if mode == 'grow':
        # Generating neighbour pixels. This will grow the mask
        neigh_pixels = hp.get_all_neighbours(nside, uniquepixels).ravel()
        neigh_pixels = np.unique(neigh_pixels)

        mask[neigh_pixels] = 1
    elif mode == 'shrink':
        pass

    #mask = hp.smoothing(mask, 0.01)
    #mask = np.where(mask > 0.1, 1, 0)

    # The given mask
    imask = hp.read_map(mfile)
    imask = hp.ud_grade(imask, nside)

    # Union of input catalog mask and input mask
    mask = (imask*mask).astype('bool')

    # Unique unmasked pixels
    uniquepixels = np.arange(hp.nside2npix(nside))[mask]

    # generating random points and their corresponding healpix pixels
    rra = np.random.uniform(0, 360, 100000)
    rdec = np.random.uniform(-90, 90, 100000)
    healpixels_rand = return_healpixel(rra, rdec, nside=nside, nest=False)
    
    # Random pixels found in unmasked pixels
    i = np.in1d(healpixels_rand, uniquepixels)
    rra = rra[i]
    rdec = rdec[i]

    write_fits_table(ofile, ['RA', 'DEC'], [rra, rdec])
    
    if do_plot: 
        hp.mollview(mask, coord='G', fig=1)
        hp.mollview(mask, coord='G', fig=2)
        pl.figure(3)
        pl.scatter(rra, rdec, c='k', label='Random')
        pl.scatter(ra, dec, c='r', s=0.2, edgecolor='r', label='Groups')
        pl.show()



def sigma_clipping(d, sigma, iter=10, mask=None):
    '''Sigma clipping d within sigma * std_dev. mask=0 for masked points'''

    i = 0
    tavg = -1e11
    if mask is None:
        mask = np.ones(d.shape).astype('bool')
    while i < iter:
        std_dev = d[mask].std()
        avg = d[mask].mean()
        mask *= np.where(abs(d - avg) > sigma * std_dev, 0, 1).astype('bool')
        frac_avg = abs(avg - tavg) / avg
        #print '%d %2.2f %2.2f %2.2f \n'%(i, avg, frac_avg, std_dev)

        if frac_avg < 0.10:
            i = iter
        tavg = avg * 1.0
        i += 1
    masked_d = ma.masked_array(d, np.logical_not(mask))
    return masked_d, avg, std_dev
    
        

def get_ra_dec(type, ifile, use_wcs=False, do_plot=False, field='KAPPA'):
    '''Given an image or table (ifile) this function returns the ra, dec of
       the pixels and the value in the pixel. If the input is image then it 
       should have WCS and the ra, dec are generated based WCS. 
       If use_wcs=False, then a simple convertion from pixel to WCS will be 
       performed based on a few parameters in the header. For table, this 
       just return the ra, dec columns and the field name is KAPPA by 
       default. Input and output coordinates must be equitorial 
    '''     
            
    if type == 'image':
        f = fits.open(ifile)
        kappa = f[0].data
        h = f[0].header
        f.close()
    
        #generate pixel coordinates
        pix_ra, pix_dec = np.meshgrid(np.arange(kappa.shape[1]),
                                      np.arange(kappa.shape[0]))
        if use_wcs:
            w = wcs.WCS(h)
            #corresponding equitorial coordinates 
            ra, dec = w.wcs_pix2world(pix_ra, pix_dec, 0)
        else:
            ra = h['CRVAL1'] + h['CDELT1'] * (pix_ra - h['CRPIX1'])
            dec = h['CRVAL2'] + h['CDELT2'] * (pix_dec - h['CRPIX2'])
        if do_plot:
            pl.subplot(121)
            pl.imshow(ra, origin='lower')
            pl.title('RA')
            pl.colorbar()
            pl.subplot(122)
            pl.imshow(dec, origin='lower')
            pl.title('DEC')
            pl.colorbar()
            pl.show()
    elif type == 'table':
        f = fits.open(ifile)[1].data
        ra = f['ra']
        dec = f['dec']
        kappa = f[field]
    else:
        raise ValueError("Input should be either image or table")
    
    return ra.ravel(), dec.ravel(), kappa.ravel()

def get_healpix_map(type, ipath, ifiles, nside, ofile='test.fits',
                    use_wcs=False, do_plot=False, field='KAPPA',
                    ocoord='galactic', opath='./'):
    '''Generage healpix map from given image(s) with WCS or table(s). Input WCS 
       should be equitorial. type is either image or table, ifiles must be 
       list or array of input images or tables. Output healpix (ocoord)
       map can be either in equitorial or galactic (default=galactic)'''
    
    ti = time.time()
    for i, f in enumerate(ifiles):  
        ifile = os.path.join(ipath, f)
        r, d, k = get_ra_dec(type, ifile, use_wcs=use_wcs, do_plot=False,
                             field=field)
        if i == 0:
            ra = r.copy()
            dec = d.copy()
            kappa = k.copy()
        else:
            ra = np.concatenate((ra, r))
            dec = np.concatenate((dec, d)) 
            kappa = np.concatenate((kappa, k))
    tf = time.time()
    
    print 'Read files in %2.2f sec'%(tf-ti)
 
    ti = time.time()
    if ocoord == 'equitorial':
        l = ra.copy()
        b = dec.copy()
        coord = 'C'
    elif ocoord == 'galactic':
        #equitorial to galactic 
        coords = co.ICRS(ra=ra, dec=dec, unit=(u.degree, u.degree))
        l = coords.galactic.l.degree
        b = coords.galactic.b.degree
        coord = 'G'

    tf = time.time()
    print 'Converted coordinates in %2.2f sec'%(tf-ti)

    #Generate healpix pixel
    healpixels = return_healpixel(l, b, nside=nside, nest=False)
    uniquepixels = np.unique(healpixels)

    ti = time.time()
    print 'Unique pixels in %2.2f sec'%(ti-tf)

    #Averaging pixels values belongs to same heal pixel
    kappa_map = [np.average(kappa[healpixels == i]) for i in uniquepixels]
    kappa_map = np.array(kappa_map)
    tf = time.time()
    print 'Averaged kappa in %2.2f sec'%(tf-ti)

    #Generating healpix base
    full_kappa_map = np.zeros(hp.nside2npix(nside)).astype(np.float64)
    full_kappa_map[uniquepixels] = kappa_map

    hp.write_map(ofile, full_kappa_map, coord=coord)

    if do_plot:
        hp.mollview(full_kappa_map, fig=1)
        pl.show()


if __name__=='__main__':
    pass
