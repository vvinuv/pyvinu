import numpy as np
import cosmolopy.distance as cd
import pylab as pl


class LensEfficiency:

    def __init__(self):
        self.cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 
                      'omega_k_0':0.0, 'h':0.72}

    def dNdz_func(self, zstar, alpha):
        """Mandelbaum (2008) Eq. 9 """
        self.zstar = zstar
        self.alpha = alpha
        z = np.linspace(0, 3, 100)
        dNdz = (z/self.zstar)**(self.alpha - 1.) * \
               np.exp(-0.5 * (z/self.zstar)**2.)
        a = (dNdz[1:] + dNdz[:-1])/2.
        a_int = np.sum(a * (z[1]-z[0]))
        dNdz /= a_int #Normalizing so that integral(dN/dz)=1
        return z, dNdz
    
    def efficiency(self, z, dNdz):
        """Given dN/dz, it calculate both normalize and unnormalized 
           lensing efficiency function. The inputs are the z array at which
           dN/dz is estimated. See notebook
           """
        self.a = 1 / (1+ z)
        self.zarr = z.copy()
        self.dz = self.zarr[1] - self.zarr[0]

        c_light = 3e5 #km/s   
        constant = ((100. * self.cosmo['h'])**2 * self.cosmo['omega_M_0']) * \
                   (3/2.) * (1/c_light**2) 
        
        chi = cd.comoving_distance(self.zarr, **self.cosmo)
        
        chi_pz = []
        for chi_l in chi:
            con = (chi > chi_l)
            f1 = (chi[con] - chi_l) * dNdz[con] 
            f1 = (f1[1:] + f1[:-1]) / 2.
            dDs = (chi[con][1:] + chi[con][:-1]) / 2.
            chi_pz.append(np.sum(f1 * dDs))
  
        chi_pz = np.array(chi_pz)

        self.e = chi * chi_pz / self.a

        self.de = (self.e[1:] + self.e[:-1]) / 2.
        
        self.int_e = np.sum(self.de * self.dz)

        self.norm_e = self.e / self.int_e

    def efficiency_single(self, zs):
        self.zs = zs
        self.zarr = np.linspace(0, self.zs+.0, 100)
        self.a = 1 / (1+ self.zarr)
        self.dz = self.zarr[1] - self.zarr[0]

        c_light = 3e5 #km/s   
        constant = ((100. * self.cosmo['h'])**2 * self.cosmo['omega_M_0']) * \
                   (3/2.) * (1/c_light**2) 
        
        chi = cd.comoving_distance(self.zarr, **self.cosmo)
        chi_s = cd.comoving_distance(self.zs, **self.cosmo)
 
        self.e = chi * (chi_s - chi) / (chi_s * self.a)

        self.de = (self.e[1:] + self.e[:-1]) / 2.
        
        self.int_e = np.sum(self.de * self.dz)

        self.norm_e = self.e / self.int_e


    def plot(self):
        try:
            pl.plot(self.zarr, self.norm_e, label=r'$z_s=%2.1f$'%self.zs)
        except:
            try:
                pl.plot(self.zarr, self.norm_e, label=r'$z_*=%2.1f \alpha=%2.1f$'%(self.zstar, self.alpha))
            except:
                pl.plot(self.zarr, self.norm_e, label=r'Test')



if __name__=='__main__':
   
    #Plotting lensing efficiency functions for different parameters 
    L = LensEfficiency()
   
    alpha = 10.
    for zstar in np.linspace(0.2, 1.2, 6):
        zarr, dNdz = L.dNdz_func(zstar, alpha)
        pl.figure(1)
        pl.subplot(211)
        pl.plot(zarr, dNdz, label=r'$z_*=%2.1f \alpha=%2.1f$'%(zstar, alpha))
        pl.subplot(212)
        L.efficiency(zarr, dNdz)
        L.plot()

    pl.subplot(211)
    pl.xticks(visible=False)
    pl.ylabel('p(z)')
    pl.legend()
 
    pl.subplot(212)
    pl.xlabel('z')
    pl.ylabel('w(z)')
    pl.legend()
    pl.show()

    #Comparing lensing efficiency functions estimated using 
    #efficiency_single() and efficiency() 
    L.efficiency_single(0.5)
    L.plot()
    zarr = np.linspace(0, 2., 100)
    dNdz = np.zeros(zarr.shape)
    dNdz[(zarr<=0.51) & (zarr >=0.49)] = 1.
    L.efficiency(zarr, dNdz)
    L.plot()
    pl.xlabel('z')
    pl.ylabel('w(z)')
    pl.legend()
    pl.show()

    #Plotting lensing efficiency functions for different source redshifts 
    for zs in np.linspace(0.2, 2., 10):
        L.efficiency_single(zs)
        L.plot()
    pl.xlabel('z')
    pl.ylabel('w(z)')
    pl.legend()
    pl.show()

