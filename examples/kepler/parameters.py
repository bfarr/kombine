import numpy as np

class Parameters(np.ndarray):
    """Parameters for radial velocity fitting for a single telescope
    observing a single planet."""

    def __new__(subclass, arr=None, nobs=1, npl=1, 
                V=None, sigma0=None, sigma=None, tau=None, K=None, n=None, chi=None, e=None, omega=None):
        r"""Create a parameter object out of the given array (or a
        fresh array, if none given), with nobs observatories and npl
        planets.

        :param arr: An initial parameter array.

        :param nobs: The number of observatories in the parameters.

        :param npl: The number of planets in the parameters.

        :param V: The amplitude of the velocity offset in each
          observatory.

        :param sigma0: The RMS amplitude of the white noise
          fluctuations.

        :param sigma: The RMS amplitude of the correlated
          fluctuations.

        :param tau: The correlation time for the correlated
          fluctuations.  The correlated noise is assumed to follow a
          correlation function like

          ..math ::
          
            \left\langle v_i v_j \right\rangle = \sigma \exp\left[ \frac{\left| t_i - t_j \right|}{\tau} \right]

        :param K: The amplitude of the RV signal.

        :param n: The mean motion of the planetary orbit.

        :param chi: The fractional phase of the planet at the time
          origin.

        :param e: The eccentricity.

        :param omega: The argument of periapse."""
        assert nobs >= 1, 'must have at least one observatory'
        assert npl >= 0, 'must have nonnegative number of planets'
        assert arr is None or arr.shape[-1] == 4*nobs+5*npl, 'final array dimensions must match 4*nobs + 5*npl'

        if arr is None:
            arr = np.zeros(nobs*4+npl*5)
        
        obj = np.asarray(arr).view(subclass)

        obj._nobs = nobs
        obj._npl = npl

        if V is not None:
            obj.V = V

        if sigma0 is not None:
            obj.sigma0 = sigma0

        if sigma is not None:
            obj.sigma = sigma

        if tau is not None:
            obj.tau = tau

        if K is not None:
            obj.K = K

        if n is not None:
            obj.n = n

        if chi is not None:
            obj.chi = chi

        if e is not None:
            obj.e = e
    
        if omega is not None:
            obj.omega = omega

        return obj

    def __array_finalize__(self, other):
        if other is None:
            pass
        else:
            self._nobs = getattr(other, 'nobs', 1)
            self._npl = getattr(other, 'npl', 1)

    @property
    def header(self):
        """A suitable header to describe parameter data, including
        comment marker ('#') and newline."""
        if self.nobs == 1:
            header='# V sigma0 sigma tau '
        else:
            header='# '
            for i in range(self.nobs):
                header += 'V%d sigma0%d sigma%d tau%d '%(i,i,i,i)

        if self.npl == 0:
            header = header[:-1] + '\n'
        elif self.npl == 1:
            header += 'K n chi e omega\n'
        else:
            for i in range(self.npl):
                header += 'K%d n%d chi%d e%d omega%d '%(i,i,i,i,i)
            header = header[:-1] + '\n'
        
        return header

    @property
    def tex_header(self):
        """A list of latex strings describing each variable.  Does not
        include formula delimiters ('$')."""

        header=[]
        if self.nobs == 1:
            header.append('V')
            header.append(r'\sigma_0')
            header.append(r'\sigma')
            header.append(r'\tau')
        else:
            for i in range(self.nobs):
                header.append('V_{%d}'%i)
                header.append(r'\sigma_{%d)^0'%i)
                header.append(r'\sigma_{%d}'%i)
                header.append(r'\tau_{%d}'%i)

        if self.npl == 0:
            pass
        elif self.npl == 1:
            header.append('K')
            header.append('n')
            header.append(r'\chi')
            header.append(r'e')
            header.append(r'\omega')
        else:
            for i in range(self.npl):
                header.append('K_{%d}'%i)
                header.append('n_{%d}'%i)
                header.append(r'\chi_{%d}'%i)
                header.append(r'e_{%d}'%i)
                header.append(r'\omega_{%d}'%i)

        return header

    @property
    def V(self):
        """The velocity offset of the observatory or observatories."""
        return np.array(self[...,0:4*self.nobs:4])

    @V.setter
    def V(self, vs):
        if self.nobs == 1:
            self[..., 0] = vs
        else:
            self[...,0:4*self.nobs:4] = vs
       
    @property
    def sigma0(self):
        """The white noise magnitude."""
        return np.array(self[...,1:4*self.nobs:4])

    @sigma0.setter
    def sigma0(self, s0):
        if self.nobs == 1:
            self[...,1]=s0
        else:
            self[...,1:4*self.nobs:4] = s0

    @property
    def sigma(self):
        """The variance at zero lag of the telescope errors."""
        return np.array(self[...,2:4*self.nobs:4])

    @sigma.setter
    def sigma(self, s0):
        if self.nobs == 1:
            self[...,2]=s0
        else:
            self[...,2:4*self.nobs:4] = s0
        
    @property
    def tau(self):
        """The exponential decay timescale for correlations in
        telescope errors."""
        return np.array(self[...,3:4*self.nobs:4])

    @tau.setter
    def tau(self, t):
        if self.nobs == 1:
            self[...,3] = t
        else:        
            self[...,3:4*self.nobs:4] = t
        
    @property
    def K(self):
        """The amplitude of the radial velocity."""
        return np.array(self[...,4*self.nobs::5])
        
    @K.setter
    def K(self, k):
        if self.npl == 1:
            self[...,4*self.nobs] = k
        else:
            self[...,4*self.nobs::5] = k

    @property
    def n(self):
        """Mean motion (2*pi/P)."""
        return np.array(self[...,4*self.nobs+1::5])

    @n.setter
    def n(self, nn):
        if self.npl == 1:
            self[..., 4*self.nobs+1] = nn
        else:
            self[...,4*self.nobs+1::5] = nn
        
    @property
    def chi(self):
        """The fraction of an orbit completed at t = 0."""
        return np.array(self[...,4*self.nobs+2::5])

    @chi.setter
    def chi(self, c):
        if self.npl == 1:
            self[...,4*self.nobs+2] = c
        else:
            self[...,4*self.nobs+2::5] = c
        
    @property
    def e(self):
        """The orbital eccentricity."""
        return np.array(self[...,4*self.nobs+3::5])

    @e.setter
    def e(self, ee):
        if self.npl == 1:
            self[...,4*self.nobs+3]=ee
        else:
            self[...,4*self.nobs+3::5]=ee
        
    @property
    def omega(self):
        """The longitude of perastron."""
        return np.array(self[...,4*self.nobs+4::5])
        
    @omega.setter
    def omega(self, o):
        if self.npl == 1:
            self[...,4*self.nobs+4] = o
        else:
            self[...,4*self.nobs+4::5]=o

    @property
    def obs(self):
        """Returns an (N,4) array of observatory parameters."""
        return np.reshape(self[:4*self.nobs], (-1, 4))

    @obs.setter
    def obs(self, o):
        self[:4*self.nobs] = o

    @property
    def planets(self):
        """Returns an (N,5) array of planet parameters."""
        return np.reshape(self[4*self.nobs:], (-1, 5))

    @planets.setter
    def planets(self, p):
        self[4*self.nobs:] = p

    @property
    def nobs(self):
        return self._nobs

    @property
    def npl(self):
        return self._npl

    @property
    def P(self):
        return 2.0*np.pi/self.n
