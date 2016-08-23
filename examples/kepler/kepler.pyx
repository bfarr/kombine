import numpy as np
cimport numpy as np

cdef extern from "math.h":
  double sqrt(double x)
  double cos(double x)
  double sin(double x)
  double tan(double x)
  double fmod(double x, double y)
  double fabs(double x)
  double M_PI
  double atan(double x)

cdef double kepler_f(double M, double E, double e):
  return E - e*sin(E) - M

cdef double kepler_fp(double E, double e):
  return 1.0 - e*cos(E)

cdef double kepler_fpp(double E, double e):
  return e*sin(E)

cdef double kepler_fppp(double E, double e):
  return e*cos(E)

cdef double kepler_solve_ea(double n, double e, double t):
  cdef double M, E, f, fp, fpp, fppp, d
  
  M = fmod(n*t, 2.0*M_PI)

  if M < M_PI:
      E = M + 0.85*e
  else:
      E = M - 0.85*e

  f = kepler_f(M, E, e)
  while fabs(f) > 1e-8:
      fp=kepler_fp(E,e)
      disc = sqrt(fabs(16.0*fp*fp - 20.0*f*kepler_fpp(E,e)))
      if fp > 0.0:
          d=-5.0*f/(fp + disc)
      else:
          d=-5.0*f/(fp - disc)

      E += d
      f = kepler_f(M,E,e)

  return E

cdef double kepler_solve_ta(double n, double e, double t):
  cdef double E, f

  E = kepler_solve_ea(n,e,t)

  f = 2.0*atan(sqrt((1.0+e)/(1.0-e))*tan(E/2.0))

  if f < 0.0:
      f += 2.0*M_PI

  return f

cpdef np.ndarray[np.float_t, ndim=2] rv_model(np.ndarray[np.float_t, ndim=1] ts, 
                                              np.ndarray[np.float_t, ndim=1] Ks,
                                              np.ndarray[np.float_t, ndim=1] es,
                                              np.ndarray[np.float_t, ndim=1] omegas,
                                              np.ndarray[np.float_t, ndim=1] chis,
                                              np.ndarray[np.float_t, ndim=1] ns):
  cdef int i, j
  cdef int npl=Ks.shape[0], nts=ts.shape[0]
  cdef double t, K, e, omega, chi, n, t0, f, ecw
  cdef np.ndarray[np.float_t, ndim=2] rvs = np.zeros((npl, nts))

  for i in range(npl):
      K=Ks[i]
      e=es[i]
      omega=omegas[i]
      chi=chis[i]
      n=ns[i]
      t0 = -chi*2.0*M_PI/n
      ecw=e*cos(omega)

      for j in range(nts):
          t = ts[j]
          f = kepler_solve_ta(n, e, (t-t0))

          rvs[i,j] = K*(cos(f + omega) + ecw)

  return rvs
