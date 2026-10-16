'''

Function to store useful functions for modeling

NOTE: This should not be very problem specific

cjperks
June 1st, 2026

'''

# Modules
import sys, os
import numpy as np
from scipy.special import erf, wofz
from scipy.integrate import quad

__all__ = [
    # Statistic function
    '_cstat',
    '_cstat_signed',
    '_chi2stat',
    '_chi2stat_squared',
    # Model functions
    '_bck_int',
    '_gauss_pt',
    '_gauss_int',
    '_voigt_pt',
    '_voigt_int',
    '_voigt_fwhm',
    # Other
    '_kahanSum',
    ]

####################################################################
#
#                   Statistic functions
#
####################################################################

# Calculates CSTAT residuals
def _cstat(
    ydata,          # Measured data
    model_value     # Model data
    ):

    # Error correction, #counts --> positive definite
    new_ydata = np.where(
        ydata <= 0, 
        sys.float_info.min, 
        ydata
        )
    new_log_model = np.where(
        np.log(model_value) <= 0, 
        sys.float_info.min_10_exp, 
        np.log(model_value)
        )

    # CSTAT = 2*Sum_i[Mi - Di + Di*(logDi - logMi)]
    # Mi = model, Di = data
    # https://cxc.cfa.harvard.edu/sherpa/statistics/
    cash_statistic = new_ydata - model_value
    cash_statistic += new_ydata * (new_log_model - np.log(new_ydata))
    cash_statistic *= -2

    # Output, dim(ydata,)
    return cash_statistic

# Calculates signed CSTAT residuals
def _cstat_signed(
    ydata,          # Measured data
    model_value     # Model data
    ):

    # Gets CSTAT residuals
    cstat = _cstat(ydata, model_value)

    # Output
    return (
        np.sign(ydata - model_value)
        * np.sqrt(np.abs(cstat))
        )

# Calculates chi2 residuals
def _chi2stat(
    ydata,          # Measured data
    model_value     # Model data
    ):

    # Error correction, #counts --> positive definite
    new_ydata = np.where(
        ydata <= 0, 
        sys.float_info.min, 
        ydata
        )

    # Gehrels variance function
    # NOTE: Assuming background isn't subtracted!!!
    # Assumes number of counts per bin is small (about <5)
    #sigs = (
    #    1 +
    #    np.sqrt(new_ydata + 0.75)
    #    )

    # Basic variance function
    sigs = np.sqrt(
        np.where(ydata <=0.0, 1.0, ydata)
        )


    # chi2 = Sum_i[(Di - Mi)/sigs]
    # Mi = model, Di = data
    # https://cxc.cfa.harvard.edu/sherpa/statistics/
    chi2_statistic = (new_ydata - model_value)
    chi2_statistic /= sigs

    # Output, dim(ydata,)
    return chi2_statistic

# Calculates chi2 residuals
def _chi2stat_squared(
    ydata,          # Measured data
    model_value     # Model data
    ):

    # Error correction, #counts --> positive definite
    new_ydata = np.where(
        ydata <= 0, 
        sys.float_info.min, 
        ydata
        )

    # Gehrels variance function
    # NOTE: Assuming background isn't subtracted!!!
    # Assumes number of counts per bin is small (about <5)
    #sigs = (
    #    1 +
    #    np.sqrt(new_ydata + 0.75)
    #    )

    # Basic variance function
    sigs = np.sqrt(
        np.where(ydata <=0.0, 1.0, ydata)
        )

    # chi2 = Sum_i[(Di - Mi)**2/sigs**2]
    # Mi = model, Di = data
    # https://cxc.cfa.harvard.edu/sherpa/statistics/
    chi2_statistic = (new_ydata - model_value)**2
    chi2_statistic /= sigs**2

    # Output, dim(1,)
    return np.sum(chi2_statistic)

####################################################################
#
#                   Model functions
#
####################################################################

# Functional form for linear background on binned data
def _bck_int(
    binlow,     # Data x-axis, [eV], dim(ydata,)
    binhigh,
    AA,         # Amplitude, [counts/eV], dim(1,)
    ):

    # Prep
    #binsize = binlow[1]-binlow[0]   # [eV], dim(1,)
    binsize = binhigh - binlow       # [AA], dim(ydata)

    # Output
    return (
        AA
        #*binsize
        *np.ones_like(binlow, dtype=float)
        )   # [counts], dim(ydata,)

# Functional form for Gaussian on point-evaluated data
def _gauss_pt(
    binlow,         # Data x-axis, [eV], dim(ydata,)
    binhigh,
    AA,             # Curve area, [counts], dim(1,)
    mu,             # Curve center, [eV], dim(1,)
    sigma           # Curve variance, [eV], dim(1,)
    ):
    '''
    # NOTE: Valid if bin size << line width

    We want to model, M(x), the partial area within the i-th bin
    --> A_i = \int_{binlow_i}^{binhigh_i} M(x)*dx
    limit = M(bincent)*(binhigh_i-binlow_i)
    '''

    # Prep
    #binsize = binlow[1]-binlow[0]   # [eV], dim(1,)
    #bincent = binlow + binsize/2    # [eV], dim(ydata,)
    binsize = binhigh-binlow         # [AA], dim(ydata,)
    bincent = (binhigh + binlow)/2      # [AA], dim(ydata,)

    # Output
    return (
        AA
        /(np.sqrt(2*np.pi)*sigma)
        *np.exp(
            -1/2
            *(bincent - mu)**2
            /sigma**2
            )
        * binsize
        )   # [counts], dim(ydata,)

# Functional form for Gaussian on binned data
def _gauss_int(
    binlow,         # Data x-axis, [eV], dim(ydata,)
    binhigh,
    AA,             # Curve area, [counts], dim(1,)
    mu,             # Curve center, [eV], dim(1,)
    sigma           # Curve variance, [eV], dim(1,)
    ):
    '''
    # NOTE: Preferred if bin size ~ line width
    '''

    # Prep
    #binsize = binlow[1]-binlow[0]   # [eV], dim(1,)
    #binhigh = binlow + binsize      # [eV], dim(ydata,)

    # See https://en.wikipedia.org/wiki/Error_function
    # for identity of integrating Gaussian over a finite interval    
    return (
        0.5 * AA * (
            erf((binhigh - mu) / (sigma * np.sqrt(2))) 
            - erf((binlow - mu) / (sigma * np.sqrt(2)))
            )
        )   # [counts], dim(ydata,)

# Functional form for Voigt on point-evaluated data
def _voigt_pt(
    binlow,         # Data x-axis, [eV], dim(ydata,)
    binhigh,
    AA,             # Curve area, [counts], dim(1,)
    mu,             # Curve center, [eV], dim(1,)
    sigma,          # Curve Gaussian variance, [eV], dim(1,)
    gamma,          # Curve Lorentzian half-width, [eV], dim(1,)
    ):
    '''
    Voigt FWHM ~ 0.5346*f_L + np.sqrt(0.2166*f_L**2 + f_G**2)

    where,
    f_G = 2.3548*sigma, Gaussian FWHM
    f_L = 2*gamma, Lorentzian FWHM

    https://en.wikipedia.org/wiki/Voigt_profile
    '''

    # Prep
    #binsize = binlow[1]-binlow[0]   # [eV], dim(1,)
    #bincent = binlow + binsize/2    # [eV], dim(ydata,)
    bincent = (binhigh + binlow)/2      # [AA], dim(ydata,)

    # Calculation
    zz = (
        ((bincent - mu) + 1j * gamma)
        / (sigma *np.sqrt(2))
        )   # [], dim(ydata,)
    return (
        AA
        * np.real(wofz(zz))
        / (sigma * np.sqrt(2 *np.pi))
        )   # [counts], dim(ydata,)

# Functional form for Voigt on binned data
def _voigt_int(
    binlow,         # Data x-axis, [eV], dim(ydata,)
    binhigh,
    AA,             # Curve area, [counts], dim(1,)
    mu,             # Curve center, [eV], dim(1,)
    sigma,          # Curve Gaussian variance, [eV], dim(1,)
    gamma,          # Curve Lorentzian half-width, [eV], dim(1,)
    ):
    '''
    Integrate Voigt profile over each bin using Gauss-Legendre quadrature.
    n_sub points per bin — machine-precision accurate, ~90x faster than scipy.quad
    '''

    # Prep
    #binsize = binlow[1]-binlow[0]   # [eV], dim(1,)
    #binhigh = binlow + binsize      # [eV], dim(ydata,)
    #bincent = binlow + binsize/2    # [eV], dim(ydata,)
    binsize = binhigh - binlow      # [AA], dim(ydata,)
    bincent = (binhigh+ binlow)/2   # [AA], dim(ydata,)

    # Init quadrature sum
    n_sub = 8
    nodes, weights = np.polynomial.legendre.leggauss(n_sub) # dim(n_sub,)
    x_pts = (
        bincent[:,None]
        + (binsize[:,None]/2 *nodes[None,:])
        )   # [eV], dim(ydata, n_sub)

    # Calculation
    zz = (
        ((x_pts - mu) + 1j* gamma)
        / (sigma *np.sqrt(2))
        )  # [eV], dim(ydata, n_sub) 
    vals = (
        AA
        * np.real(wofz(zz))
        / (sigma * np.sqrt(2 *np.pi))
        )   # [counts], dim(ydata, n_sub)
    return (
        np.dot(vals, weights)
        * binsize/2
        )   # [counts], dim(ydata,)

# Calculates Voigt FWHM
def _voigt_fwhm(
    sigma,          # [eV], dim(1,)
    gamma,          # [eV], dim(1,)
    dsigma = None,  # [eV], dim(1,)
    dgamma = None,  # [eV], dim(1,)
    ):
    '''
    Olivero & Longbothum / Kielkopf approximation
    Voigt FWHM accurate to 0.02%
    https://en.wikipedia.org/wiki/Voigt_profile

    Optionally propagate uncertainties
    '''

    # Calculates
    fG = 2.3548 *sigma      # Gaussian FWHM, [eV], dim(1,)
    fL = 2.0    *gamma      # Lorentzian FWHM, [eV], dim(1,)
    fV = (
        0.5346 *fL
        + np.sqrt(
            0.2169 *fL**2
            + fG**2
            )
        )   # Voigt FWHM, [eV], dim(1,)
    
    # Output, [eV], dim(1,)
    if dsigma is None or dgamma is None:
        return fV, 0.0

    # Propogation of errors
    denom = np.sqrt(0.2169 *fL**2 +fG**2)
    dV_dG = fG / denom
    dV_dL = 0.5346 + (0.2166 *fL) /denom
    dV_dsig = dV_dG *2.3548
    dV_dgam = dV_dL *2.0

    dV = np.sqrt(
        (dV_dsig *dsigma)**2 + (dV_dgam *dgamma)**2
        )   # [eV], dim(1,)

    # Output, [eV], dim(1,)
    return fV, dV


####################################################################
#
#                   Other
#
####################################################################

# Function to convert residual array to scalar
# Kahan summation algorithm
# https://en.wikipedia.org/wiki/Kahan_summation_algorithm
# Reduces numerical error in summing finite-precision floating-point numbers
def _kahanSum(
    residuals      # Array of residuals, dim(ydata,)
    ):

    # Prepare the accumulator.
    ksum = 0.0

    # A running compensation for lost low-order bits.
    cc = 0.0

    # The array input has elements indexed input[1] to input[input.length].
    for rr in residuals:

        # c is zero the first time around.
        yy = rr - cc

        # Alas, sum is big, y small, so low-order digits of y are lost.
        tt = ksum + yy

        # Algebraically, c is always 0
        # when t is replaced by its
        # value from the above expression.
        # But, when there is a loss,
        # the higher-order y is cancelled
        # out by subtracting y from c and
        # all that remains is the
        # lower-order error in c

        # (t - sum) cancels the high-order part of y;
        # subtracting y recovers negative (low part of y)
        cc = (tt - ksum) - yy

        # Algebraically, c should always be zero. Beware
        # overly-aggressive optimizing compilers!
        ksum = tt
        # Next time around, the lost low part will be added to y in a fresh attempt.

    # Output, dim(scalar)
    return ksum