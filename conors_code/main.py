'''

Main script to facilitate scpectral fitting using lmfit
Adapted from A. Fairchild's (LLNL) code, May 22nd, 2026
https://lmfit.github.io/lmfit-py/

NOTE: This should not be very problem specific

cjperks
June 1st, 2026

'''

# Modules
import sys, os
import numpy as np
from lmfit import (
    #Model, 
    fit_report, 
    conf_interval,
    Minimizer, 
    printfuncs, 
    #Parameters      # Dictionary of `Parameter` objects for free variables
    )

fol = os.path.join(
    '/home/cjperks/work',
    'thesis',
    'SPARC_XRS_FDR/fitting'
    )
sys.path.insert(0, fol)
from _utils import *
from _def_models import *
from _uncert_quant import *
sys.path.pop(0)

__all__ = [
    'fit_main',
    '_get_yfit_comps',
    '_cash_scalar_fn',
    ]


####################################################################
#
#                   Main function
#
####################################################################

# Performs spectral fit
def fit_main(
    binlow, binhigh, ydata,          # Data to be fit
    case = 'gauss_int',     # Model case of interest
    stat = 'Cash',          # Statistics to minimize
    diag = 'XRSHRKr',       # Diagnostic of interest
    ):
    '''
    Performs minimization twice

    If chi2 then Nelder --> leastsq
    If Cash then Nelder --> L-BFGS-B

    NOTE: Returns "minimizer2" since lmfit.conf_interval (F-test) needs
    Minimizer object later to refit model at each profile grid point

    '''

    # Gets fit parameters for this case
    params = _params(case=case, diag=diag)

    # Dictionary of Minimizer keywords
    kw = {}
    kw['fcn_args'] = (
        ydata, binlow, binhigh, 
        case, stat, 1
        )   # Positional arguments to pass to userfcn
    if stat == 'Cash':
        kw['reduce_fcn'] = _kahanSum    # Function to reduce residuals to scalar, i.e, sum of squares

    # First minimization
    minimizer1 = Minimizer(
        _wrapper,              # userfcn, Objective function returning residuals
        params,                     # Dictionary of free variables with constraints
        **kw
        )
    result1 = minimizer1.minimize(
        method='nelder'        # Use a non-gradient descent method
        )                      # NOTE: minizes the scalar sum-of-squares

    # Dictionary of Minimizer keywords
    kw = {}
    kw['fcn_args'] = (
        ydata, binlow, binhigh,
        case, stat, 2
        )   # Positional arguments to pass to userfcn
    if stat == 'Cash':
        kw['reduce_fcn'] = _kahanSum    # Function to reduce residuals to scalar, i.e, sum of squares

    # Second minimization
    minimizer2 = Minimizer(
        _wrapper,              # userfcn, Objective function returning residuals
        result1.params,        # Dictionary of free variables with constraints
        **kw
        )
    result2 = minimizer2.minimize(
        method='lbfgsb' if stat == 'Cash' else 'leastsq'   # Use a non-gradient descent method if Cash
        )

    # Output data
    if stat == 'chi2':
        dof = result2.nfree
        scalar = result2.chisqr

    elif stat == 'Cash':
        dof = (
            len(ydata)
            - sum(p.vary for p in result2.params.values())
            )
        #yfit = _eval_model(binlow, binhigh, result2.params, case=case)
        #res = _cstat(ydata, yfit)
        #scalar = _kahanSum(res)
        scalar = _cash_scalar_fn(
            result2.params, 
            binlow, binhigh, ydata, case
            )


    ### --- Calculate symmetric error bars --- ###
    sym_err = hessian_stderr(
        result2, _cash_scalar_fn,
        (binlow, binhigh, ydata, case)
        )
    #print(sym_err)

    # Report fit quality
    if False:
        print(fit_report(result))

    # Confidence intervals
    #try:
    #    ci = conf_interval(
    #        minimizer2, result2, 
    #        maxiter=1e4, min_rel_change=1e-10
    #        )
    #    printfuncs.report_ci(ci)
    #except:
    #    print('Could not calculate doubled-sided error bars')

    # Output
    return (
        minimizer2, result2,
        scalar, dof, sym_err
        )


####################################################################
#
#                   Utilities
#
####################################################################

# Evaluates the Cash scalar
def _cash_scalar_fn(
    params,
    binlow,
    binhigh,
    ydata,
    case,
    ):

    yfit = _eval_model(binlow, binhigh, params, case=case)
    res = _cstat(ydata, yfit)
    return _kahanSum(res)

# Objective function to minimize
def _wrapper(
    params,     # Parameters dictionary defining model, [dictionary]
    ydata,      # Measured counts, [counts], dim(ydata,)
    binlow,     # Measurement associated x-axis, [eV], dim(ydata,)
    binhigh,
    case,       # Model functions, [scalar]
    stat,       # Statistic to minimize, [scalar]
    num,        # Iteration number
    ):
    '''
    NOTE: Why does chi2 use two minimizers?
    1) Nelder-Mead is gradient-free and explores broadly
        --> minimizes the scalar sum-of-squares
        --> Finds the right basin even from a poor starting point

    2) leastsq need the residual array (not a scalar) to
        build Jacobian for covariance estimation
        --> converges more precisely than Nelder-Mead
        --> Starting from Nelder result means it's already in right basin

    '''

    # Evaulate model with parameters
    model_value = _eval_model(binlow, binhigh, params, case)

    # Output Cash residuals of model v. data, dim(ydata,)
    if stat == 'Cash':
        return _cstat(ydata, model_value)
    elif stat == 'chi2' and num == 1:
        return _chi2stat_squared(ydata, model_value)
    elif stat == 'chi2' and num == 2:
        return _chi2stat(ydata, model_value)


# Get fit output with components
def _get_yfit_comps(
    binlow = None,
    binhigh = None,
    dresult = None,
    case = None,
    ):

    # Gets fit result
    yfit = _eval_model(binlow, binhigh, dresult['result'].params, case=case)

    ycomp = {}
    # Loop over parameters
    for kk in dresult['result'].params.keys():
        # Naming scheme, 'kk1_kk2'
        kk1 = kk.split('_')[0]
        kk2 = kk.split('_')[-1] 

        # Skip redundant keys
        if kk1 in ['mu', 'sigma']:
            continue

        # Add linear background
        elif kk1 == 'bck':
            ycomp[kk1] = _bck_int(
                binlow, binhigh,
                dresult['result'].params[kk1].value,
                )   # [counts], dim(ydata,)

        # Adds spectral feature
        elif kk1 == 'area':
            if case == 'gauss_int':
                ycomp[kk2] = _gauss_int(
                    binlow, binhigh,
                    dresult['result'].params['area_'+kk2].value,
                    dresult['result'].params['mu_'+kk2].value,
                    dresult['result'].params['sigma_'+kk2].value
                    )   # [counts], dim(ydata,)
            elif case == 'gauss_pt':
                ycomp[kk2] = _gauss_pt(
                    binlow, binhigh,
                    dresult['result'].params['area_'+kk2].value,
                    dresult['result'].params['mu_'+kk2].value,
                    dresult['result'].params['sigma_'+kk2].value
                    )   # [counts], dim(ydata,)
            elif case == 'voigt_int':
                ycomp[kk2] = _voigt_int(
                    binlow, binhigh,
                    dresult['result'].params['area_'+kk2].value,
                    dresult['result'].params['mu_'+kk2].value,
                    dresult['result'].params['sigma_'+kk2].value,
                    dresult['result'].params['gamma_'+kk2].value,
                    )   # [counts], dim(ydata,)
            elif case == 'voigt_pt':
                ycomp[kk2] = _voigt_pt(
                    binlow, binhigh,
                    dresult['result'].params['area_'+kk2].value,
                    dresult['result'].params['mu_'+kk2].value,
                    dresult['result'].params['sigma_'+kk2].value,
                    dresult['result'].params['gamma_'+kk2].value,
                    )   # [counts], dim(ydata,)

    # Output, [counts], dim(ydata,)
    return yfit, ycomp