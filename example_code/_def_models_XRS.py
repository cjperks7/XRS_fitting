'''

Function to store modeling options

NOTE: This should be very problem specific

cjperks
June 1st, 2026

'''

# Modules
import sys, os
import numpy as np
from lmfit import (
    #Model, 
    #report_fit, 
    #conf_interval,
    #Minimizer, 
    #printfuncs, 
    Parameters      # Dictionary of `Parameter` objects for free variables
    )

fol = os.path.join(
    '/home/cjperks/work',
    'thesis',
    'SPARC_XRS_FDR/fitting'
    )
sys.path.insert(0, fol)
from _utils import *
sys.path.pop(0)

__all__ = [
    # Evaulates model
    '_eval_model',
    # Model parameters
    '_params',
    # Model functions
    '_model_gauss_int',
    '_model_gauss_pt',
    ]

####################################################################
#
#                   Evaulates model
#
####################################################################

# Wrapper for evaulating model
def _eval_model(
    binlow,         # Left side of binned x-scale, [eV], dim(ydata,)
    binhigh,
    params,
    case=None,      # Model of interest
    ):

    # Fit model for bin-integrated Gaussians
    if case=='gauss_int':
        return _model_gauss_int(binlow, binhigh, params) # [counts], dim(ydata,)
    # Fit model for point-evaulated Gaussians
    elif case == 'gauss_pt':
        return _model_gauss_pt(binlow, binhigh, params)  # [counts], dim(ydata,)
    # Fit model for bin-integrated Voigts
    elif case=='voigt_int':
        return _model_voigt_int(binlow, binhigh, params) # [counts], dim(ydata,)
    # Fit model for point-evaulated Voigts
    elif case == 'voigt_pt':
        return _model_voigt_pt(binlow, binhigh, params)  # [counts], dim(ydata,)

####################################################################
#
#                   Model Parameters
#
####################################################################

# Model free parameter for problem of interest
def _params(
    case = None,        # Model of interest
    diag = None,
    ):
    '''
    # NOTE: Here we define our problem to have 5 spectral lines
    with a linear background
    '''

    # Init
    params = Parameters()

    # Default values
    ddefs = {
        'bck': {'val': 21.0, 'min': 1e-1, 'max': 50.0},   # [counts/bin]
        'G1_XRSHRKr': {     # feature #1 for He-like Kr
            #'area': {'val': 1e4, 'min': 1e3, 'max': 2e4},    # [counts]
            #'mu': {'val': 0.94501, 'min': 0.9449, 'max': 0.9452},    # [AA]
            #'sigma': {'val': 1.9e-4, 'min': 1.8e-4, 'max': 6.0e-4},    # [AA]
            'area': {'val': 1e4, 'min': 1e3, 'max': 2e4},    # [counts]
            'mu': {'val': 0.94535, 'min': 0.945, 'max': 0.9455},    # [AA]
            'sigma': {'val': 4.0e-4, 'min': 2.0e-4, 'max': 6.0e-4},    # [AA]
            'gamma': {'val': 1e-4, 'min': 1e-5, 'max': 5e-4},   # [AA]
            },
        'G1_XRSHRXe': {     # feature #1 for Ne-like Xe
            #'area': {'val': 1e3, 'min': 1e3, 'max': 2e4},    # [counts]
            ##'mu': {'val': 2.7201, 'min': 2.719, 'max': 2.722},    # [AA]
            #'mu': {'val': 2.7401, 'min': 2.739, 'max': 2.742},    # [AA]
            #'sigma': {'val': 1.9e-4, 'min': 1e-4, 'max': 6.0e-4},    # [AA]
            'area': {'val': 1e3, 'min': 1e3, 'max': 2e4},    # [counts]
            'mu': {'val': 2.72020, 'min': 2.715, 'max': 2.725},    # [AA]
            'sigma': {'val': 6.0e-4, 'min': 1.0e-4, 'max': 9.0e-4},    # [AA]
            'gamma': {'val': 1e-4, 'min': 1e-5, 'max': 5e-4},   # [AA]
            },
        }

    # Loop over features
    for kk in ddefs.keys():
        # Background
        if kk == 'bck':
            params.add(
                kk,
                value = ddefs[kk]['val'],
                min = ddefs[kk]['min'],
                max = ddefs[kk]['max']
                )

        # Spectral features
        elif diag in kk:
            # Loop over feature variables
            for varb in ddefs[kk].keys():
                # Handles extra free parameter for Voigt model
                if varb == 'gamma' and 'voigt' not in case:
                    continue

                params.add(
                    '%s_%s'%(varb, kk.split('_')[0]),
                    value = ddefs[kk][varb]['val'],
                    min = ddefs[kk][varb]['min'],
                    max = ddefs[kk][varb]['max'],
                    )

            # Fixed Gaussian FWHM
            if 'sigma' not in ddefs[kk].keys():
                params.add(
                    'sigma_%s'%(kk.split('_')[0]),
                    expr = 'sigma_G1'
                    )

            # Fixed Lorentzian FWHM
            if 'gamma' not in ddefs[kk].keys() and 'voigt' in case:
                params.add(
                    'gamma_%s'%(kk.split('_')[0]),
                    expr = 'gamma_G1'
                    )

    # Output
    return params

####################################################################
#
#                   Model functions
#
####################################################################

# Defines fit model for bin-integrated Guassians
def _model_gauss_int(
    binlow,         # Left side of binned x-scale, [eV], dim(ydata,)
    binhigh,
    params,
    ):

    # Init ouput
    out = np.zeros_like(binlow, dtype=float) # [counts], dim(ydata,)

    # Loop over parameters
    for kk in params.keys():
        # Naming scheme, 'kk1_kk2'
        kk1 = kk.split('_')[0]
        kk2 = kk.split('_')[-1] 

        # Skip redundant keys
        if kk1 in ['mu', 'sigma']:
            continue

        # Add linear background
        elif kk1 == 'bck':
            out += _bck_int(
                binlow, binhigh,
                params[kk1].value,
                )

        # Add binned Gaussian
        elif kk1 == 'area':
            out += _gauss_int(
                binlow, binhigh,
                params['area_'+kk2].value,
                params['mu_'+kk2].value,
                params['sigma_'+kk2].value
                )

    # Output, dim(ydata,)
    return out

# Defines fit model for point-evaluated Guassian
def _model_gauss_pt(
    binlow,         # Left side of binned x-scale, [eV], dim(ydata,)
    binhigh,
    params,
    ):

    # Init ouput
    out = np.zeros_like(binlow, dtype=float) # [counts], dim(ydata,)

    # Loop over parameters
    for kk in params.keys():
        # Naming scheme, 'kk1_kk2'
        kk1 = kk.split('_')[0]
        kk2 = kk.split('_')[-1] 

        # Skip redundant keys
        if kk1 in ['mu', 'sigma']:
            continue

        # Add linear background
        elif kk1 == 'bck':
            out += _bck_int(
                binlow, binhigh,
                params[kk1].value,
                )

        # Add binned Gaussian
        elif kk1 == 'area':
            out += _gauss_pt(
                binlow, binhigh,
                params['area_'+kk2].value,
                params['mu_'+kk2].value,
                params['sigma_'+kk2].value
                )

    # Output, dim(ydata,)
    return out


# Defines fit model for bin-integrated Voigts
def _model_voigt_int(
    binlow,         # Left side of binned x-scale, [eV], dim(ydata,)
    binhigh,
    params,
    ):

    # Init ouput
    out = np.zeros_like(binlow, dtype=float) # [counts], dim(ydata,)

    # Loop over parameters
    for kk in params.keys():
        # Naming scheme, 'kk1_kk2'
        kk1 = kk.split('_')[0]
        kk2 = kk.split('_')[-1] 

        # Skip redundant keys
        if kk1 in ['mu', 'sigma', 'gamma']:
            continue

        # Add linear background
        elif kk1 == 'bck':
            out += _bck_int(
                binlow, binhigh,
                params[kk1].value,
                )

        # Add binned Voigts
        elif kk1 == 'area':
            out += _voigt_int(
                binlow, binhigh,
                params['area_'+kk2].value,
                params['mu_'+kk2].value,
                params['sigma_'+kk2].value,
                params['gamma_'+kk2].value
                )

    # Output, dim(ydata,)
    return out

# Defines fit model for point-evaluated Voigt
def _model_voigt_pt(
    binlow,         # Left side of binned x-scale, [eV], dim(ydata,)
    binhigh,
    params,
    ):

    # Init ouput
    out = np.zeros_like(binlow, dtype=float) # [counts], dim(ydata,)

    # Loop over parameters
    for kk in params.keys():
        # Naming scheme, 'kk1_kk2'
        kk1 = kk.split('_')[0]
        kk2 = kk.split('_')[-1] 

        # Skip redundant keys
        if kk1 in ['mu', 'sigma', 'gamma']:
            continue

        # Add linear background
        elif kk1 == 'bck':
            out += _bck_int(
                binlow, binhigh,
                params[kk1].value,
                )

        # Add binned Voigts
        elif kk1 == 'area':
            out += _voigt_pt(
                binlow, binhigh,
                params['area_'+kk2].value,
                params['mu_'+kk2].value,
                params['sigma_'+kk2].value,
                params['gamma_'+kk2].value,
                )

    # Output, dim(ydata,)
    return out