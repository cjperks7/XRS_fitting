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
    '/home/cjperks',
    'sharing/to_alexFAIRCHILD',
    '260601_fit_uncert_v2/funcs',
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
    params,
    case=None,      # Model of interest
    ):

    # Fit model for bin-integrated Gaussians
    if case=='gauss_int':
        return _model_gauss_int(binlow, params) # [counts], dim(ydata,)
    # Fit model for point-evaulated Gaussians
    elif case == 'gauss_pt':
        return _model_gauss_pt(binlow, params)  # [counts], dim(ydata,)
    # Fit model for bin-integrated Voigts
    elif case=='voigt_int':
        return _model_voigt_int(binlow, params) # [counts], dim(ydata,)
    # Fit model for point-evaulated Voigts
    elif case == 'voigt_pt':
        return _model_voigt_pt(binlow, params)  # [counts], dim(ydata,)

####################################################################
#
#                   Model Parameters
#
####################################################################

# Model free parameter for problem of interest
def _params(
    case = None,        # Model of interest
    ):
    '''
    # NOTE: Here we define our problem to have 5 spectral lines
    with a linear background
    '''

    # Init
    params = Parameters()

    # Default values
    ddefs = {
        'bck': {'val': 21.0, 'min': 1.0, 'max': 50.0},   # [counts/bin]
        'G1': {     # feature #1
            'area': {'val': 350.0, 'min': 0.0, 'max': 2000.0},    # [counts]
            'mu': {'val': 4558.5, 'min': 4554.0, 'max': 4563.0},    # [eV]
            'sigma': {'val': 1.5, 'min': 0.3, 'max': 5.0},    # [eV]
            'gamma': {'val': 0.3, 'min': 1e-4, 'max': 3.0},   # [eV]
            },
        'G2': {     # feature #2
            'area': {'val': 150.0, 'min': 0.0, 'max': 2000.0},    # [counts]
            'mu': {'val': 4548.0, 'min': 4544.0, 'max': 4552.0},    # [eV]
            },
        'G3': {     # feature #3
            'area': {'val': 200.0, 'min': 0.0, 'max': 2000.0},    # [counts]
            'mu': {'val': 4532.0, 'min': 4528.0, 'max': 4536.0},    # [eV]
            },
        'G4': {     # feature #4
            'area': {'val': 120.0, 'min': 0.0, 'max': 1500.0},    # [counts]
            'mu': {'val': 4494.0, 'min': 4490.0, 'max': 4498.0},    # [eV]
            },
        'G5': {     # feature #5
            'area': {'val': 600.0, 'min': 10.0, 'max': 5000.0},    # [counts]
            'mu': {'val': 4482.5, 'min': 4479.0, 'max': 4486.0},    # [eV]
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
        else:
            # Loop over feature variables
            for varb in ddefs[kk].keys():
                # Handles extra free parameter for Voigt model
                if varb == 'gamma' and 'voigt' not in case:
                    continue

                params.add(
                    '%s_%s'%(varb, kk),
                    value = ddefs[kk][varb]['val'],
                    min = ddefs[kk][varb]['min'],
                    max = ddefs[kk][varb]['max'],
                    )

            # Fixed Gaussian FWHM
            if 'sigma' not in ddefs[kk].keys():
                params.add(
                    'sigma_%s'%(kk),
                    expr = 'sigma_G1'
                    )

            # Fixed Lorentzian FWHM
            if 'gamma' not in ddefs[kk].keys() and 'voigt' in case:
                params.add(
                    'gamma_%s'%(kk),
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
                binlow,
                params[kk1].value,
                )

        # Add binned Gaussian
        elif kk1 == 'area':
            out += _gauss_int(
                binlow,
                params['area_'+kk2].value,
                params['mu_'+kk2].value,
                params['sigma_'+kk2].value
                )

    # Output, dim(ydata,)
    return out

# Defines fit model for point-evaluated Guassian
def _model_gauss_pt(
    binlow,         # Left side of binned x-scale, [eV], dim(ydata,)
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
                binlow,
                params[kk1].value,
                )

        # Add binned Gaussian
        elif kk1 == 'area':
            out += _gauss_pt(
                binlow,
                params['area_'+kk2].value,
                params['mu_'+kk2].value,
                params['sigma_'+kk2].value
                )

    # Output, dim(ydata,)
    return out


# Defines fit model for bin-integrated Voigts
def _model_voigt_int(
    binlow,         # Left side of binned x-scale, [eV], dim(ydata,)
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
                binlow,
                params[kk1].value,
                )

        # Add binned Voigts
        elif kk1 == 'area':
            out += _voigt_int(
                binlow,
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
                binlow,
                params[kk1].value,
                )

        # Add binned Voigts
        elif kk1 == 'area':
            out += _voigts_pt(
                binlow,
                params['area_'+kk2].value,
                params['mu_'+kk2].value,
                params['sigma_'+kk2].value,
                params['gamma_'+kk2].value,
                )

    # Output, dim(ydata,)
    return out