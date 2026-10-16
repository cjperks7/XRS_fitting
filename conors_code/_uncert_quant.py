'''

Functions for uncertainty quantification

cjperks
June 1st, 2026

'''

# Modules
import numpy as np

__all__ = [
    'hessian_stderr',
    ]

#############################################
#
#       Symmetric errors
#
############################################

def hessian_stderr(
    result, 
    stat_scalar_fn, 
    fcn_args, 
    eps_frac=1e-4
    ):
    '''
    Diagonal Hessian stderr from 
    scalar stat surface (for grid sizing).

    '''

    p0 = result.params.copy()
    S0 = stat_scalar_fn(p0, *fcn_args)
    se = {}
    for name in [n for n in p0 if p0[n].vary]:
        v = p0[name].value
        h = max(abs(v)*eps_frac, 1e-4)
        pp = p0.copy(); pp[name].value = v + h
        pm = p0.copy(); pm[name].value = v - h
        d2 = (stat_scalar_fn(pp, *fcn_args) - 2*S0 +
              stat_scalar_fn(pm, *fcn_args)) / h**2
        se[name] = np.sqrt(2.0/d2) if d2 > 1e-12 else abs(v)*0.05
    return se