'''

Common functions to do plotting

cjperks
June 1st, 2026

'''

# Modules
import numpy as np
import sys, os
import matplotlib.pyplot as plt

fol = os.path.join(
    '/home/cjperks',
    'sharing/to_alexFAIRCHILD',
    '260601_fit_uncert_v2/funcs',
    )
sys.path.insert(0, fol)
from _utils import *
from _def_models import *
from main import *
sys.path.pop(0)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H']
lines = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']


__all__ = [
    '_plot_fit'
    ]

##################################################################
#
#           Fit results
#
##################################################################

# PLots fit results
def _plot_fit(
    # Figure
    ax_fit = None,
    ax_res = None,
    do_xlab = False,
    # Data
    binlow = None,
    ydata = None,
    # Fit
    dresult = None,
    # Control
    stat = None,
    case = None,
    ):

    # Plots raw data
    yerr = np.sqrt(np.where(ydata <= 0, 1, ydata))
    xerr = np.mean(np.diff(binlow))/2    # [eV]
    bincent = binlow+xerr
    ax_fit.errorbar(
        bincent, 
        ydata, 
        yerr=yerr, 
        xerr=xerr,
        fmt='.', 
        color='k', 
        ms=3, 
        elinewidth=0.7,
        label = 'Data'
        )

    # Gets fit result
    yfit, ycomp = _get_yfit_comps(
        binlow = binlow,
        dresult = dresult,
        case = case
        )

    # Plots total fit result
    ax_fit.step(
        binlow,
        yfit,
        where='post',
        color ='r',
        linewidth = 2,
        zorder = 10,
        label = 'Total fit'
        )

    # Plots fit components
    for ii, kk in enumerate(ycomp.keys()):
        if kk == 'bck':
            lsty = ':'
        else:
            lsty = '--'

        ax_fit.step(
            binlow,
            ycomp[kk],
            where='post',
            color =colors[ii],
            linewidth = 2,
            zorder = 10,
            label = kk,
            linestyle = lsty
            )

    # Plots residuals
    if stat == 'chi2':
        err = np.sqrt(np.where(ydata<=0, 1, ydata))
        res = (ydata - yfit)/err
        rlab = r'$(d-m)/\sigma$'
    elif stat == 'Cash':
        res = _cstat_signed(ydata, yfit)
        rlab = r'$\pm\sqrt{|C_i|}$'

    ax_res.errorbar(
        bincent,
        res,
        yerr=1,
        fmt = '.',
        color = 'k',
        ms = 2.5,
        elinewidth = 0.5,
        )
    ax_res.axhline(0, color='r', lw=1)
    ax_res.axhline(2, color='r', lw=1, ls=':', alpha = 0.5)
    ax_res.axhline(-2, color='r', lw=1, ls=':', alpha =0.5)

    # Labeling
    if case.split('_')[-1] == 'int':
        lcase = 'bin-integrated'
    elif case.split('_')[-1] == 'pt':
        lcase = 'point-evaluated'

    if stat == 'Cash':
        lstat = 'Cash/%s, C/dof=%0.3f, FWHM=%0.3f eV'%(
            lcase, dresult['scalar']/dresult['dof'], 
            dresult['result'].params['sigma_F1'].value*2.35482
            )
    elif stat == 'chi2':
        lstat = 'chi2/%s, chi2/dof=%0.3f, FWHM=%0.3f eV'%(
            lcase, dresult['scalar']/dresult['dof'], 
            dresult['result'].params['sigma_F1'].value*2.35482
            )

    # Formatting
    ax_fit.grid('on')
    ax_fit.set_xlim(min(binlow), max(binlow)+xerr*2)
    ax_fit.set_xticklabels([])
    ax_fit.set_ylim(0, 1.1*max(ydata+yerr))

    ax_fit.set_ylabel('Counts/bin')

    ax_fit.set_title(lstat, loc='left')

    ax_fit.legend(
        #fontsize=8, 
        frameon=False, 
        ncol=4,
        bbox_to_anchor=(1.0, 1.0), 
        loc='upper right'
        )

    ax_res.grid('on')
    ax_res.set_xlim(min(binlow), max(binlow)+xerr*2)

    ax_res.set_ylabel(rlab)
    ax_res.set_ylim(-4.5, 4.5)

    if do_xlab:
        ax_res.set_xlabel('Photon energy [eV]')

