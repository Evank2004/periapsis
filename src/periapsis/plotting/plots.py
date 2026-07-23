import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from periapsis.fitting.results import FitResults, SampledPriors
from periapsis.model import thieleinnes
from periapsis.model import campbell
from periapsis.model.orbit import Orbit
from periapsis.utils.solvers import solve_kepler
import matplotlib.gridspec as gridspec
from periapsis.prior import FixedPrior
from periapsis.utils.solvers import solve_mass
from periapsis.utils.solvers import campbell_to_thiele
from periapsis.utils.solvers import transform_theile
from periapsis.utils.helpers import _build_model
from periapsis.data.gaia import GaiaData
from scipy.stats import gaussian_kde

rng_plots = np.random.default_rng(5377)

def mcmc_autocorrelation_plot(results,savepath=None):
    '''
    Plots the autocorrelation function for each parameter
    This can be used to diagnose convergence and mixing of MCMC chain
    '''
   
    param_means = np.asarray(results.samples['param_means'])
    param_names = results.param_names
    if results.fit_method =='linear':
        param_names = [name for name in param_names if name in ('P', 'e', 't0')]
        name_to_idx = {name: i for i, name in enumerate(results.param_names)}
        param_means = param_means[:, [name_to_idx[name] for name in param_names]]

    lags = np.arange(param_means.shape[0])
    autocorrs = {}
    for i, name in enumerate(param_names):
        acorrs = emcee.autocorr.function_1d(param_means[:, i])
        autocorrs[name]= acorrs
        
    fig,ax = plt.subplots()
    #TODO: make labels pretty and not just param names
    for name, acorrs in autocorrs.items():
        ax.plot(lags, acorrs, label=name)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Autocorrelation')
    ax.legend(loc='best',ncol=3,fontsize='small')
    if savepath is not None:
        fig.savefig(savepath,dpi=300)
        print(f"Saved autocorrelation plot to {savepath}")

    return fig

def corner_plot(results,params=None,savepath=None):
    '''
    Plots the corner plot for sampled parameters 
    '''
    
    param_names = results.param_names if params is None else params

    samples = np.array([results[name] for name in param_names]).T

    fig = corner.corner(samples,quantiles=[0.16,0.5,0.84],
        color='tab:blue',labels=param_names,show_titles=True,verbose=False,
        title_fmt='.2f',plot_datapoints=False,plot_contours=True,fill_contours=True,quiet=True)
    if savepath is not None:
        fig.savefig(savepath,dpi=300)
        print(f"Saved corner plot to {savepath}")
    return fig


def ess_distribution_plot(results,savepath=None):
    '''
    Plots distribution of effective sample size (ESS) for each parameter.
    This can be used to diagnose convergence
    '''
    ess = results.samples['Ess']
    param_names = results.param_names

    ess_values = np.atleast_1d(ess)  # Ensure ESS is an array
    ndim = len(ess_values)
    labels = param_names[:ndim]

    tick = np.arange(ndim)

    fig, ax = plt.subplots()
    
    ax.bar(tick, ess_values,color='tab:blue', alpha=0.4,align='center')
    ax.set_xticks(tick)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Effective Sample Size (ESS)')
    # ax.axhline(1000, color='r', linestyle='--', label='ESS=1000')
    ax.legend()
    if savepath is not None:
        fig.savefig(savepath,bbox_inches= 'tight',dpi=300)
        print(f"Saved ESS distribution plot to {savepath}")
    return fig


def prior_dist_plot(sampled_priors: SampledPriors, params=None, savepath=None, bins=100, ncols=None):
    '''
    Plots the prior distribution for each parameter.
    This can be used to diagnose convergence and mixing of MCMC chain
    '''

    param_names = sampled_priors.param_order if params is None else params

    # Initialize Axes
    ncols = max(1, min(int(np.ceil(np.sqrt(len(param_names)))), len(param_names))) if ncols is None else ncols
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
    axes = axes.ravel()

    for i, name in enumerate(param_names):
        ax = axes[i]
        prior_samples = sampled_priors[name]
        if prior_samples is not None and len(prior_samples) > 0:
            ax.hist(
                prior_samples,
                bins=bins,
                density=True,
                histtype='step',
                color='gray',
                linewidth=1.5,
                label='Prior',
            )
            ax.set_ylabel('Probability Density')
            ax.set_xlabel(f'{name} Value')
        else:
            ax.set_axis_off()

    for j in range(len(param_names), len(axes)):
        axes[j].axis('off')

    if savepath is not None:
        fig.savefig(savepath,dpi=300)
        print(f"Saved prior distribution plot to {savepath}")
    
    return fig

def prior_histogram_2d(sampled_priors: SampledPriors, param_x, param_y, savepath=None, bins=100):
    '''
    Plots the 2D histogram of the prior distribution for two parameters.
    This can be used to diagnose convergence and mixing of MCMC chain
    '''

    prior_samples_x = sampled_priors[param_x]
    prior_samples_y = sampled_priors[param_y]

    if prior_samples_x is None or prior_samples_y is None:
        raise ValueError(f"Prior samples for {param_x} or {param_y} are not available.")

    fig, ax = plt.subplots()
    h = ax.hist2d(prior_samples_x, prior_samples_y, bins=bins, density=True, cmap='Blues')
    plt.colorbar(h[3], ax=ax)
    ax.set_xlabel(f'{param_x} Value')
    ax.set_ylabel(f'{param_y} Value')
    ax.set_title(f'2D Prior Histogram: {param_x} vs {param_y}')

    if savepath is not None:
        fig.savefig(savepath,dpi=300)
        print(f"Saved 2D prior histogram plot to {savepath}")

    return fig

def prior_conditional_histogram_2d(sampled_priors: SampledPriors, param_fixed, param_other, bins=100, savepath=None):
    '''
    For bins of param_fixed, plots the conditional histogram of param_other on a 2d histogram
    '''

    prior_samples_fixed = sampled_priors[param_fixed]
    prior_samples_other = sampled_priors[param_other]

    fixed_bins = np.linspace(np.min(prior_samples_fixed), np.max(prior_samples_fixed), bins + 1)
    results = np.zeros((bins, bins))
    for i in range(len(fixed_bins) - 1):
        bin_mask = (prior_samples_fixed >= fixed_bins[i]) & (prior_samples_fixed < fixed_bins[i + 1])
        conditional_samples = prior_samples_other[bin_mask]

        if len(conditional_samples) > 0:
            hist, _ = np.histogram(conditional_samples, bins=bins, density=True)
            results[i, :] = hist
    
    fig, ax = plt.subplots()
    extent = [np.min(prior_samples_other), np.max(prior_samples_other), np.min(prior_samples_fixed), np.max(prior_samples_fixed)]
    im = ax.imshow(results, aspect='auto', origin='lower', extent=extent, cmap='Blues', norm='log')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(f'{param_other} Value')
    ax.set_ylabel(f'{param_fixed} Value')
    ax.set_title(f'Conditional 2D Histogram: {param_other} vs {param_fixed}')

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
        print(f"Saved conditional 2D histogram plot to {savepath}")

    return fig


def posterior_over_prior(results: FitResults, params=None, savepath=None, random_state=np.random.default_rng(), ncols=2, bins=100):
    '''
    Plots the posterior distribution over the prior distribution for each parameter.
    This can be used to diagnose convergence and mixing of MCMC chain
    '''

    param_names = results.param_names if params is None else params

    # Get Samples 
    sample_arrays = []
    for name in param_names:
        value = results[name]
        if isinstance(value, np.ndarray):
            sample_arrays.append(np.asarray(value, dtype=float).ravel())
        else:
            sample_arrays.append(np.asarray(value, dtype=float).reshape(-1))
    samples = np.array(sample_arrays, dtype=float).T

    # Initialize Axes
    ncols = max(1, min(int(ncols), len(param_names)))
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 2.8 * nrows), squeeze=False, constrained_layout=True)
    axes = axes.ravel()

    priors = results.sample_priors(random_state, size=10000)

    for i, name in enumerate(param_names):
        ax = axes[i]
        prior_samples = priors[name]
        sample_values = samples[:, i]
        finite_sample_values = sample_values[np.isfinite(sample_values)]

        range_values = finite_sample_values #create range this way incase, for parameters like A,B,F,G, the samples are outside of induced prior, since they arent really being impacted by that prior
        if prior_samples is not None and len(prior_samples) > 0:
            range_values = np.concatenate([range_values, prior_samples]) if len(range_values) > 0 else prior_samples

        if len(range_values) == 0:
            ax.set_axis_off()
            continue

        range_min = np.nanmin(range_values)
        range_max = np.nanmax(range_values)
        bins_edges = np.linspace(range_min, range_max, int(bins) + 1)

        if prior_samples is not None and len(prior_samples) > 0:
            ax.hist(
                prior_samples,
                bins=bins_edges,
                density=True,
                histtype='step',
                color='gray',
                linewidth=1.5,
                label='Prior',
            )

        ax.hist(
            finite_sample_values,
            bins=bins_edges,
            density=True,
            color='tab:blue',
            alpha=0.35,
            edgecolor='tab:blue',
            label='Posterior',
        )

        ax.set_ylabel('Probability Density')
        ax.set_xlabel(f'{name} Value')
        ax.legend()

    for j in range(len(param_names), len(axes)):
        axes[j].axis('off')
    
    if savepath is not None:
        fig.savefig(savepath,dpi=300)
        print(f"Saved posterior over prior plot to {savepath}")
    
    return fig


#---------------Orbit Visualization Plots ---------------------------------

def _apply_center_offset(x, y, params, dt, center=True):
    if params is None or not center:
        return np.asarray(x), np.asarray(y)

    dx = params.get('dx', 0)
    dy = params.get('dy', 0)
    dpmra = params.get('dpmra', 0)
    dpmdec = params.get('dpmdec', 0)
    return np.asarray(x) - dx - dpmra * dt, np.asarray(y) - dy - dpmdec * dt


def orbit_plot(results, data, system=1, savepath=None):

    if isinstance(data, GaiaData):
        Map_plot_dict = data._astrometry(Orbit(**results.MAP_params))
        Med_plot_dict = data._astrometry(Orbit(**results.median_params))

        fig,ax = plt.subplots(figsize=(8,6))
        ax.plot(Map_plot_dict['ra_orb'],Map_plot_dict['dec_orb'],label='MAP Orbit',color='red',linestyle='-',zorder=1)
        ax.plot(Med_plot_dict['ra_orb'],Med_plot_dict['dec_orb'],label='Median Orbit',color='purple',linestyle='--',zorder=1)

        for ri,di,ei,si,ci in zip(Map_plot_dict['ra_obs'],Map_plot_dict['dec_obs'],data.err,data.spsi,data.cpsi):

            x0 = ri - ei * si
            x1 = ri + ei * si
            y0 = di - ei * ci
            y1 = di + ei * ci

            ax.plot([x0,x1],[y0,y1],color='tab:orange',alpha=0.5,zorder=2)

        ax.scatter(Map_plot_dict['ra_orb_obs'],Map_plot_dict['dec_orb_obs'],color='tab:blue',s=15,zorder=3)
        ax.scatter(0,0,color='k',marker='*',label = 'COM',zorder = 10)
        ax.plot([0,Map_plot_dict['ra_peri']],[0,Map_plot_dict['dec_peri']],color='gray',linestyle='--',label='Periastron',zorder=4,alpha=0.6)

        ax.set_xlabel(r"$\Delta \alpha^*$ (mas)")
        ax.set_ylabel(r"$\Delta \delta$ (mas)")
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='best')
        ax.invert_xaxis()

        if savepath is not None:
            fig.savefig(savepath, dpi=300)
            print(f"Saved orbit plot to {savepath}")
        return fig


    tfold = np.linspace(data.t.min(), data.t.max(), 1000)
    ref_epoch = getattr(data, 'ref_epoch', 0)
    dt_obs = data.t - ref_epoch
    dt_model = tfold - ref_epoch

    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)

    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)

    if map_params is None or med_params is None:
        raise ValueError("Both MAP and median parameter sets are required for orbit plotting.")
    
    for k, p in results.priors.items():
        if isinstance(p, FixedPrior):
            map_params[k] = p.value
            med_params[k] = p.value

    # map_model = _build_model(results, map_params)
    # med_model = _build_model(results, med_params)
    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    x_map_raw, y_map_raw = map_model.astrometry(tfold, system=system)
    x_med_raw, y_med_raw = med_model.astrometry(tfold, system=system)

    x_map, y_map = _apply_center_offset(x_map_raw, y_map_raw, map_params, dt_model, center=True)
    x_med, y_med = _apply_center_offset(x_med_raw, y_med_raw, med_params, dt_model, center=True)
    x_obs, y_obs = _apply_center_offset(data.x, data.y, map_params, dt_obs, center=True)

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.05)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.errorbar(dt_obs, x_obs, yerr=data.x_err, fmt='o',markersize=4,zorder=2)
    ax1.plot(dt_model, x_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
    ax1.plot(dt_model, x_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)

    ax2.errorbar(dt_obs, y_obs, yerr=data.y_err, fmt='o',markersize=4,zorder=2)
    ax2.plot(dt_model, y_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
    ax2.plot(dt_model, y_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)

    ax3.scatter(x_obs, y_obs, color='tab:blue',s=15,zorder=2)
    ax3.plot(x_map, y_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
    ax3.plot(x_med, y_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)
    ax3.set_aspect('equal',adjustable = 'datalim')
    ax3.legend()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
        print(f"Saved orbit plot to {savepath}")
    return fig

def sky_motion_plot(results, data, savepath=None):
    '''
    Plots full sky motion over time
    '''

    if isinstance(data, GaiaData):
        map_plot_dict = data._astrometry(Orbit(**results.MAP_params))
        med_plot_dict = data._astrometry(Orbit(**results.median_params))

        fig,ax = plt.subplots()

        ax.plot(map_plot_dict['ra_lin'],map_plot_dict['dec_lin'],label='MAP Linear Model',color='red',linestyle='--',zorder=1,alpha=0.7)
        ax.plot(med_plot_dict['ra_lin'],med_plot_dict['dec_lin'],label='Median Linear Model',color='purple',linestyle='--',zorder=1,alpha=0.7)
        ax.plot(map_plot_dict['ra_sky'],map_plot_dict['dec_sky'],label='MAP Sky Track',color='red',linestyle='-',zorder=1)
        ax.plot(med_plot_dict['ra_sky'],med_plot_dict['dec_sky'],label='Median Sky Track',color='purple',linestyle='-',zorder=1)
        ax.scatter(map_plot_dict['ra_sky_data'],map_plot_dict['dec_sky_data'],color='k',s=15,zorder=3)

        ax.set_xlabel(r"$\Delta \alpha^*$ (mas)")
        ax.set_ylabel(r"$\Delta \delta$ (mas)")
        ax.set_aspect('equal',adjustable='datalim')
        ax.legend(loc='best')
        ax.invert_xaxis()
        if savepath is not None:
            fig.savefig(savepath, dpi=300)
            print(f"Saved sky motion plot to {savepath}")

        return fig

    tfold = np.linspace(data.t.min(), data.t.max(), 1000)
    ref_epoch = getattr(data, 'ref_epoch', 0)
    dt = tfold - ref_epoch

    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)
    
    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)
    
    if map_params is None or med_params is None:
        raise ValueError("Both MAP and median parameter sets are required for multi-orbit plotting.")
        
    fixed_prior_params = {}
    for k, p in results.priors.items():
        if isinstance(p, FixedPrior):
            map_params[k] = p.value
            med_params[k] = p.value
            fixed_prior_params[k] = p.value

    x0 = results.PM_fit['params']['x0']
    y0 = results.PM_fit['params']['y0']
    mu_x = results.PM_fit['params']['mu_x']
    mu_y = results.PM_fit['params']['mu_y']

    map_dx = map_params.get('dx', 0)
    map_dy = map_params.get('dy', 0)
    map_dpmra = map_params.get('dpmra', 0)
    map_dpmdec = map_params.get('dpmdec', 0)

    med_dx = med_params.get('dx', 0)
    med_dy = med_params.get('dy', 0)
    med_dpmra = med_params.get('dpmra', 0)
    med_dpmdec = med_params.get('dpmdec', 0)

    ra_lin_map = x0 + mu_x*dt + map_dpmra*dt + map_dx
    dec_lin_map = y0 + mu_y*dt + map_dpmdec*dt + map_dy
    ra_lin_med = x0 + mu_x*dt + med_dpmra*dt + med_dx
    dec_lin_med = y0 + mu_y*dt + med_dpmdec*dt

    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)
    ra_map, dec_map = map_model.astrometry(tfold, system=1)
    ra_med, dec_med = med_model.astrometry(tfold, system=1)

    ra_map_full = ra_lin_map + ra_map
    dec_map_full = dec_lin_map + dec_map
    ra_med_full = ra_lin_med + ra_med
    dec_med_full = dec_lin_med + dec_med

    fig,ax = plt.subplots()

    ax.plot(ra_lin_map,dec_lin_map,label='Map Linear Model',color='red',linestyle='--',zorder=1,alpha=0.7)
    ax.plot(ra_lin_med,dec_lin_med,label='Median Linear Model',color='purple',linestyle='--',zorder=1,alpha=0.7)
    ax.plot(ra_map_full,dec_map_full,label='Map Sky Track',color='red',linestyle='-',zorder=1)
    ax.plot(ra_med_full,dec_med_full,label='Median Sky Track',color='purple',linestyle='-',zorder=1)
    ax.scatter(data.x,data.y,color='k',s=15,zorder=3)

    ax.set_aspect('equal',adjustable='datalim')
    ax.legend(loc='best')

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
        print(f"Saved sky motion plot to {savepath}")

    return fig
    


def multi_orbit_plot(results, data, Nplot=100, system=1, savepath=None):
    '''
    Plots multiple orbits from the posterior samples
    '''

    #------Gaia---------------
    if isinstance(data, GaiaData):
        Map_plot_dict = data._astrometry(Orbit(**results.MAP_params))
        med_plot_dict = data._astrometry(Orbit(**results.median_params))

        param_names = results.param_names
        samples = results.samples.get('samples', None)
        if samples is None:
            if not param_names:
                raise ValueError("Posterior samples are not available for multi-orbit plotting.")
        
            sample_arrays = [results.samples[name] for name in param_names if name in results.samples]
            if len(sample_arrays) != len(param_names):
                raise ValueError("Posterior samples are not available for multi-orbit plotting.")
            samples = np.column_stack(sample_arrays)
        
        idx = np.random.choice(samples.shape[0], size=min(Nplot, samples.shape[0]), replace=False)
        samps = samples[idx]

        fig,ax = plt.subplots()
        for samp in samps:
            model = Orbit(**dict(zip(param_names, samp)))
            plot_dict = data._astrometry(model)
            ax.plot(plot_dict['ra_orb'],plot_dict['dec_orb'],color='tab:blue',alpha=0.3)

        ax.plot(Map_plot_dict['ra_orb'],Map_plot_dict['dec_orb'],label='MAP Orbit',color='red',linestyle='-',zorder=1)
        ax.plot(med_plot_dict['ra_orb'],med_plot_dict['dec_orb'],label='Median Orbit',color='purple',linestyle='--',zorder=1)
        ax.scatter(0,0,color='k',marker='*',label = 'COM',zorder = 10)

        ax.set_xlabel(r"$\Delta \alpha^*$ (mas)")
        ax.set_ylabel(r"$\Delta \delta$ (mas)")

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.legend(fontsize='small', loc='best')
        if savepath is not None:
            fig.savefig(savepath, dpi=300)
            print(f"Saved multi-orbit plot to {savepath}")
        return fig

    #--------------------------------

    tfold = np.linspace(data.t.min(), data.t.max(), 1000)

    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)

    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)

    if map_params is None or med_params is None:
        raise ValueError("Both MAP and median parameter sets are required for multi-orbit plotting.")
    
    fixed_prior_params = {}
    for k, p in results.priors.items():
        if isinstance(p, FixedPrior):
            map_params[k] = p.value
            med_params[k] = p.value
            fixed_prior_params[k] = p.value

    # map_model = _build_model(results, map_params)
    # med_model = _build_model(results, med_params)
    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    x_map, y_map = map_model.astrometry(tfold, system=system)
    x_med, y_med = med_model.astrometry(tfold, system=system)

    ref_epoch = getattr(data, 'ref_epoch', 0)
    dt = tfold - ref_epoch

    x_map, y_map = _apply_center_offset(x_map, y_map, map_params, dt, center=True)
    x_med, y_med = _apply_center_offset(x_med, y_med, med_params, dt, center=True)

    param_names = results.param_names
    samples = results.samples.get('samples', None)
    if samples is None:
        if not param_names:
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")

        sample_arrays = [results.samples[name] for name in param_names if name in results.samples]
        if len(sample_arrays) != len(param_names):
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")
        samples = np.column_stack(sample_arrays)

    idx = np.random.choice(samples.shape[0], size=min(Nplot, samples.shape[0]), replace=False)
    samps = samples[idx]

    fig, ax = plt.subplots()

    for samp in samps:
        # model = _build_model(results, dict(zip(param_names, samp)))
        model = Orbit(**dict(zip(param_names, samp)), **fixed_prior_params)
        x, y = model.astrometry(tfold, system=system)
        x, y = _apply_center_offset(x, y, dict(zip(param_names, samp)), dt, center=True)
        ax.plot(x, y, color='tab:blue', alpha=0.3)

    ax.plot(x_map, y_map, label='MAP Orbit', color='red', linestyle='-')
    ax.plot(x_med, y_med, label='Median Orbit', color='purple', linestyle='-')
    ax.scatter(0,0,color='k',marker='*',label = 'COM',zorder = 10)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.legend(fontsize='small', loc='best')

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
        print(f"Saved multi-orbit plot to {savepath}")
    return fig

def mass_distribution(results,scale='linear',savepath=None):
    '''
    Plots distribution of secondary mass (M2) from posterior samples
    '''
    try:
        M2_samples = results['M2']
    except KeyError:
        print('No M2 samples found in results.')
        return None

    med_m2 = np.median(M2_samples)
    m2_16 = np.percentile(M2_samples, 16)
    m2_84 = np.percentile(M2_samples, 84)
    m2_m2sig = np.percentile(M2_samples, 2.5)
    m2_p2sig = np.percentile(M2_samples, 97.5)
    
    if scale == 'linear':
        kde = gaussian_kde(M2_samples)
        x = np.linspace(M2_samples.min(), M2_samples.max(), 1000)
        pdf = kde(x)
        #normalize the pdf
        pdf /= np.trapezoid(pdf, x)

        bins = np.linspace(M2_samples.min(), M2_samples.max(), 40)

        fig,ax=plt.subplots()
        ax.hist(M2_samples,bins=bins,
                 density = True, alpha = 0.5, histtype='step',
                 color='gray',label='Samples')
        
        ax.plot(x,pdf,'r-', lw=2.0,label='KDE')

        ax.axvspan(m2_16,m2_84,color='tab:blue',alpha=0.35,
                    label=fr'$1\,\sigma$  [{m2_16:.2f},{m2_84:.2f}] M$_\odot$')
        
        ax.axvspan(m2_m2sig,m2_p2sig,color='tab:blue',
            alpha=0.25,
            label=fr'$2\,\sigma$  [{m2_m2sig:.2f},{m2_p2sig:.2f}] M$_\odot$')

        ax.axvline(med_m2,color='k',linestyle='--'
            ,label=fr'Median = {med_m2:.2f} M$_\odot$')

        ax.set_xlabel("$M_{comp.}$ (M$_\\odot$)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right')
        if savepath is not None:
            fig.savefig(savepath,dpi=300)
            print(f"Saved mass distribution plot to {savepath}")
        return fig

    if scale == 'log':
        kde = gaussian_kde(np.log10(M2_samples))
        x = np.linspace(np.log10(M2_samples).min(), np.log10(M2_samples).max(), 1000)
        pdf = kde(x)
        #normalize the pdf
        pdf /= np.trapezoid(pdf, x)

        bins = np.logspace(np.log10(M2_samples).min(), np.log10(M2_samples).max(), 40)

        fig,ax=plt.subplots()

        ax.hist(M2_samples,bins=bins,
                 density = True, alpha = 0.5, histtype='step',
                 color='gray',label='Samples')
        
        ax.plot(10**x,pdf,'r-', lw=2.0,label='KDE')

        ax.axvspan(m2_16,m2_84,color='tab:blue',alpha=0.35,
                    label=fr'$1\,\sigma$  [{m2_16:.2f},{m2_84:.2f}] M$_\odot$')
        
        ax.axvspan(m2_m2sig,m2_p2sig,color='tab:blue',
            alpha=0.25,
            label=fr'$2\,\sigma$  [{m2_m2sig:.2f},{m2_p2sig:.2f}] M$_\odot$')
        
        ax.axvline(med_m2,color='k',linestyle='--'
            ,label=fr'Median = {med_m2:.2f} M$_\odot$')
        

        ax.set_xlim(m2_m2sig*0.8, m2_p2sig*1.2)
        ax.set_xscale('log')
        ax.set_xlabel("$M_{comp.}$ (M$_\\odot$)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right')

        if savepath is not None:
            fig.savefig(savepath,dpi=300)
            print(f"Saved mass distribution plot to {savepath}")
        return fig


        
def all_plots(results, data, scale=None, savepath=None):
    '''
    Generates all diagnostic and orbit plots
    '''

    if scale is None:
        scale = 'linear'

    if results.backend=='emcee':

        auto_corr = mcmc_autocorrelation_plot(results,savepath=savepath)
        corner = corner_plot(results,savepath=savepath)
        ess_dist = ess_distribution_plot(results,savepath=savepath)
        posterior_prior = posterior_over_prior(results, savepath=savepath)
        orbit_vis=orbit_plot(results,data,savepath=savepath)
        sky_vis=sky_motion_plot(results,data,savepath=savepath)
        multi_orb = multi_orbit_plot(results,data,savepath=savepath)
        mass_dist = mass_distribution(results,scale=scale,savepath=savepath)


        return auto_corr, corner, ess_dist,posterior_prior, orbit_vis, multi_orb, mass_dist

    if results.backend=='ultranest':
        posterior_prior = posterior_over_prior(results, savepath=savepath)
        corner = corner_plot(results,savepath=savepath)
        orbit_vis=orbit_plot(results,data,savepath=savepath)
        sky_vis=sky_motion_plot(results,data,savepath=savepath)
        multi_orb = multi_orbit_plot(results,data,savepath=savepath)
        mass_dist = mass_distribution(results,scale=scale,savepath=savepath)
        
                

        return posterior_prior,corner, orbit_vis, multi_orb, mass_dist