import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from periapsis.fitting.results import FitResults
from periapsis.model import thieleinnes
from periapsis.model import campbell
import matplotlib.gridspec as gridspec
from periapsis.utils.solvers import solve_mass
from periapsis.utils.solvers import campbell_to_thiele
from periapsis.utils.solvers import transform_theile
from periapsis.utils.helpers import _build_model
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

def corner_plot(results,savepath=None):
    '''
    Plots the corner plot for sampled parameters 
    '''

    param_names = results.param_names
    samples = np.array([results.samples[name] for name in param_names]).T

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


def _sample_induced_m2_prior(results, priors, m1=None, size=5000, rng=None):
    rng = rng_plots if rng is None else rng

    if m1 is None:
        m1 = getattr(results, 'm1', None)
    if m1 is None:
        m1 = getattr(results, 'samples', {}).get('m1', None)
    if m1 is None:
        return None

    fit_method = getattr(results, 'fit_method', None)
    if fit_method is None:
        fit_method = getattr(results, 'samples', {}).get('fit_method', None)

    if fit_method == 'linear' and all(name in priors for name in ['a', 'cosi', 'omega', 'Omega']):
        P = np.asarray(priors['P'].sample(rng, size=size), dtype=float).ravel()
        a = np.asarray(priors['a'].sample(rng, size=size), dtype=float).ravel()
        cosi = np.asarray(priors['cosi'].sample(rng, size=size), dtype=float).ravel()
        omega = np.asarray(priors['omega'].sample(rng, size=size), dtype=float).ravel()
        Omega = np.asarray(priors['Omega'].sample(rng, size=size), dtype=float).ravel()

        A, B, F, G = campbell_to_thiele(a, cosi, omega, Omega)
        a1, _, _, _ = transform_theile(A, B, F, G)
        m2 = solve_mass(np.asarray(a1, dtype=float), P, float(m1))
    elif all(name in priors for name in ['P', 'a']):
        P = np.asarray(priors['P'].sample(rng, size=size), dtype=float).ravel()
        a = np.asarray(priors['a'].sample(rng, size=size), dtype=float).ravel()
        m2 = solve_mass(np.asarray(a, dtype=float), P, float(m1))
    else:
        return None

    m2 = np.asarray(m2, dtype=float).ravel()
    return m2[np.isfinite(m2) & (m2 > 0) & (m2<50)] #TODO fix this later


def _sample_induced_thiele_priors(priors, size=5000, rng=None):
    rng = rng_plots if rng is None else rng

    required = ['a', 'cosi', 'omega', 'Omega']
    if not all(name in priors for name in required):
        return None

    a = np.asarray(priors['a'].sample(rng, size=size), dtype=float).ravel()
    cosi = np.asarray(priors['cosi'].sample(rng, size=size), dtype=float).ravel()
    omega = np.asarray(priors['omega'].sample(rng, size=size), dtype=float).ravel()
    Omega = np.asarray(priors['Omega'].sample(rng, size=size), dtype=float).ravel()

    A, B, F, G = campbell_to_thiele(a, cosi, omega, Omega)
    return {
        'A': np.asarray(A, dtype=float).ravel(),
        'B': np.asarray(B, dtype=float).ravel(),
        'F': np.asarray(F, dtype=float).ravel(),
        'G': np.asarray(G, dtype=float).ravel(),
    }


def posterior_over_prior(results,priors,m1=None,savepath=None,ncols=2,bins=100):
    '''
    Plots the posterior distribution over the prior distribution for each parameter.
    This can be used to diagnose convergence and mixing of MCMC chain
    '''
    param_names = [name for name in results.param_names if name in results.samples]
    
    for derived_name in ('M2',):
        if derived_name in results.samples and derived_name not in param_names:
            param_names.append(derived_name)
        

    sample_arrays = []
    for name in param_names:
        value = results.samples[name]
        if isinstance(value, np.ndarray):
            sample_arrays.append(np.asarray(value, dtype=float).ravel())
        else:
            sample_arrays.append(np.asarray(value, dtype=float).reshape(-1))
    samples = np.array(sample_arrays, dtype=float).T

    induced_thiele_priors = None
    if any(name in {'A', 'B', 'F', 'G'} for name in param_names):
        induced_thiele_priors = _sample_induced_thiele_priors(priors, size=5000, rng=rng_plots)

    induced_m2_prior = None
    if 'M2' in param_names:
        induced_m2_prior = _sample_induced_m2_prior(results, priors, m1=m1, size=5000, rng=rng_plots)

    ncols = max(1, min(int(ncols), len(param_names)))
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 2.8 * nrows), squeeze=False)
    axes = axes.ravel()

    

    for i, name in enumerate(param_names):
        ax = axes[i]
        prior = priors.get(name)
        sample_values = samples[:, i]
        finite_sample_values = sample_values[np.isfinite(sample_values)]

        prior_samples = None
        if prior is not None:
            prior_samples = np.asarray(prior.sample(rng_plots, size=5000), dtype=float).ravel()
            prior_samples = prior_samples[np.isfinite(prior_samples)]
        elif induced_thiele_priors is not None and name in induced_thiele_priors:
            prior_samples = induced_thiele_priors[name]
        elif induced_m2_prior is not None and name == 'M2':
            prior_samples = induced_m2_prior

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

    plt.tight_layout()
    
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


def orbit_plot(results, data, savepath=None):

    

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

    map_model = _build_model(results, map_params)
    med_model = _build_model(results, med_params)

    x_map_raw, y_map_raw = map_model.astrometry(tfold)
    x_med_raw, y_med_raw = med_model.astrometry(tfold)

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


def multi_orbit_plot(results, data, savepath=None, Nplot=100):
    '''
    Plots multiple orbits from the posterior samples
    '''
    tfold = np.linspace(data.t.min(), data.t.max(), 1000)

    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)

    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)

    if map_params is None or med_params is None:
        raise ValueError("Both MAP and median parameter sets are required for multi-orbit plotting.")

    map_model = _build_model(results, map_params)
    med_model = _build_model(results, med_params)

    x_map, y_map = map_model.astrometry(tfold)
    x_med, y_med = med_model.astrometry(tfold)

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
        model = _build_model(results, dict(zip(param_names, samp)))
        x, y = model.astrometry(tfold)
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
    M2_samples = results.samples.get('M2', None)
    if M2_samples is None:
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
            fig.savefig(f'{savepath}/Mass_dist.png',dpi=300)
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
            fig.savefig(f'{savepath}/Mass_dist_log.png',dpi=300)
            print(f"Saved mass distribution plot to {savepath}")
        return fig


        
def all_plots(results,data,priors,m1,scale=None,savepath=None):
    '''
    Generates all diagnostic and orbit plots
    '''

    if scale is None:
        scale = 'linear'

    if results.backend=='emcee':

        auto_corr = mcmc_autocorrelation_plot(results,savepath=savepath)
        corner = corner_plot(results,savepath=savepath)
        ess_dist = ess_distribution_plot(results,savepath=savepath)
        posterior_prior = posterior_over_prior(results,priors,m1,savepath=savepath)
        orbit_vis=orbit_plot(results,data,savepath=savepath)
        multi_orb = multi_orbit_plot(results,data,savepath=savepath)
        mass_dist = mass_distribution(results,scale=scale,savepath=savepath)


        return auto_corr, corner, ess_dist,posterior_prior, orbit_vis, multi_orb, mass_dist

    if results.backend=='ultranest':
        posterior_prior = posterior_over_prior(results,priors,m1,savepath=savepath)
        corner = corner_plot(results,savepath=savepath)
        orbit_vis=orbit_plot(results,data,savepath=savepath)
        multi_orb = multi_orbit_plot(results,data,savepath=savepath)
        mass_dist = mass_distribution(results,scale=scale,savepath=savepath)
        
                

        return posterior_prior,corner, orbit_vis, multi_orb, mass_dist