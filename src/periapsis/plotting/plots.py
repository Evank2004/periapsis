import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from periapsis.data import Data, GaiaData, JointData, AstrometryData, RadialVelocityData
from periapsis.fitting.results import FitResults, SampledPriors
from periapsis.model.orbit import Orbit
import matplotlib.gridspec as gridspec
from periapsis.prior import FixedPrior
from scipy.stats import gaussian_kde
import periapsis.params as par
from periapsis.utils.helpers import _flatten_and_join, _flatten_joint

rng_plots = np.random.default_rng(5377)

def mcmc_autocorrelation_plot(results,savepath=None):
    '''
    Plots the autocorrelation function for each parameter
    This can be used to diagnose convergence and mixing of MCMC chain
    '''
   
    param_means = np.asarray(results.samples['param_means'])
    param_names = results.param_names
    if results.fit_method =='linear':
        param_names = [name for name in param_names if name in ('P', 'e', 'Tp')]
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
        fig.savefig(f'{savepath}/mcmc_autocorrelation_plot.png',dpi=300)
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
        fig.savefig(f'{savepath}/corner.png',dpi=300)
        print(f"Saved corner plot to {savepath}")
    return fig


def ess_distribution_plot(results,savepath=None):
    '''
    Plots distribution of effective sample size (ESS) for each parameter.
    This can be used to diagnose convergence
    '''
    ess = results.Ess
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
    # ax.legend()
    if savepath is not None:
        fig.savefig(f'{savepath}/ess_dist.png',bbox_inches= 'tight',dpi=300)
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
        fig.savefig(f'{savepath}/prior_dist.png',dpi=300)
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
        fig.savefig(f'{savepath}/prior_2dhist.png',dpi=300)
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
        fig.savefig(f'{savepath}/conditional_2dhist.png', dpi=300)
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
        fig.savefig(f'{savepath}/posterior_over_prior.png',dpi=300)
        print(f"Saved posterior over prior plot to {savepath}")
    
    return fig


#---------------Orbit Visualization Plots ---------------------------------

def _apply_center_offset_astro(x, y,plxf_x,plxf_y, params, dt, center=True):
    if params is None or not center:
        return np.asarray(x), np.asarray(y)

    dalpha = params.get(par.dalpha, 0)
    ddelta = params.get(par.ddelta, 0)
    mu_alpha = params.get(par.mu_alpha, 0)
    mu_delta = params.get(par.mu_delta, 0)
    parallax = params.get(par.parallax, 0)
    return np.asarray(x) - dalpha - mu_alpha * dt - plxf_x*parallax, np.asarray(y) - ddelta - mu_delta * dt - plxf_y*parallax

def _pericenter_coords(params, system):

    alpha_peri = params[f'B{system}'] * (1 - params['e'])
    delta_peri = params[f'A{system}'] * (1 - params['e'])
    return alpha_peri, delta_peri

def orbit_plot(results, data, savepath=None):

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.05)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    

    def _plot_model(results,tfold,system):

        map_params = getattr(results, 'MAP_params', None)
        if map_params is None:
            map_params = results.samples.get('MAP_params', None)
        
        med_params = getattr(results, 'median_params', None)
        if med_params is None:
            med_params = results.samples.get('median_params', None)

        map_model = Orbit(**map_params)
        med_model = Orbit(**med_params)

        alpha_map, delta_map = map_model.pure_orbit(tfold, system=system)
        alpha_med, delta_med = med_model.pure_orbit(tfold, system=system)

        alpha_peri, delta_peri = _pericenter_coords(map_params, system)

        ax1.plot(tfold, alpha_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
        ax1.plot(tfold, alpha_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)
        ax2.plot(tfold, delta_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
        ax2.plot(tfold, delta_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)
        ax3.plot(alpha_map, delta_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
        ax3.plot(alpha_med, delta_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)
        ax3.scatter(0, 0, color='k', marker='*', label='COM', zorder=10, s=100)
        ax3.plot([0, alpha_peri], [0, delta_peri], color='gray', linestyle='--', label='Periastron', zorder=4, alpha=0.6)
     




    def _plot_astrometry_data(results,data):

        alpha_obs, delta_obs = _apply_center_offset_astro(data.x, data.y, data.plxf_x,data.plxf_y, results.MAP_params, data.t)
        dt_obs = data.t

        ax1.errorbar(dt_obs, alpha_obs, yerr=data.x_err,color='k', fmt='o',markersize=4,zorder=2)
        ax2.errorbar(dt_obs, delta_obs, yerr=data.y_err,color='k', fmt='o',markersize=4,zorder=2)
        ax3.scatter(alpha_obs, delta_obs, color='k',s=15,zorder=2)
   

    def _plot_gaia_data(results,data):
        Map_plot_dict = data._astrometry(Orbit(**results.MAP_params))
        Med_plot_dict = data._astrometry(Orbit(**results.median_params))

        dt_obs = data.t 

        for i, (ri, di, ei, si, ci) in enumerate(zip(Map_plot_dict['ra_orb_obs'], Map_plot_dict['dec_orb_obs'], data.err, data.spsi, data.cpsi)):
            x0 = ri - ei * si
            x1 = ri + ei * si
            y0 = di - ei * ci
            y1 = di + ei * ci

            ax1.plot([dt_obs[i], dt_obs[i]], [x0, x1], color='tab:orange', alpha=0.5, zorder=2)
            ax2.plot([dt_obs[i], dt_obs[i]], [y0, y1], color='tab:orange', alpha=0.5, zorder=2)
            ax3.plot([x0, x1], [y0, y1], color='tab:orange', alpha=0.5, zorder=2)

        ax1.scatter(dt_obs, Map_plot_dict['ra_orb_obs'], color='k', s=15, zorder=3)
        ax2.scatter(dt_obs, Map_plot_dict['dec_orb_obs'], color='k', s=15, zorder=3)
        ax3.scatter(Map_plot_dict['ra_orb_obs'], Map_plot_dict['dec_orb_obs'], color='k', s=15, zorder=3)
        ax3.plot([0, Map_plot_dict['ra_peri']], [0, Map_plot_dict['dec_peri']], color='gray', linestyle='--', label='Periastron', zorder=4, alpha=0.6)
        ax3.scatter(0, 0, color='k', marker='*', label='COM', zorder=10, s=100)
        

    datas = _flatten_joint(data)
    systems = {d.system for d in datas}
    min_t = []
    max_t = []
    for d in datas:
        min_t.append(np.min(d.t))
        max_t.append(np.max(d.t))
    tfold = np.linspace(np.min(min_t), np.max(max_t), 1000)
    for s in systems:
        _plot_model(results,tfold,s)
    for d in datas:
        if d.system not in systems:
            continue
        if isinstance(d, AstrometryData):
            _plot_astrometry_data(results,d)
        elif isinstance(d, GaiaData):
            _plot_gaia_data(results,d)

    ax2.set_xlabel('Time')
    ax1.set_ylabel(r"$\Delta \alpha^*$ ")
    ax2.set_ylabel(r"$\Delta \delta$ ")
    ax3.set_xlabel(r"$\Delta \alpha^*$ ")
    ax3.set_ylabel(r"$\Delta \delta$ ")
    ax3.set_aspect('equal',adjustable = 'datalim')
    #Deduplicates labels if multiple systems are plotted
    handles, labels = ax3.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax3.legend(unique_legend.values(), unique_legend.keys())
    ax3.invert_xaxis()
    

    if savepath is not None:
        fig.savefig(f'{savepath}/orbit_plot.png', dpi=300)
        print(f"Saved orbit plot to {savepath}")
    return fig

def sky_motion_plot(results, data, savepath=None):
    '''
    Plots full sky motion over time
    '''

    fig, ax = plt.subplots()

    def _plot_lin_model(results, data,tfold):

        null_fit = results.null_hypothesis
        dalpha = null_fit['params']['dalpha']
        ddelta = null_fit['params']['ddelta']
        mu_alpha = null_fit['params']['mu_alpha']
        mu_delta = null_fit['params']['mu_delta']
        parallax = null_fit['params']['parallax']

        if isinstance(data, GaiaData):
            tfold = np.linspace(data.t.min(), data.t.max(), 1000)
            plx_alpha = np.interp(tfold,data.t,data.plx_fac*data.spsi)
            plx_delta = np.interp(tfold,data.t,data.plx_fac*data.cpsi)

            alpha_lin = mu_alpha *tfold + dalpha + plx_alpha*parallax
            delta_lin = mu_delta *tfold + ddelta + plx_delta*parallax
            ax.plot(alpha_lin, delta_lin, label='Linear Model', color='orange', linestyle='-.', zorder=1, alpha=0.7)

            map_plot_dict = data._astrometry(Orbit(**results.MAP_params))
            med_plot_dict = data._astrometry(Orbit(**results.median_params))
            ax.plot(map_plot_dict['ra_sky'], map_plot_dict['dec_sky'], label='MAP Sky Track', color='red', linestyle='-', zorder=1)
            ax.plot(med_plot_dict['ra_sky'], med_plot_dict['dec_sky'], label='Median Sky Track', color='purple', linestyle='-', zorder=1)

        if isinstance(data,AstrometryData):
            
            alpha_lin = mu_alpha * tfold + dalpha + data.plxf_x*parallax
            delta_lin = mu_delta * tfold + ddelta + data.plxf_y*parallax
            ax.plot(alpha_lin, delta_lin, label='Linear Model', color='orange', linestyle='-.', zorder=1, alpha=0.7)

            map_model = Orbit(**results.MAP_params)
            med_model = Orbit(**results.median_params)
            alpha_map, delta_map = map_model.astrometry(tfold, data.plxf_x, data.plxf_y, system=data.system)
            alpha_med, delta_med = med_model.astrometry(tfold, data.plxf_x, data.plxf_y, system=data.system)
            ax.plot(alpha_map, delta_map, label='MAP Sky Track', color='red', linestyle='-', zorder=1)
            ax.plot(alpha_med, delta_med, label='Median Sky Track', color='purple', linestyle='-', zorder=1)


    def _plot_data(results, data):
        if isinstance(data, GaiaData):
            map_plot_dict = data._astrometry(Orbit(**results.MAP_params))
            ax.scatter(map_plot_dict['ra_sky_data'], map_plot_dict['dec_sky_data'], color='k', s=15, zorder=3)
            
        elif isinstance(data, AstrometryData):
            ax.scatter(data.x, data.y, color='k', s=15, zorder=3)
            


    datas = _flatten_joint(data)
    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t))
        t_max.append(np.max(d.t))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        _plot_lin_model(results, d,tfold)
        _plot_data(results, d)

    ax.legend(loc='best')
    ax.set_xlabel(r"$\Delta \alpha^*$ ")
    ax.set_ylabel(r"$\Delta \delta$ ")
    ax.set_aspect('equal', adjustable='datalim')
    ax.invert_xaxis()
    
    if savepath is not None:
        fig.savefig(f'{savepath}/sky_motion.png', dpi=300)
        print(f"Saved sky motion plot to {savepath}")

    return fig
    


def multi_orbit_plot(results, data, Nplot=100, savepath=None):
    '''
    Plots multiple orbits from the posterior samples
    '''
    fig,ax = plt.subplots()

    def _plot_orbit_samples(results, data, Nplot,tfold):
        param_names = results.param_names
        samples = results.samples.get('samples', None)
        if samples is None:
            if not param_names:
                raise ValueError("Posterior samples are not available for multi-orbit plotting.")

        sample_arrays = [results.samples[name] for name in param_names if name in results.samples]
        if len(sample_arrays) != len(param_names):
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")

        idx = np.random.choice(samples.shape[0], size=min(Nplot, samples.shape[0]), replace=False)
        samps = samples[idx]

        for samp in samps:
            model = Orbit(**dict(zip(param_names, samp)))
            alpha,delta = model.pure_orbit(tfold, system=data.system)
            ax.plot(alpha, delta, color='tab:blue', alpha=0.3, zorder=1)

    def _plot_map_median_orbits(results, data, tfold):

        map_model = Orbit(**results.MAP_params)
        med_model = Orbit(**results.median_params)

        alpha_map, delta_map = map_model.pure_orbit(tfold, system=data.system)
        alpha_med, delta_med = med_model.pure_orbit(tfold, system=data.system)

        ax.plot(alpha_map, delta_map, label='MAP Orbit', color='red', linestyle='-', zorder=4)
        ax.plot(alpha_med, delta_med, label='Median Orbit', color='purple', linestyle='--', zorder=4)
        ax.scatter(0, 0, color='k', marker='*', label='COM', zorder=10)


    datas = _flatten_joint(data)
    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t))
        t_max.append(np.max(d.t))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        _plot_orbit_samples(results, d, Nplot, tfold)
        _plot_map_median_orbits(results, d, tfold)

    
    ax.set_xlabel(r"$\Delta \alpha^*$")
    ax.set_ylabel(r"$\Delta \delta$")
    ax.legend(fontsize='small', loc='best')
    ax.set_aspect('equal')
    ax.invert_xaxis()

    if savepath is not None:
        fig.savefig(f'{savepath}/multi_orbit.png', dpi=300)
        print(f"Saved multi-orbit plot to {savepath}")

    return fig


def distribution(results, param, scale='linear', unit='', savepath=None):
    '''
    Plots distribution of a parameter from posterior samples
    '''
    try:
        samples = results[param]
    except KeyError:
        print(f'No {param} samples found in results.')
        return None

    med = np.median(samples)
    p16 = np.percentile(samples, 16)
    p84 = np.percentile(samples, 84)
    par_m2sig = np.percentile(samples, 2.5)
    par_p2sig = np.percentile(samples, 97.5)

    if scale == 'linear':
        kde = gaussian_kde(samples)
        x = np.linspace(samples.min(), samples.max(), 1000)
        pdf = kde(x)
        #normalize the pdf
        pdf /= np.trapezoid(pdf, x)

        bins = np.linspace(samples.min(), samples.max(), 40)

        fig,ax=plt.subplots()
        ax.hist(samples,bins=bins,
                 density = True, alpha = 0.5, histtype='step',
                 color='gray',label='Samples')
        
        ax.plot(x,pdf,'r-', lw=2.0,label='KDE')

        ax.axvspan(p16,p84,color='tab:blue',alpha=0.35,
                    label=fr'$1\,\sigma$  [{p16:.2f},{p84:.2f}] {unit}')
        
        ax.axvspan(par_m2sig,par_p2sig,color='tab:blue',
            alpha=0.25,
            label=fr'$2\,\sigma$  [{par_m2sig:.2f},{par_p2sig:.2f}] {unit}')

        ax.axvline(med,color='k',linestyle='--'
            ,label=fr'Median = {med:.2f} {unit}')

        ax.set_xlabel(f"{param} [{unit}]")
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right')
        if savepath is not None:
            fig.savefig(f'{savepath}/distribution_plot.png',dpi=300)
            print(f"Saved distribution plot to {savepath}")
        return fig
    
    if scale == 'log':
        kde = gaussian_kde(np.log10(samples))
        x = np.linspace(np.log10(samples).min(), np.log10(samples).max(), 1000)
        pdf = kde(x)
        #normalize the pdf
        pdf /= np.trapezoid(pdf, x)

        bins = np.logspace(np.log10(samples).min(), np.log10(samples).max(), 40)

        fig,ax=plt.subplots()

        ax.hist(samples,bins=bins,
                 density = True, alpha = 0.5, histtype='step',
                 color='gray',label='Samples')
        
        ax.plot(10**x,pdf,'r-', lw=2.0,label='KDE')

        ax.axvspan(p16,p84,color='tab:blue',alpha=0.35,
                    label=fr'$1\,\sigma$  [{p16:.2f},{p84:.2f}] {unit}')
        
        ax.axvspan(par_m2sig,par_p2sig,color='tab:blue',
            alpha=0.25,
            label=fr'$2\,\sigma$  [{par_m2sig:.2f},{par_p2sig:.2f}] {unit}')
        
        ax.axvline(med,color='k',linestyle='--'
            ,label=fr'Median = {med:.2f} {unit}')
        

        ax.set_xlim(par_m2sig*0.8, par_p2sig*1.2)
        ax.set_xscale('log')
        ax.set_xlabel(f"{param} [{unit}]")
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right')

        if savepath is not None:
            fig.savefig(f'{savepath}/distribution_plot.png',dpi=300)
            print(f"Saved distribution plot to {savepath}")
        return fig



def mass_distribution(results,scale='linear',savepath=None):
    '''
    Plots distribution of secondary mass (M2) from posterior samples
    '''
    if 'M2' in results and isinstance(results['M2'], np.ndarray) and len(results['M2']) > 1:
        return distribution(results, 'M2', scale=scale, unit=r'M$_\odot$', savepath=savepath)
    elif 'M1' in results and isinstance(results['M1'], np.ndarray) and len(results['M1']) > 1:
        return distribution(results, 'M1', scale=scale, unit=r'M$_\odot$', savepath=savepath)
    elif 'minM2' in results and isinstance(results['minM2'], np.ndarray) and len(results['minM2']) > 1:
        return distribution(results, 'minM2', scale=scale, unit=r'M$_\odot$', savepath=savepath)
    elif 'minM1' in results and isinstance(results['minM1'], np.ndarray) and len(results['minM1']) > 1:
        return distribution(results, 'minM1', scale=scale, unit=r'M$_\odot$', savepath=savepath)
    else:
        print('No mass samples found in results.')
        return None


def rv_fit_plot(results, data,unit_conv=1, savepath=None):
    '''
    Plots radial velocity fit over time
    '''
    if isinstance(data, JointData):
        datas = [d for d in data.datas if isinstance(d, RadialVelocityData)]
    else:
        datas = [data] if isinstance(data, RadialVelocityData) else []

    if not datas:
        print("No radial velocity data found for plotting.")
        return None

    fig,ax = plt.subplots()

    def _plot_rv_fit(results,data,unit_conv,system,tfold):
        Map_fit = Orbit(**results.MAP_params)
        med_fit = Orbit(**results.median_params)

        ax.plot(tfold, Map_fit.rv(tfold,system), label='MAP Fit', color='red', linestyle='-')
        ax.plot(tfold, med_fit.rv(tfold,system), label='Median Fit',color='purple', linestyle='--',alpha=0.7)
        ax.errorbar(data.t, data.rv*unit_conv, yerr=data.rv_err*unit_conv, fmt='o', color='k', markersize=4)
        

    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t))
        t_max.append(np.max(d.t))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        _plot_rv_fit(results,d,unit_conv,d.system,tfold)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('RV') #TODO: make this able to provide units
    ax.legend(loc='upper right')

    if savepath is not None:
        fig.savefig(f'{savepath}/rv_fit.png', dpi=300)
        print(f"Saved radial velocity fit plot to {savepath}")
    return fig


def phase_fold_rv_plot(results, data, unit_conv=1,savepath=None): #TODO: remove unit_conv once units are normalized
    '''
    Plots phase-folded radial velocity fit
    '''
    if isinstance(data, JointData):
        datas = [d for d in data.datas if isinstance(d, RadialVelocityData)]
    else:
        datas = [data] if isinstance(data, RadialVelocityData) else []

    if not datas:
        print("No radial velocity data found for plotting.")
        return None

    fig, ax = plt.subplots()
    map_model = Orbit(**results.MAP_params)
    map_period = map_model['P']
    map_tp = map_model['Tp']

    def plot_system(system, system_datas):
        tfold = np.linspace(map_tp - 0.5 * map_period, map_tp + 0.5 * map_period, 10000)
        phase_fold = (tfold - map_tp + 0.5 * map_period) / map_period
        sort_map = np.argsort(phase_fold)

        ax.plot(
            phase_fold[sort_map],
            map_model.rv(tfold, system)[sort_map] * unit_conv,
            label=f'MAP Fit ({system})',
            color='red',
            lw=1.5,
            linestyle='-'
        )

        for d in system_datas:
            phase = (d.t - map_tp + 0.5 * map_period) / map_period % 1
            ax.errorbar(phase, d.rv * unit_conv, yerr=d.rv_err * unit_conv, fmt='o', color='k', markersize=4)

    for system in sorted({d.system for d in datas}):
        system_datas = [d for d in datas if d.system == system]
        plot_system(system, system_datas)

    ax.set_xlabel('Phase')
    ax.set_ylabel('RV')
    ax.legend(loc='best')

    if savepath is not None:
        fig.savefig(f'{savepath}/phase_fold_rv.png', dpi=300)
        print(f"Saved phase-folded radial velocity plot to {savepath}")

    return fig

def rv_multi_fit_plot(results,data,Nplot=100,unit_conv=1,savepath=None):
    '''
    Plots multiple radial velocity fits from posterior samples
    '''
    if isinstance(data, JointData):
        datas = [d for d in data.datas if isinstance(d, RadialVelocityData)]
    else:
        datas = [data] if isinstance(data, RadialVelocityData) else []

    if not datas:
        print("No radial velocity data found for plotting.")
        return None

    param_names = results.param_names
    samples = results.samples.get('samples', None)
    if samples is None:
        if not param_names:
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")
    
        sample_arrays = [results.samples[name] for name in param_names if name in results.samples]
        if len(sample_arrays) != len(param_names):
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")
        samples = np.column_stack(sample_arrays)

    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)

    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)

    fixed_prior_params = {}
    for k, p in results.priors.items():
        if isinstance(p, FixedPrior):
            map_params[k] = p.value
            med_params[k] = p.value
            fixed_prior_params[k] = p.value

    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    fig,ax = plt.subplots()

    def _multi_rvs(Nplot,tfold,system):
        
        idx = np.random.choice(samples.shape[0], size=min(Nplot, samples.shape[0]), replace=False)
        samps = samples[idx]

        for samp in samps:
            model = Orbit(**dict(zip(param_names, samp)), **fixed_prior_params)
            ax.plot(tfold, model.rv(tfold,system), color='tab:blue', alpha=0.3)

        ax.plot(tfold, map_model.rv(tfold,system), label='MAP Fit', color='red', linestyle='-')
        ax.plot(tfold, med_model.rv(tfold,system), label='Median Fit', color='purple', linestyle='--', alpha=0.7)

    def _plot_data(data,unit_conv):
        ax.errorbar(data.t, data.rv*unit_conv, yerr=data.rv_err*unit_conv, fmt='o', color='k', markersize=4)

    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t))
        t_max.append(np.max(d.t))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        _multi_rvs(Nplot,tfold,d.system)
        _plot_data(d,unit_conv)

    ax.set_xlabel('Time')
    ax.set_ylabel('RV') #TODO: make this able to provide units
    ax.legend(loc='best')

    if savepath is not None:
        fig.savefig(f'{savepath}/rv_multi_fit.png', dpi=300)
        print(f"Saved multi-radial velocity fit plot to {savepath}")
    return fig


def multi_phase_plot(results,data,Nplot=100,unit_conv=1,savepath=None):
    '''
    Plots multiple phase-folded RV fits
    '''
    if isinstance(data, JointData):
        datas = [d for d in data.datas if isinstance(d, RadialVelocityData)]
    else:
        datas = [data] if isinstance(data, RadialVelocityData) else []

    param_names = results.param_names
    samples = results.samples.get('samples', None)
    if samples is None:
        if not param_names:
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")
        
        sample_arrays = [results.samples[name] for name in param_names if name in results.samples]
        if len(sample_arrays) != len(param_names):
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")
        samples = np.column_stack(sample_arrays)
        
    
    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)
    
    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)
    
    fixed_prior_params = {}
    for k, p in results.priors.items():
        if isinstance(p, FixedPrior):
            map_params[k] = p.value
            med_params[k] = p.value
            fixed_prior_params[k] = p.value
    
    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    if not datas:
        print("No radial velocity data found for plotting.")
        return None

    fig, ax = plt.subplots()

    def _multi_phase(Nplot, system):
        idx = np.random.choice(samples.shape[0], size=min(Nplot, samples.shape[0]), replace=False)
        samps = samples[idx]
        
        for samp in samps:
            model = Orbit(**dict(zip(param_names, samp)), **fixed_prior_params)
            P_samp = model['P']
            Tp_samp = model['Tp']
            tfold_samp = np.linspace(Tp_samp - 0.5*P_samp, Tp_samp + 0.5*P_samp, 10000)
            phase_fold_samp = (tfold_samp - Tp_samp + 0.5*P_samp) / P_samp
            rv_samp = model.rv(tfold_samp, system)
            ax.plot(phase_fold_samp, rv_samp/unit_conv, color='tab:blue', alpha=0.3)


    def _map_median_phase(system):
        P_map = map_model['P']
        Tp_map = map_model['Tp']
        P_med = med_model['P']
        Tp_med = med_model['Tp']

        tfold_map = np.linspace(Tp_map - 0.5*P_map, Tp_map + 0.5*P_map, 10000)
        phase_fold_map = (tfold_map - Tp_map + 0.5*P_map) / P_map
        rv_map = map_model.rv(tfold_map, system)

        tfold_med = np.linspace(Tp_med - 0.5*P_med, Tp_med + 0.5*P_med, 10000)
        phase_fold_med = (tfold_med - Tp_med + 0.5*P_med) / P_med
        rv_med = med_model.rv(tfold_med, system)

        ax.plot(phase_fold_map, rv_map/unit_conv, label='MAP Fit', color='red', lw=1.5, linestyle='-')
        ax.plot(phase_fold_med, rv_med/unit_conv, label='Median Fit', color='purple', lw=1.5, linestyle='--', alpha=0.7)

    def _plot_data(data, unit_conv):
        P = map_model['P']
        Tp = map_model['Tp']
        phase = (data.t - Tp + 0.5*P) / P % 1
        ax.errorbar(phase, data.rv*unit_conv, yerr=data.rv_err*unit_conv, fmt='o', color='k', markersize=4)

    systems = {d.system for d in datas}
    for system in sorted(systems):
        system_datas = [d for d in datas if d.system == system]
        _multi_phase(Nplot, system)
        _map_median_phase(system)
        for d in system_datas:
            _plot_data(d, unit_conv)

    ax.set_xlabel('Phase')
    ax.set_ylabel('RV') #TODO: make this able to provide units
    ax.legend(loc='best')

    if savepath is not None:
        fig.savefig(f'{savepath}/multi_phase.png', dpi=300)
        print(f"Saved multi-phase-folded radial velocity plot to {savepath}")
    return fig

        
def all_plots(results, data: Data, scale=None,unit_conv=1, savepath=None):
    '''
    Generates all diagnostic and orbit plots
    '''

    if scale is None:
        scale = 'linear'


    if results.backend=='emcee':
        mcmc_autocorrelation_plot(results,savepath=savepath)
        ess_distribution_plot(results,savepath=savepath)
    corner_plot(results,savepath=savepath)
    posterior_over_prior(results, savepath=savepath)
    if data.has_radial_velocity():
        rv_fit_plot(results,data,unit_conv=unit_conv,savepath=savepath)
        phase_fold_rv_plot(results,data,unit_conv=unit_conv,savepath=savepath)
        rv_multi_fit_plot(results,data,unit_conv=unit_conv,savepath=savepath)
        multi_phase_plot(results,data,unit_conv=unit_conv,savepath=savepath)
    if data.has_astrometry():
        orbit_plot(results,data,savepath=savepath)
        sky_motion_plot(results,data,savepath=savepath)
        multi_orbit_plot(results,data,savepath=savepath)
    mass_distribution(results,scale=scale,savepath=savepath)


