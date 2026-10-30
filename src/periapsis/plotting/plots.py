import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from periapsis.data import Data, GaiaData, JointData, AstrometryData, PhysicalAstrometryData, RadialVelocityData
from periapsis.fitting.results import FitResults, SampledPriors
from periapsis.model.orbit import Orbit
import matplotlib.gridspec as gridspec
from periapsis.prior import FixedPrior
from scipy.stats import gaussian_kde
import periapsis.params as par
from periapsis.utils.helpers import _flatten_and_join, _flatten_joint
from periapsis.params.units import CanonicalUnits, normalize_unit

rng_plots = np.random.default_rng(5377)


def _canonical_to_plot(data, parameter, values):
    unit = data.parameter_unit(parameter)
    dimension = data.parameter_dimension(parameter)
    return np.asarray(values) / unit.to(CanonicalUnits[dimension])


def _plot_to_canonical(data, parameter, values):
    unit = data.parameter_unit(parameter)
    dimension = data.parameter_dimension(parameter)
    return np.asarray(values) * unit.to(CanonicalUnits[dimension])


def _plot_unit(data, parameter):
    return str(data.parameter_unit(parameter))


def _plot_time(data, values):
    return _canonical_to_plot(data, 't', values)


def _plot_position(data, values):
    return _canonical_to_plot(data, 'x', values)


def _plot_velocity(data, values):
    return _canonical_to_plot(data, 'rv', values)

def mcmc_autocorrelation_plot(results,savepath=None):
    '''
    Plots the autocorrelation function for each parameter
    This can be used to diagnose convergence and mixing of MCMC chain
    '''
   
    param_means = np.asarray(results.samples['param_means'])
    if param_means.ndim != 2:
        raise ValueError("Parameter means must be a two-dimensional array.")
    param_names = getattr(results, 'sampled_param_names', None)
    if param_names is None:
        param_names = list(results.param_names)
        

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
    units = [results.unit_system.get_parameter_unit(name) for name in param_names]

    samples = np.array([results[name, unit] for name, unit in zip(param_names, units)]).T

    labels = [f"{name} [{unit.to_string('latex_inline')}] " if str(unit).strip() else name for name, unit in zip(param_names, units)]
    fig = corner.corner(samples, quantiles=[0.16,0.5,0.84],
        color='tab:blue', labels=labels, show_titles=True, verbose=False,
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


def prior_dist_plot(result: FitResults, rng=np.random.default_rng(), nsamples=10000, params=None, savepath=None, bins=100, ncols=None):
    '''
    Plots the prior distribution for each parameter.
    This can be used to diagnose convergence and mixing of MCMC chain
    '''
    sampled_priors = result.sample_priors(rng, nsamples)
    param_names = sampled_priors.param_order if params is None else params
    samples = {name: sampled_priors[name] for name in param_names}
    samples = {name: value for name, value in samples.items() if not np.all(np.isnan(value))}


    # Initialize Axes
    ncols = max(1, min(int(np.ceil(np.sqrt(len(samples)))), len(samples))) if ncols is None else ncols
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
    axes = axes.ravel()

    for i, name in enumerate(samples.keys()):
        ax = axes[i]
        unit = result.unit_system.get_parameter_unit(name)
        prior_samples = result.unit_system.convert_from_canonical(name, samples[name], unit=unit)
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
            ax.set_xlabel(f'{name} [{unit.to_string("latex_inline")}]' if str(unit).strip() else name)
        else:
            ax.set_axis_off()

    for j in range(len(samples), len(axes)):
        axes[j].axis('off')

    if savepath is not None:
        fig.savefig(f'{savepath}/prior_dist.png',dpi=300)
        print(f"Saved prior distribution plot to {savepath}")
    
    return fig

def prior_histogram_2d(result: FitResults, param_x, param_y, rng=np.random.default_rng(), nsamples=10000, unit_x=None, unit_y=None, savepath=None, bins=100):
    '''
    Plots the 2D histogram of the prior distribution for two parameters.
    This can be used to diagnose convergence and mixing of MCMC chain
    '''
    sampled_priors = result.sample_priors(rng, nsamples)

    if unit_x is None:
        unit_x = result.unit_system.get_parameter_unit(param_x)
    else:
        unit_x = normalize_unit(unit_x)
    if unit_y is None:
        unit_y = result.unit_system.get_parameter_unit(param_y)
    else:
        unit_y = normalize_unit(unit_y)

    prior_samples_x = result.unit_system.convert_from_canonical(param_x, sampled_priors[param_x], unit=unit_x)
    prior_samples_y = result.unit_system.convert_from_canonical(param_y, sampled_priors[param_y], unit=unit_y)

    if prior_samples_x is None or prior_samples_y is None:
        raise ValueError(f"Prior samples for {param_x} or {param_y} are not available.")

    fig, ax = plt.subplots()
    h = ax.hist2d(prior_samples_x, prior_samples_y, bins=bins, density=True, cmap='Blues')
    plt.colorbar(h[3], ax=ax)
    ax.set_xlabel(f'{param_x} [{unit_x.to_string("latex_inline")}]' if str(unit_x).strip() else param_x)
    ax.set_ylabel(f'{param_y} [{unit_y.to_string("latex_inline")}]' if str(unit_y).strip() else param_y)
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
    units = [results.unit_system.get_parameter_unit(name) for name in param_names]

    # Get Samples 
    sample_arrays = []
    for name, unit in zip(param_names, units):
        value = results[name, unit]
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
        prior_samples = results.unit_system.convert_from_canonical(name, priors[name], unit=units[i])
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
        ax.set_xlabel(f'{name} [{units[i].to_string("latex_inline")}]' if str(units[i]).strip() else name)
        ax.legend()

    for j in range(len(param_names), len(axes)):
        axes[j].axis('off')
    
    if savepath is not None:
        fig.savefig(f'{savepath}/posterior_over_prior.png',dpi=300)
        print(f"Saved posterior over prior plot to {savepath}")
    
    return fig


#---------------Orbit Visualization Plots ---------------------------------

def _pericenter_coords(model, system, ang: bool = False):
    if system == 'relative':
        system = ''
    params = model.derived_params
    alpha_peri = params[f"B{system}{'_ang' if ang else ''}"] * (1 - params['e'])
    delta_peri = params[f"A{system}{'_ang' if ang else ''}"] * (1 - params['e'])
    return alpha_peri, delta_peri

def orbit_plot(results, data, savepath=None):

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.05)
    
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    datas = [
        d for d in _flatten_joint(data)
        if isinstance(d, (AstrometryData, PhysicalAstrometryData, GaiaData))
    ]
    if not datas:
        raise ValueError("Multi-orbit plotting requires astrometry or Gaia data.")

    def _astrometry_data_for_system(system):
        return next(
            (d for d in datas if d.system == system and isinstance(d, (AstrometryData, PhysicalAstrometryData, GaiaData))),
            None,
        )
    

    def _plot_model(results,tfold,system):
        plot_data = _astrometry_data_for_system(system)
        if plot_data is None:
            return

        map_params = getattr(results, 'MAP_params', None)
        if map_params is None:
            map_params = results.samples.get('MAP_params', None)
        
        med_params = getattr(results, 'median_params', None)
        if med_params is None:
            med_params = results.samples.get('median_params', None)

        # map_params = results.canonical_MAP_params()
        # med_params = results.canonical_median_params()
        map_model = Orbit(**map_params)
        med_model = Orbit(**med_params)

        if isinstance(plot_data, (AstrometryData, GaiaData)):
            alpha_map, delta_map = map_model.pure_orbit_ang(tfold, system=system)
            alpha_med, delta_med = med_model.pure_orbit_ang(tfold, system=system)
            alpha_peri, delta_peri = _pericenter_coords(map_model, system, ang=True)
        elif isinstance(plot_data, PhysicalAstrometryData):
            alpha_map, delta_map = map_model.pure_orbit(tfold, system=system)
            alpha_med, delta_med = med_model.pure_orbit(tfold, system=system)
            alpha_peri, delta_peri = _pericenter_coords(map_model, system, ang=False)

        plot_t = _plot_time(plot_data, tfold)
        plot_alpha_map = _plot_position(plot_data, alpha_map)
        plot_alpha_med = _plot_position(plot_data, alpha_med)
        plot_delta_map = _plot_position(plot_data, delta_map)
        plot_delta_med = _plot_position(plot_data, delta_med)

        ax1.plot(plot_t, plot_alpha_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
        ax1.plot(plot_t, plot_alpha_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)
        ax2.plot(plot_t, plot_delta_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
        ax2.plot(plot_t, plot_delta_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)
        ax3.plot(plot_alpha_map, plot_delta_map, label='MAP Orbit', color='red', linestyle='-',zorder=3)
        ax3.plot(plot_alpha_med, plot_delta_med, label='Median Orbit', color='purple', linestyle='--',zorder=3)
        ax3.scatter(0, 0, color='k', marker='*', label='COM', zorder=10, s=100)
        ax3.plot(
            [0, _plot_position(plot_data, alpha_peri)],
            [0, _plot_position(plot_data, delta_peri)],
            color='gray', linestyle='--', label='Periapsis', zorder=4, alpha=0.6,
        )
     




    def _plot_astrometry_data(results,data):
        map_model = Orbit(**results.MAP_params)
        if isinstance(data, AstrometryData):
            model_alpha, model_delta = map_model.astrometry_ang(
                data.t_canonical, data.plxf_x, data.plxf_y, system=data.system
            )
            orbit_alpha, orbit_delta = map_model.pure_orbit_ang(data.t_canonical, system=data.system)
            alpha_obs = data.x_ang_canonical - (model_alpha - orbit_alpha)
            delta_obs = data.y_ang_canonical - (model_delta - orbit_delta)
            alpha_err = data.x_err_ang_canonical
            delta_err = data.y_err_ang_canonical
        elif isinstance(data, PhysicalAstrometryData):
            model_alpha, model_delta = map_model.astrometry(
                data.t_canonical, data.plxf_x, data.plxf_y, system=data.system
            )
            orbit_alpha, orbit_delta = map_model.pure_orbit(data.t_canonical, system=data.system)
            alpha_obs = data.x_canonical - (model_alpha - orbit_alpha)
            delta_obs = data.y_canonical - (model_delta - orbit_delta)
            alpha_err = data.x_err_canonical
            delta_err = data.y_err_canonical
        dt_obs = _plot_time(data, data.t_canonical)

        ax1.errorbar(dt_obs, _plot_position(data, alpha_obs), yerr=_plot_position(data, alpha_err),color='k', fmt='o',markersize=4,zorder=2)
        ax2.errorbar(dt_obs, _plot_position(data, delta_obs), yerr=_plot_position(data, delta_err),color='k', fmt='o',markersize=4,zorder=2)
        ax3.scatter(_plot_position(data, alpha_obs), _plot_position(data, delta_obs), color='k',s=15,zorder=2)
   

    def _plot_gaia_data(results,data):
        Map_plot_dict = data._astrometry(Orbit(**results.MAP_params))
        Med_plot_dict = data._astrometry(Orbit(**results.median_params))

        dt_obs = _plot_time(data, data.t_canonical)

        for i, (ri, di, ei, si, ci) in enumerate(zip(Map_plot_dict['ra_orb_obs'], Map_plot_dict['dec_orb_obs'], data.err_canonical, data.spsi, data.cpsi)):
            x0 = _plot_position(data, ri - ei * si)
            x1 = _plot_position(data, ri + ei * si)
            y0 = _plot_position(data, di - ei * ci)
            y1 = _plot_position(data, di + ei * ci)

            ax1.plot([dt_obs[i], dt_obs[i]], [x0, x1], color='tab:orange', alpha=0.5, zorder=2)
            ax2.plot([dt_obs[i], dt_obs[i]], [y0, y1], color='tab:orange', alpha=0.5, zorder=2)
            ax3.plot([x0, x1], [y0, y1], color='tab:orange', alpha=0.5, zorder=2)

        ax1.scatter(dt_obs, _plot_position(data, Map_plot_dict['ra_orb_obs']), color='k', s=15, zorder=3)
        ax2.scatter(dt_obs, _plot_position(data, Map_plot_dict['dec_orb_obs']), color='k', s=15, zorder=3)
        ax3.scatter(_plot_position(data, Map_plot_dict['ra_orb_obs']), _plot_position(data, Map_plot_dict['dec_orb_obs']), color='k', s=15, zorder=3)
        ax3.plot([0, _plot_position(data, Map_plot_dict['ra_peri'])], [0, _plot_position(data, Map_plot_dict['dec_peri'])], color='gray', linestyle='--', label='Periapsis', zorder=4, alpha=0.6)
        ax3.scatter(0, 0, color='k', marker='*', label='COM', zorder=10, s=100)
        

    systems = {d.system for d in datas}
    min_t = []
    max_t = []
    for d in datas:
        min_t.append(np.min(d.t_canonical))
        max_t.append(np.max(d.t_canonical))
    tfold = np.linspace(np.min(min_t), np.max(max_t), 1000)
    for s in systems:
        _plot_model(results,tfold,s)
    for d in datas:
        if d.system not in systems:
            continue
        if isinstance(d, AstrometryData) or isinstance(d, PhysicalAstrometryData):
            _plot_astrometry_data(results,d)
        elif isinstance(d, GaiaData):
            _plot_gaia_data(results,d)

    label_data = next((d for d in datas if isinstance(d, (AstrometryData, GaiaData))), datas[0])
    position_unit = _plot_unit(label_data, 'x')
    ax2.set_xlabel(f'Time [{_plot_unit(label_data, "t")}]')
    ax1.set_ylabel(rf"$\Delta \alpha^*$ [{position_unit}]")
    ax2.set_ylabel(rf"$\Delta \delta$ [{position_unit}]")
    ax3.set_xlabel(rf"$\Delta \alpha^*$ [{position_unit}]")
    ax3.set_ylabel(rf"$\Delta \delta$ [{position_unit}]")
    ax3.set_aspect('equal',adjustable = 'datalim')
    #Deduplicates labels if multiple systems are plotted
    handles, labels = ax3.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax3.legend(unique_legend.values(), unique_legend.keys(),fontsize='small',loc='best')
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

    def _plot_lin_model(results, data, tfold):

        null_fit = results.null_hypothesis

        if isinstance(data, PhysicalAstrometryData):
            dalpha = null_fit['params']['dalpha']
            ddelta = null_fit['params']['ddelta']
            mu_alpha = null_fit['params']['mu_alpha']
            mu_delta = null_fit['params']['mu_delta']
            parallax_value = null_fit['params'].get('parallax', 0.0)
            parallax = parallax_value
        elif isinstance(data, AstrometryData):
            dalpha = null_fit['params']['dalpha_ang']
            ddelta = null_fit['params']['ddelta_ang']
            mu_alpha = null_fit['params']['mu_alpha_ang']
            mu_delta = null_fit['params']['mu_delta_ang']
            parallax_value = null_fit['params'].get('parallax', 0.0)
            parallax = null_fit['params'].get('parallax', 0.0)

        if isinstance(data, GaiaData):
            tfold = np.linspace(data.t_canonical.min(), data.t_canonical.max(), 1000)
            dt = tfold - results.canonical_MAP_params().get('Tepoch', 0)
            plx_alpha = np.interp(tfold,data.t_canonical,data.plx_fac*data.spsi)
            plx_delta = np.interp(tfold,data.t_canonical,data.plx_fac*data.cpsi)

            alpha_lin = mu_alpha * dt + dalpha + plx_alpha*parallax
            delta_lin = mu_delta * dt + ddelta + plx_delta*parallax
            ax.plot(_plot_position(data, alpha_lin), _plot_position(data, delta_lin), label='Linear Model', color='orange', linestyle='-.', zorder=1, alpha=0.7)

            map_plot_dict = data._astrometry(Orbit(**results.canonical_MAP_params()))
            med_plot_dict = data._astrometry(Orbit(**results.canonical_median_params()))
            ax.plot(_plot_position(data, map_plot_dict['ra_sky']), _plot_position(data, map_plot_dict['dec_sky']), label='MAP Sky Track', color='red', linestyle='-', zorder=1)
            ax.plot(_plot_position(data, med_plot_dict['ra_sky']), _plot_position(data, med_plot_dict['dec_sky']), label='Median Sky Track', color='purple', linestyle='-', zorder=1)

        if isinstance(data,AstrometryData) or isinstance(data, PhysicalAstrometryData):
            plxf_x = 0.0 if data.plxf_x is None else data.plxf_x
            plxf_y = 0.0 if data.plxf_y is None else data.plxf_y
            dt = tfold - results.MAP_params.get('Tepoch', 0)
            alpha_lin = mu_alpha * dt + dalpha + plxf_x*parallax
            delta_lin = mu_delta * dt + ddelta + plxf_y*parallax
            ax.plot(_plot_position(data, alpha_lin), _plot_position(data, delta_lin), label='Linear Model', color='orange', linestyle='-.', zorder=1, alpha=0.7)

            map_model = Orbit(**results.MAP_params)
            med_model = Orbit(**results.median_params)
            if isinstance(data, PhysicalAstrometryData):
                alpha_map, delta_map = map_model.astrometry(tfold, data.plxf_x, data.plxf_y, system=data.system)
                alpha_med, delta_med = med_model.astrometry(tfold, data.plxf_x, data.plxf_y, system=data.system)
            elif isinstance(data, AstrometryData):
                alpha_map, delta_map = map_model.astrometry_ang(tfold, data.plxf_x, data.plxf_y, system=data.system)
                alpha_med, delta_med = med_model.astrometry_ang(tfold, data.plxf_x, data.plxf_y, system=data.system)
            ax.plot(_plot_position(data, alpha_map), _plot_position(data, delta_map), label='MAP Sky Track', color='red', linestyle='-', zorder=3)
            ax.plot(_plot_position(data, alpha_med), _plot_position(data, delta_med), label='Median Sky Track', color='purple', linestyle='-', zorder=3)


    def _plot_data(results, data):
        if isinstance(data, GaiaData):
            map_plot_dict = data._astrometry(Orbit(**results.canonical_MAP_params()))
            ax.scatter(_plot_position(data, map_plot_dict['ra_sky_data']), _plot_position(data, map_plot_dict['dec_sky_data']), color='k', s=15, zorder=1)
            
        elif isinstance(data, AstrometryData):
            ax.scatter(_plot_position(data, data.x_ang_canonical), _plot_position(data, data.y_ang_canonical), color='k', s=15, zorder=1)

        elif isinstance(data, PhysicalAstrometryData):
            ax.scatter(_plot_position(data, data.x_canonical), _plot_position(data, data.y_canonical), color='k', s=15, zorder=1)
            


    datas = _flatten_joint(data)
    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t_canonical))
        t_max.append(np.max(d.t_canonical))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        _plot_lin_model(results, d,tfold)
        _plot_data(results, d)

    ax.legend(loc='best')
    label_data = next((d for d in datas if isinstance(d, (AstrometryData, GaiaData))), datas[0])
    ax.set_xlabel(rf"$\Delta \alpha^*$ [{_plot_unit(label_data, 'x')}]")
    ax.set_ylabel(rf"$\Delta \delta$ [{_plot_unit(label_data, 'x')}]")
    ax.set_aspect('equal', adjustable='datalim')
    ax.invert_xaxis()
    
    if savepath is not None:
        fig.savefig(f'{savepath}/sky_motion.png', dpi=300)
        print(f"Saved sky motion plot to {savepath}")

    return fig

def astrometry_phase_plot(results, data, savepath=None):
    '''
    Plots astrometry data in phase space
    '''
    if isinstance(data, JointData):
        datas = [d for d in _flatten_joint(data) if isinstance(d, (AstrometryData, PhysicalAstrometryData))]
    elif isinstance(data, (AstrometryData, PhysicalAstrometryData)):
        datas = [data]
    elif isinstance(data, (RadialVelocityData, GaiaData)):
        return None
    else:
        datas = []

    if not datas:
        return None

    fig, ax = plt.subplots(2, 1, sharex=True)
    map_model = Orbit(**results.MAP_params)
    period = map_model['P']
    periastron_time = map_model['Tp']

    def plot_system(system, system_datas):
        time_fold = np.linspace(
            periastron_time - 0.5 * period,
            periastron_time + 0.5 * period,
            10000,
        )
        phase_fold = (time_fold - periastron_time + 0.5 * period) / period
        sort_map = np.argsort(phase_fold)

        if isinstance(system_datas[0], AstrometryData):
            alpha_model, delta_model = map_model.pure_orbit_ang(time_fold, system=system)
        elif isinstance(system_datas[0], PhysicalAstrometryData):
            alpha_model, delta_model = map_model.pure_orbit(time_fold, system=system)

        ax[0].plot(
            phase_fold[sort_map],
            _plot_position(system_datas[0], alpha_model)[sort_map],
            label=f"MAP Fit ({system})", color='red', linestyle='-', linewidth=2,
        )
        ax[1].plot(
            phase_fold[sort_map],
            _plot_position(system_datas[0], delta_model)[sort_map],
            label=f"MAP Fit ({system})", color='red', linestyle='-', linewidth=2,
        )

        alpha_resids = []
        delta_resids = []
        alpha_errs = []
        delta_errs = []
        phases = []


        for d in system_datas:
            if isinstance(d, AstrometryData):
                model_alpha, model_delta = map_model.astrometry_ang(
                    d.t_canonical, d.plxf_x, d.plxf_y, system=d.system
                )
                orbit_alpha, orbit_delta = map_model.pure_orbit_ang(
                    d.t_canonical, system=d.system
                )
                alpha_data = d.x_ang_canonical - (model_alpha - orbit_alpha)
                delta_data = d.y_ang_canonical - (model_delta - orbit_delta)
                alpha_err = d.x_err_ang_canonical
                delta_err = d.y_err_ang_canonical
            elif isinstance(d, PhysicalAstrometryData):
                model_alpha, model_delta = map_model.astrometry(
                    d.t_canonical, d.plxf_x, d.plxf_y, system=d.system
                )
                orbit_alpha, orbit_delta = map_model.pure_orbit(
                    d.t_canonical, system=d.system
                )
                alpha_data = d.x_canonical - (model_alpha - orbit_alpha)
                delta_data = d.y_canonical - (model_delta - orbit_delta)
                alpha_err = d.x_err_canonical
                delta_err = d.y_err_canonical
            phase = (d.t_canonical - periastron_time + 0.5 * period) / period % 1

            alpha_resids.append(_plot_position(d, alpha_data))
            delta_resids.append(_plot_position(d, delta_data))
            alpha_errs.append(_plot_position(d, alpha_err))
            delta_errs.append(_plot_position(d, delta_err))
            phases.append(phase)

        alpha_resids = np.concatenate(alpha_resids)
        delta_resids = np.concatenate(delta_resids)
        alpha_errs = np.concatenate(alpha_errs)
        delta_errs = np.concatenate(delta_errs)
        phases = np.concatenate(phases)

        points_per_bin = 20
        minimum_bins = 4
        if len(phases) < points_per_bin * minimum_bins:
            ax[0].errorbar(
                phases,
                alpha_resids,
                yerr=alpha_errs,
                fmt='o', color='blue', alpha=0.3, markersize=3, zorder=1,
            )
            ax[1].errorbar(
                phases,
                delta_resids,
                yerr=delta_errs,
                fmt='o', color='blue', alpha=0.3, markersize=3, zorder=1,
            )
        else:
            sort = np.argsort(phases)
            phases_sorted = phases[sort]
            alpha_resids_sorted = alpha_resids[sort]
            delta_resids_sorted = delta_resids[sort]
            alpha_errs_sorted = alpha_errs[sort]
            delta_errs_sorted = delta_errs[sort]

            num_bins = int(np.ceil(len(phases) / points_per_bin))
            b_edges = np.linspace(0, 1, num_bins + 1)
            b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
            alpha_binned = []
            delta_binned = []
            alpha_err_binned = []
            delta_err_binned = []
            center = []
            for i in range(num_bins):
                good = (phases_sorted >= b_edges[i]) & (phases_sorted < b_edges[i + 1])
                if np.sum(good) > 0:

                    w_a = 1/(alpha_errs_sorted[good]**2)
                    w_d = 1/(delta_errs_sorted[good]**2)
                    mean_a = np.sum(w_a * alpha_resids_sorted[good]) / np.sum(w_a)
                    mean_d = np.sum(w_d * delta_resids_sorted[good]) / np.sum(w_d)
                    err_a = np.sqrt(1/np.sum(w_a))
                    err_d = np.sqrt(1/np.sum(w_d))

                    alpha_binned.append(mean_a)
                    delta_binned.append(mean_d)
                    alpha_err_binned.append(err_a)
                    delta_err_binned.append(err_d)
                    center.append(b_centers[i])

            ax[0].errorbar(
                center,
                alpha_binned,
                yerr=alpha_err_binned,
                fmt='s', color='orange', markersize=5, zorder=3,
                label=f'Binned Data ({system})',
            )
            ax[1].errorbar(
                center,
                delta_binned,
                yerr=delta_err_binned,
                fmt='s', color='orange', markersize=5, zorder=3,
                label=f'Binned Data ({system})',
            )

    for system in sorted({d.system for d in datas}):
        plot_system(system, [d for d in datas if d.system == system])

    ax[0].set_ylabel(rf"RA [{_plot_unit(datas[0], 'x')}]" )
    ax[1].set_ylabel(rf"Dec [{_plot_unit(datas[0], 'y')}]" )
    ax[1].set_xlabel("Phase")
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')

    if savepath is not None:
        fig.savefig(f'{savepath}/astrometry_phase.png', dpi=300)
        print(f"Saved astrometry phase plot to {savepath}")

    return fig

def multi_orbit_plot(results, data, Nplot=100, savepath=None):
    '''
    Plots multiple orbits from the posterior samples
    '''
    fig,ax = plt.subplots()

    def _plot_orbit_samples(results, data, Nplot,tfold):
        param_names = results.param_names
        if not param_names:
            raise ValueError("Posterior samples are not available for multi-orbit plotting.")

        samples = results.samples

        idx = np.random.choice(samples[param_names[0]].shape[0], size=min(Nplot, samples[param_names[0]].shape[0]), replace=False)
        samps = {
            name: samples[name][idx] for name in param_names
        }

        fixed_prior_values = {name: p.value for name, p in results.canonical_priors.items() if isinstance(p, FixedPrior)}
        for i in range(len(idx)):
            samp = {name: samps[name][i] for name in param_names}
            model = Orbit(**samp, **fixed_prior_values)
            if isinstance(data, AstrometryData):
                alpha, delta = model.pure_orbit_ang(tfold, system=data.system)
            elif isinstance(data, PhysicalAstrometryData):
                alpha,delta = model.pure_orbit(tfold, system=data.system)
            ax.plot(_plot_position(data, alpha), _plot_position(data, delta), color='tab:blue', alpha=0.3, zorder=1)

    def _plot_map_median_orbits(results, data, tfold):

        map_model = Orbit(**results.MAP_params)
        med_model = Orbit(**results.median_params)

        if isinstance(data, AstrometryData):
            alpha_map, delta_map = map_model.pure_orbit_ang(tfold, system=data.system)
            alpha_med, delta_med = med_model.pure_orbit_ang(tfold, system=data.system)
        elif isinstance(data, PhysicalAstrometryData):
            alpha_map, delta_map = map_model.pure_orbit(tfold, system=data.system)
            alpha_med, delta_med = med_model.pure_orbit(tfold, system=data.system)

        ax.plot(_plot_position(data, alpha_map), _plot_position(data, delta_map), label='MAP Orbit', color='red', linestyle='-', zorder=4)
        ax.plot(_plot_position(data, alpha_med), _plot_position(data, delta_med), label='Median Orbit', color='purple', linestyle='--', zorder=4)
        ax.scatter(0, 0, color='k', marker='*', label='COM', zorder=10)


    datas = _flatten_joint(data)
    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t_canonical))
        t_max.append(np.max(d.t_canonical))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        if not isinstance(d, (AstrometryData, PhysicalAstrometryData, GaiaData)):
            continue
        _plot_orbit_samples(results, d, Nplot, tfold)
        _plot_map_median_orbits(results, d, tfold)

    
    label_data = next((d for d in datas if isinstance(d, (AstrometryData, GaiaData))), datas[0])
    position_unit = _plot_unit(label_data, 'x')
    ax.set_xlabel(rf"$\Delta \alpha^*$ [{position_unit}]")
    ax.set_ylabel(rf"$\Delta \delta$ [{position_unit}]")
    ax.legend(fontsize='small', loc='best')
    ax.set_aspect('equal')
    ax.invert_xaxis()

    if savepath is not None:
        fig.savefig(f'{savepath}/multi_orbit.png', dpi=300)
        print(f"Saved multi-orbit plot to {savepath}")

    return fig

def residuals_plot(results, data, savepath=None):
    '''
    Plots residuals of data from the MAP model
    '''
    
    datas = _flatten_joint(data)
    for d in datas:
        if isinstance(d,(AstrometryData, PhysicalAstrometryData)):
            fig, ax = plt.subplots(2, 1, sharex=True)
            map_model = Orbit(**results.MAP_params)
            median_model = Orbit(**results.median_params)
            if isinstance(d, AstrometryData):
                model_alpha, model_delta = map_model.astrometry_ang(d.t_canonical, d.plxf_x, d.plxf_y, system=d.system)
                median_alpha, median_delta = median_model.astrometry_ang(d.t_canonical, d.plxf_x, d.plxf_y, system=d.system)
                alpha_resids = d.x_ang_canonical - model_alpha
                delta_resids = d.y_ang_canonical - model_delta
                median_alpha_resids = d.x_ang_canonical - median_alpha
                median_delta_resids = d.y_ang_canonical - median_delta
                alpha_err = d.x_err_ang_canonical
                delta_err = d.y_err_ang_canonical
            elif isinstance(d, PhysicalAstrometryData):
                model_alpha, model_delta = map_model.astrometry(d.t_canonical, d.plxf_x, d.plxf_y, system=d.system)
                median_alpha, median_delta = median_model.astrometry(d.t_canonical, d.plxf_x, d.plxf_y, system=d.system)
                alpha_resids = d.x_canonical - model_alpha
                delta_resids = d.y_canonical - model_delta
                median_alpha_resids = d.x_canonical - median_alpha
                median_delta_resids = d.y_canonical - median_delta
                alpha_err = d.x_err_canonical
                delta_err = d.y_err_canonical

            dt_obs = _plot_time(d, d.t_canonical)

            ax[0].errorbar(dt_obs, _plot_position(d, alpha_resids), yerr=_plot_position(d, alpha_err), color='tab:blue', fmt='o',markersize=4,label=f'RA MAP Residuals',zorder=2)
            ax[1].errorbar(dt_obs, _plot_position(d, delta_resids), yerr=_plot_position(d, delta_err), color='tab:blue', fmt='o',markersize=4,label=f'Dec MAP Residuals',zorder=2)
            ax[0].plot(dt_obs, _plot_position(d, median_alpha_resids), color='tab:purple', marker='s', linestyle='', markersize=3, label='RA Median Residuals', zorder=2)
            ax[1].plot(dt_obs, _plot_position(d, median_delta_resids), color='tab:purple', marker='s', linestyle='', markersize=3, label='Dec Median Residuals', zorder=2)
            ax[0].set_ylabel(rf"Residuals [{_plot_unit(d, 'x')}]")
            ax[1].set_ylabel(rf"Residuals [{_plot_unit(d, 'y')}]")
            ax[1].set_xlabel(rf"Time [{_plot_unit(d, 't')}]")
            ax[0].axhline(0, color='red', linestyle='--')
            ax[1].axhline(0, color='red', linestyle='--')
            ax[0].legend(loc='best')
            ax[1].legend(loc='best')

            if savepath is not None:
                fig.savefig(f'{savepath}/residuals_plot_{d.system}_astrometry.png', dpi=300)
                print(f"Saved residuals plot for system {d.system} to {savepath}")
            return fig

        if isinstance(d,RadialVelocityData):
            fig, ax = plt.subplots()
            map_model = Orbit(**results.MAP_params)
            median_model = Orbit(**results.median_params)
            model_rv = map_model.rv(d.t_canonical, system=d.system)
            median_rv = median_model.rv(d.t_canonical, system=d.system)
            rv_resids = d.rv_canonical - model_rv
            median_rv_resids = d.rv_canonical - median_rv
            dt_obs = _plot_time(d, d.t_canonical)

            ax.errorbar(dt_obs, _plot_velocity(d, rv_resids), yerr=_plot_velocity(d, d.rv_err_canonical),color='tab:blue', fmt='o',markersize=4,label=f'RV MAP Residuals',zorder=2)
            ax.plot(dt_obs, _plot_velocity(d, median_rv_resids), color='tab:purple', marker='s', linestyle='', markersize=3, label='RV Median Residuals', zorder=2)
            ax.set_ylabel(rf"Residuals [{_plot_unit(d, 'rv')}]")
            ax.set_xlabel(rf"Time [{_plot_unit(d, 't')}]")
            ax.axhline(0, color='red', linestyle='--')
            ax.legend(loc='best')

            if savepath is not None:
                fig.savefig(f'{savepath}/residuals_plot_{d.system}_rv.png', dpi=300)
                print(f"Saved residuals plot for system {d.system} to {savepath}")
            return fig
        
        if isinstance(d, (GaiaData)):
            fig, ax = plt.subplots()
            map_model = Orbit(**results.MAP_params)
            median_model = Orbit(**results.median_params)
            x_model = map_model.gaia_astrometry(d.t_canonical, d.spsi,d.cpsi,d.plx_fac, system=d.system)
            median_x_model = median_model.gaia_astrometry(d.t_canonical, d.spsi,d.cpsi,d.plx_fac, system=d.system)
            x_resids = d.x_canonical - x_model
            median_x_resids = d.x_canonical - median_x_model
            dt_obs = _plot_time(d, d.t_canonical)

            ax.errorbar(dt_obs, _plot_position(d, x_resids), yerr=_plot_position(d, d.err_canonical),color='tab:blue', fmt='o',markersize=4,label=f'Gaia MAP Residuals',zorder=2)
            ax.plot(dt_obs, _plot_position(d, median_x_resids), color='tab:purple', marker='s', linestyle='', markersize=3, label='Gaia Median Residuals', zorder=2)    
            ax.set_ylabel(rf"Residuals [{_plot_unit(d, 'x')}]")
            ax.set_xlabel(rf"Time [{_plot_unit(d, 't')}]")
            ax.axhline(0, color='red', linestyle='--')
            ax.legend(loc='best')

        if savepath is not None:
            fig.savefig(f'{savepath}/residuals_plot_{d.system}_gaia.png', dpi=300)
            print(f"Saved residuals plot for system {d.system} to {savepath}")

        return fig



def distribution(results, param, scale='linear', unit=None, savepath=None):
    '''
    Plots distribution of a parameter from posterior samples
    '''
    try:
        unit = results.unit_system.get_parameter_unit(param) if unit is None else normalize_unit(unit)
        samples = results[param, unit]
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
                    label=fr'$1\,\sigma$  [{p16:.2f},{p84:.2f}] {unit.to_string("latex_inline")}')
        
        ax.axvspan(par_m2sig,par_p2sig,color='tab:blue',
            alpha=0.25,
            label=fr'$2\,\sigma$  [{par_m2sig:.2f},{par_p2sig:.2f}] {unit.to_string("latex_inline")}')

        ax.axvline(med,color='k',linestyle='--'
            ,label=fr'Median = {med:.2f} {unit.to_string("latex_inline")}')

        ax.set_xlabel(f"${par.display_latex(param)}$ [{unit.to_string('latex_inline')}]")
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
                    label=fr'$1\,\sigma$  [{p16:.2f},{p84:.2f}] {unit.to_string("latex_inline")}')
        
        ax.axvspan(par_m2sig,par_p2sig,color='tab:blue',
            alpha=0.25,
            label=fr'$2\,\sigma$  [{par_m2sig:.2f},{par_p2sig:.2f}] {unit.to_string("latex_inline")}')
        
        ax.axvline(med,color='k',linestyle='--'
            ,label=fr'Median = {med:.2f} {unit.to_string("latex_inline")}')
        

        ax.set_xlim(par_m2sig*0.8, par_p2sig*1.2)
        ax.set_xscale('log')
        ax.set_xlabel(f"{param} [{unit.to_string('latex_inline')}]")
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
    if 'M2' in results and len(results['M2']) > 1 and len(results['M2']) > 1:
        return distribution(results, 'M2', scale=scale, unit=r'Msun', savepath=savepath)
    elif 'M1' in results and len(results['M1']) > 1 and len(results['M1']) > 1:
        return distribution(results, 'M1', scale=scale, unit=r'Msun', savepath=savepath)
    elif 'minM2' in results and len(results['minM2']) > 1 and len(results['minM2']) > 1:
        return distribution(results, 'minM2', scale=scale, unit=r'Msun', savepath=savepath)
    elif 'minM1' in results and len(results['minM1']) > 1 and len(results['minM1']) > 1:
        return distribution(results, 'minM1', scale=scale, unit=r'Msun', savepath=savepath)
    else:
        print('No mass samples found in results.')
        return None


def rv_fit_plot(results, data, savepath=None):
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

    def _plot_rv_fit(results,data,system,tfold):
        Map_fit = Orbit(**results.MAP_params)
        med_fit = Orbit(**results.median_params)

        ax.plot(_plot_time(data, tfold), _plot_velocity(data, Map_fit.rv(tfold,system)), label='MAP Fit', color='red', linestyle='-')
        ax.plot(_plot_time(data, tfold), _plot_velocity(data, med_fit.rv(tfold,system)), label='Median Fit',color='purple', linestyle='--',alpha=0.7)
        ax.errorbar(_plot_time(data, data.t_canonical), _plot_velocity(data, data.rv_canonical), yerr=_plot_velocity(data, data.rv_err_canonical), fmt='o', color='k', markersize=4)
        

    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t_canonical))
        t_max.append(np.max(d.t_canonical))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        _plot_rv_fit(results,d,d.system,tfold)
    
    ax.set_xlabel(f'Time [{_plot_unit(datas[0], "t")}]')
    ax.set_ylabel(f'RV [{_plot_unit(datas[0], "rv")}]')
    ax.legend(loc='upper right')

    if savepath is not None:
        fig.savefig(f'{savepath}/rv_fit.png', dpi=300)
        print(f"Saved radial velocity fit plot to {savepath}")
    return fig


def phase_fold_rv_plot(results, data, savepath=None):
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
            _plot_velocity(system_datas[0], map_model.rv(tfold, system))[sort_map],
            label=f'MAP Fit ({system})',
            color='red',
            lw=1.5,
            linestyle='-'
        )

        for d in system_datas:
            phase = (d.t_canonical - map_tp + 0.5 * map_period) / map_period % 1
            ax.errorbar(phase, _plot_velocity(d, d.rv_canonical), yerr=_plot_velocity(d, d.rv_err_canonical), fmt='o', color='k', markersize=4)

    for system in sorted({d.system for d in datas}):
        system_datas = [d for d in datas if d.system == system]
        plot_system(system, system_datas)

    ax.set_xlabel('Phase')
    ax.set_ylabel(f'RV [{_plot_unit(datas[0], "rv")}]')
    ax.legend(loc='best')

    if savepath is not None:
        fig.savefig(f'{savepath}/phase_fold_rv.png', dpi=300)
        print(f"Saved phase-folded radial velocity plot to {savepath}")

    return fig

def rv_multi_fit_plot(results,data,Nplot=100,savepath=None):
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
    if not param_names:
        raise ValueError("Posterior samples are not available for multi-orbit plotting.")
    samples = results.samples
    
    idx = np.random.choice(samples[param_names[0]].shape[0], size=min(Nplot, samples[param_names[0]].shape[0]), replace=False)
    samps = {
        name: samples[name][idx] for name in param_names
    }

    map_params = results.MAP_params
    med_params = results.median_params

    fixed_prior_params = {
        name: map_params[name]
        for name, prior in results.priors.items()
        if isinstance(prior, FixedPrior) and name in map_params
    }

    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    fig,ax = plt.subplots()

    def _multi_rvs(Nplot,tfold,system):
        plot_data = next(d for d in datas if d.system == system)
        
        for i in range(len(idx)):
            samp = {name: samps[name][i] for name in param_names}
            model = Orbit(**samp, **fixed_prior_params)
            ax.plot(_plot_time(plot_data, tfold), _plot_velocity(plot_data, model.rv(tfold,system)), color='tab:blue', alpha=0.3)

        ax.plot(_plot_time(plot_data, tfold), _plot_velocity(plot_data, map_model.rv(tfold,system)), label='MAP Fit', color='red', linestyle='-')
        ax.plot(_plot_time(plot_data, tfold), _plot_velocity(plot_data, med_model.rv(tfold,system)), label='Median Fit', color='purple', linestyle='--', alpha=0.7)

    def _plot_data(data):
        ax.errorbar(_plot_time(data, data.t_canonical), _plot_velocity(data, data.rv_canonical), yerr=_plot_velocity(data, data.rv_err_canonical), fmt='o', color='k', markersize=4)

    systems = {d.system for d in datas}
    t_min = []
    t_max = []
    for d in datas:
        if d.system not in systems:
            continue
        t_min.append(np.min(d.t_canonical))
        t_max.append(np.max(d.t_canonical))
    tfold = np.linspace(np.min(t_min), np.max(t_max), 1000)
    for d in datas:
        _multi_rvs(Nplot,tfold,d.system)
        _plot_data(d)

    ax.set_xlabel(f'Time [{_plot_unit(datas[0], "t")}]')
    ax.set_ylabel(f'RV [{_plot_unit(datas[0], "rv")}]')
    ax.legend(loc='best')

    if savepath is not None:
        fig.savefig(f'{savepath}/rv_multi_fit.png', dpi=300)
        print(f"Saved multi-radial velocity fit plot to {savepath}")
    return fig


def multi_phase_plot(results,data,Nplot=100,savepath=None):
    '''
    Plots multiple phase-folded RV fits
    '''
    if isinstance(data, JointData):
        datas = [d for d in data.datas if isinstance(d, RadialVelocityData)]
    else:
        datas = [data] if isinstance(data, RadialVelocityData) else []

    param_names = results.param_names
    if not param_names:
        raise ValueError("Posterior samples are not available for multi-orbit plotting.")

    samples = results.samples
        
    idx = np.random.choice(samples[param_names[0]].shape[0], size=min(Nplot, samples[param_names[0]].shape[0]), replace=False)
    samps = {
        name: samples[name][idx] for name in param_names
    }  
    
    map_params = results.MAP_params
    med_params = results.median_params
    
    fixed_prior_params = {
        name: map_params[name]
        for name, prior in results.priors.items()
        if isinstance(prior, FixedPrior) and name in map_params
    }
    
    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    if not datas:
        print("No radial velocity data found for plotting.")
        return None

    fig, ax = plt.subplots()

    def _multi_phase(Nplot, system):
        plot_data = next(d for d in datas if d.system == system)
        for i in range(len(idx)):
            samp = {name: samps[name][i] for name in param_names}
            model = Orbit(**samp, **fixed_prior_params)
            P_samp = model['P']
            Tp_samp = model['Tp']
            tfold_samp = np.linspace(Tp_samp - 0.5*P_samp, Tp_samp + 0.5*P_samp, 10000)
            phase_fold_samp = (tfold_samp - Tp_samp + 0.5*P_samp) / P_samp
            rv_samp = model.rv(tfold_samp, system)
            ax.plot(phase_fold_samp, _plot_velocity(plot_data, rv_samp), color='tab:blue', alpha=0.3)


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

        plot_data = next(d for d in datas if d.system == system)
        ax.plot(phase_fold_map, _plot_velocity(plot_data, rv_map), label='MAP Fit', color='red', lw=1.5, linestyle='-')
        ax.plot(phase_fold_med, _plot_velocity(plot_data, rv_med), label='Median Fit', color='purple', lw=1.5, linestyle='--', alpha=0.7)

    def _plot_data(data):
        P = map_model['P']
        Tp = map_model['Tp']
        phase = (data.t_canonical - Tp + 0.5*P) / P % 1
        ax.errorbar(phase, _plot_velocity(data, data.rv_canonical), yerr=_plot_velocity(data, data.rv_err_canonical), fmt='o', color='k', markersize=4)

    systems = {d.system for d in datas}
    for system in sorted(systems):
        system_datas = [d for d in datas if d.system == system]
        _multi_phase(Nplot, system)
        _map_median_phase(system)
        for d in system_datas:
            _plot_data(d)

    ax.set_xlabel('Phase')
    ax.set_ylabel(f'RV [{_plot_unit(datas[0], "rv")}]')
    ax.legend(loc='best')

    if savepath is not None:
        fig.savefig(f'{savepath}/multi_phase.png', dpi=300)
        print(f"Saved multi-phase-folded radial velocity plot to {savepath}")
    return fig

        
def all_plots(results, data: Data, scale=None, savepath=None):
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
        rv_fit_plot(results,data,savepath=savepath)
        phase_fold_rv_plot(results,data,savepath=savepath)
        rv_multi_fit_plot(results,data,savepath=savepath)
        multi_phase_plot(results,data,savepath=savepath)
    if data.has_astrometry():
        orbit_plot(results,data,savepath=savepath)
        sky_motion_plot(results,data,savepath=savepath)
        multi_orbit_plot(results,data,savepath=savepath)
        astrometry_phase_plot(results,data,savepath=savepath)
    residuals_plot(results,data,savepath=savepath)
    mass_distribution(results,scale=scale,savepath=savepath)


