if __name__ == '__main__':
    # I have to do this __name__ == '__main__' to run the code with multiprocessing.
    # However, I am running this file on a Mac with silicon chip; that may be the reason why I needed to do this
    # If you are running this file on a non-mac-silicon machine, it is worth trying to run without __name__ == '__main__'
    import numpy as np
    import matplotlib.pyplot as plt

    import jax
    jax.config.update(
        "jax_enable_x64", True
    )  #

    from kelp.jax import reflected_phase_curve
    import batman
    import juliet
    import os
    from exotoolbox import utils
    import corner
    import matplotlib.gridspec as gd

    np.random.seed(9)

    # Kelp homogeneous reflection phase curve model simulation for 1/2 instrument
    # Set this variable to True if you want to fit the same phase curve model to both instruments
    # Set this variable to False if you want to fit the diff phase curve model to both instruments
    # For single instrument, simply put name of only one instrument in instruments array
    SamePC_2Inst = False
    kelp_knots = None                        # To use interpolation in kelp
    instruments = ['CHEOPS1', 'CHEOPS2']
    pout = 'kelphomo_1Inst'
    if SamePC_2Inst:
        ## Instrument 1 and 2 both have same phase curve parameters (single scattering albedo and scattering asymmetry factor)
        single_scat_w = [0.85, 0.85]
        g = [0.5, 0.5]
    else:
        ## Instrument 1 and 2 have different phase curve parameters (single scattering albedo and scattering asymmetry factor)
        single_scat_w = [0.85, 0.90]
        g = [0.6, 0.3]

    # ---------------------------------------------------------------------
    #
    #                    Generating simulated data
    #
    # ---------------------------------------------------------------------

    # Planetary parameters are for WASP-189 b (Deline et al. 2022)

    per, per_err = 2.724035, np.sqrt((0.000022**2) + (0.000023**2))
    tc, tc_err = 2459016.434866, 0.000060
    bb, bb_err = 0.433, np.sqrt(0.014**2 + 0.015**2)
    inc, inc_err = 84.58, np.sqrt(0.23**2 + 0.22**2)
    rprs, rprs_err = 0.06958, 0.00016
    rst = 2.365
    ar, ar_err = 2.587, np.sqrt(0.037**2 + 0.034**2)
    q1, q2 = utils.convert_ld_coeffs('quadratic', 0.1, 0.3)

    tsecondary = tc + (per/2)

    sigma = 100e-6

    # Batman transit model
    pars = batman.TransitParams()
    pars.per = per
    pars.t0 = tc
    pars.a = ar
    pars.rp = rprs
    pars.inc = np.rad2deg(np.arccos(bb/ar))
    pars.ecc = 0.
    pars.w = 0.
    pars.u = [0.1, 0.3]
    pars.limb_dark = 'quadratic'

    # ------------- Full dataset
    ## Making the dataset: Full Dataset
    tim, fl, fle = {}, {}, {}
    total_models = {}
    for ins in range(len(instruments)):

        # Simulated time
        times = np.linspace(tc-(0.65*per), tc+(0.65*per), 10000) + ( ((ins+1)**3) * per)

        m1 = batman.TransitModel(pars, times)
        flx_tra = m1.light_curve(pars)

        # Batman occultation model
        pars.fp = 100e-6
        pars.t_secondary = tsecondary
        m2 = batman.TransitModel(params=pars, t=times, transittype='secondary')
        flx_ecl = m2.light_curve(pars)

        # kelp homogeneous reflective planet phase curve model
        phases_unsorted = ((times- tc) % per) / per                       ## Un-sorted phases
        idx_phase_sort = np.argsort(phases_unsorted)                      ## This would sort any array acc to phase
        phases_sorted = phases_unsorted[idx_phase_sort]                   ## Sorted phase array
        times_sorted_acc_phs = times[idx_phase_sort]                      ## Time array sorted acc to phase
        idx_that_sort_arr_acc_times = np.argsort(times_sorted_acc_phs)    ## This array would sort array acc to time

        # Reflective phase curve (homogeneous)
        refl_fl_ppm, _, _ = reflected_phase_curve(phases=phases_sorted, omega=single_scat_w[ins], g=g[ins], a_rp=ar/rprs)
        refl_pc_sorted_acc_time = refl_fl_ppm[idx_that_sort_arr_acc_times]/1e6
        sine_model = 1. + refl_pc_sorted_acc_time * ((flx_ecl - 1.) / 100e-6)

        # Total model
        total_model = flx_tra * sine_model
        total_models[instruments[ins]] = total_model


        # ------------------------------------------------------
        # Simulated data
        tim7 = times
        fl7 = total_model + np.random.normal(0., sigma, len(times))
        fle7 = np.ones(len(times))*sigma*0.1

        # ------------- Full dataset
        ## Saving the dataset
        tim[instruments[ins]], fl[instruments[ins]], fle[instruments[ins]] = tim7, fl7, fle7


    # ---------------------------------------------------------------------
    #
    #                    And now, the fitting!
    #
    # ---------------------------------------------------------------------

    # And priors
    ## Planetary priors
    par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + '_'.join(instruments), 'q2_' + '_'.join(instruments), 'ecc_p1', 'omega_p1', 'a_p1']
    dist_P = ['fixed', 'fixed', 'uniform', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
    hyper_P = [per, tc, [0., 1.], [0., 1.], [0., 1.], [0., 1.], 0., 90., [ar, ar_err]]

    if SamePC_2Inst:
        par_pc = ['singlescat_p1', 'g_p1']
        dist_pc = ['uniform', 'uniform']
        hyper_pc = [[0., 1.], [0., 1.]]
    else:
        par_pc, dist_pc, hyper_pc = [], [], []
        for ins in range(len(instruments)):
            par_pc = par_pc + ['singlescat_p1_' + instruments[ins], 'g_p1_' + instruments[ins]]
            dist_pc = dist_pc + ['uniform', 'uniform']
            hyper_pc = hyper_pc + [[0., 1.], [0., 1.]]


    ## Instrumental priors
    par_ins, dist_ins, hyper_ins = [], [], []
    for ins in range(len(instruments)):
        par_ins = par_ins + ['mdilution_' + instruments[ins], 'mflux_' + instruments[ins], 'sigma_w_' + instruments[ins]]
        dist_ins = dist_ins + ['fixed', 'normal', 'loguniform']
        hyper_ins = hyper_ins + [1., [0., 0.1], [0.1, 10000.]]

    ### Total priros
    par_tot = par_P + par_pc + par_ins
    dist_tot = dist_P + dist_pc + dist_ins
    hyper_tot = hyper_P + hyper_pc + hyper_ins

    priors_tot = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

    # And fitting
    dataset = juliet.load(priors=priors_tot, t_lc=tim, y_lc=fl, yerr_lc=fle, out_folder=pout, verbose=True)
    res = dataset.fit(sampler = 'dynesty', nthreads=8, light_travel_delay=True, stellar_radius=rst, kelp_refl_interpolation_knots=kelp_knots)

    # ---------------------------------------------------------------------
    #
    #                           Some plotting
    #
    # ---------------------------------------------------------------------

    for ins in range(len(instruments)):
        instrument = instruments[ins]

        # Let's plot some cool results!
        model = res.lc.evaluate(instrument)

        # Let's make sure that it works:
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
        ax1.plot(tim[instrument], model, c='k', zorder=100, label='Fitted model')
        ax1.plot(times, total_models[instrument], c='r', lw=2.5, alpha=0.5, zorder=50, label='Ingested model')
        ax1.set_ylabel('Relative Flux')
        ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
        ax1.set_ylim([1-300e-6, 1+300e-6])
        ax1.xaxis.set_major_formatter(plt.NullFormatter())
        ax1.legend()

        # Bottom panel
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
        ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD)')
        ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
        #plt.show()
        plt.savefig(pout + '/full_model_' + instrument + '.png')

    # ----------------------------------------------------
    #  Corner plot
    # ----------------------------------------------------
    post1 = res.posteriors['posterior_samples']

    if SamePC_2Inst:
        data = np.vstack([ post1['p_p1'], post1['b_p1'], post1['q1_' + '_'.join(instruments)], post1['q2_' + '_'.join(instruments)],\
                        post1['a_p1'], post1['singlescat_p1'], post1['g_p1'] ])
        data = np.transpose(data)
        lbls = np.array([ 'Rp/R*', 'b', 'q1', 'q2', 'a/R*', 'w', 'g'])
        truths = np.array([ rprs, bb, q1, q2, ar, single_scat_w[0], g[1] ])
    else:
        data = np.vstack([ post1['p_p1'], post1['b_p1'], post1['q1_' + '_'.join(instruments)], post1['q2_' + '_'.join(instruments)],\
                        post1['a_p1'], post1['singlescat_p1_' + instruments[0]], post1['g_p1_' + instruments[0]],\
                        post1['singlescat_p1_' + instruments[1]], post1['g_p1_' + instruments[1]] ])
        data = np.transpose(data)
        lbls = np.array([ 'Rp/R*', 'b', 'q1', 'q2', 'a/R*', 'w0', 'g0', 'w1', 'g1'])
        truths = np.array([ rprs, bb, q1, q2, ar, single_scat_w[0], g[0], single_scat_w[1], g[1]])

    fig = corner.corner(data, labels=lbls, show_titles=True, truths=truths)
    plt.savefig(pout + '/corner.png')