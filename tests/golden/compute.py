"""Run one configuration and collect every number the frozen-numerics layer compares."""
import numpy as np

from support.build import build, make_samples

EVALUATION_POINTS = 37


def flatten_model(model_dictionary, prefix, out):
    """Every numeric entry of model.model. Objects (batman parameters, radvel parameters) are skipped."""
    for key, value in model_dictionary.items():
        name = '{}/{}'.format(prefix, key)
        if isinstance(value, dict):
            flatten_model(value, name, out)
        elif isinstance(value, np.ndarray) and value.dtype.kind in 'fi':
            out[name] = np.copy(value)
        elif isinstance(value, (float, np.floating)):
            out[name] = np.array(value)


def instruments_of(model):
    return list(model.inames)


def evaluation_times(model, instrument):
    times = model.times[instrument]
    return np.linspace(times[0], times[-1], EVALUATION_POINTS)


def has_gp(model, instrument):
    key = 'global_model' if model.global_model else instrument
    return model.dictionary[key]['GPDetrend']


def evaluation_regressors(model, instrument, t):
    """The GP and linear regressors evaluate needs at new times, built like the data's regressors."""
    kwargs = {}
    if has_gp(model, instrument):
        key = 'global_model' if model.global_model else instrument
        nX = model.dictionary[key]['noise_model'].nX
        kwargs['GPregressors'] = t if nX == 1 else np.column_stack([t, np.sin(t)])
    if model.lm_boolean[instrument]:
        kwargs['LMregressors'] = np.column_stack([np.sin(t - model.times[instrument][0]),
                                                  np.cos(2. * (t - model.times[instrument][0]))])
    return kwargs


def run_op(op, build_fresh, instrument, out, prefix):
    model = build_fresh()
    name = '{}/evaluate/{}/{}'.format(prefix, op, instrument)
    if op == 'plain':
        samples, median = model.evaluate(instrument, all_samples=True, return_samples=True)
        out[name + '/samples'], out[name + '/median'] = samples, median
    elif op == 'components':
        median, components = model.evaluate(instrument, all_samples=True, return_components=True)
        out[name + '/median'] = median
        for key, value in components.items():
            out['{}/{}'.format(name, key)] = np.array(value)
    elif op == 'at_t':
        t = evaluation_times(model, instrument)
        kwargs = evaluation_regressors(model, instrument, t)
        out[name + '/median'] = model.evaluate(instrument, t=t, all_samples=True, **kwargs)
    elif op == 'transit':
        out[name + '/median'] = model.evaluate(instrument, all_samples=True, evaluate_transit=True)
    else:
        raise ValueError(op)


def compute(config, log_like_calc=False):
    """Returns {name: array}. Names are stable, so they can be compared between runs and with stored baselines."""
    out = {}
    model_kwargs = dict(config.model_kwargs, log_like_calc=log_like_calc)
    data = config.data()

    def fresh(modeltype, posterior):
        model = build(config.params, modeltype, fixed=config.fixed, posterior=posterior,
                      model_kwargs=model_kwargs, **data)[1]
        if posterior and config.extra_samples:
            model.set_posterior_samples(dict(model.posteriors, **make_samples(config.extra_samples)))
        return model

    total = 0.
    for modeltype in config.modeltypes:
        model = fresh(modeltype, False)
        model.generate(config.params)
        out[modeltype + '/modelOK'] = np.array(float(model.modelOK))
        flatten_model(model.model, modeltype + '/model', out)
        if model.modelOK:
            loglike = model.get_log_likelihood(config.params)
            out[modeltype + '/loglike'] = np.array(loglike)
            total += loglike
        for instrument in instruments_of(model):
            for op in config.ops:
                run_op(op, lambda: fresh(modeltype, True), instrument, out, modeltype)
    if len(config.modeltypes) > 1:
        out['joint/loglike'] = np.array(total)
    if config.loglike_only:
        out = {name: value for name, value in out.items() if name.endswith(('loglike', 'modelOK'))}
    return out
