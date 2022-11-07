# Standard imports
import torch as tc
from time import time

# Project imports
from evaluator import eval, evaluate, standard_env
from utils import log_sample_to_wandb, log_samples_to_wandb
from utils import check_addresses, calculate_log_evidence, resample_using_importance_weights

def get_samples(ast:dict, num_samples:int, tmax=None, inference=None, wandb_name=None, verbose=False):
    '''
    Get some samples from a HOPPL program
    '''
    if inference is None:
        samples = get_prior_samples(ast, num_samples, tmax=tmax, wandb_name=wandb_name, verbose=verbose)
    elif inference == 'IS':
        samples = get_importance_samples(ast, num_samples, tmax=tmax, wandb_name=wandb_name, verbose=verbose)
    elif inference == 'SMC':
        samples = get_SMC_samples(ast, num_samples, wandb_name=wandb_name, verbose=verbose)
    else:
        print('Inference scheme:', inference, type(inference))
        raise ValueError('Inference scheme not recognised')
    return samples


def get_prior_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a HOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, _ = evaluate(ast, verbose=verbose)
        if wandb_name is not None: log_sample_to_wandb(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and (time() > max_time): break
    return samples


def get_importance_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of importamnce samples from a HOPPL program
    '''
    samples = []; log_weights = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sig = {'logW': tc.tensor(0.)}
        sample, sig = evaluate(ast, sig, verbose=verbose)
        log_weight = sig['logW']
        if wandb_name is not None: log_sample_to_wandb(sample, i, wandb_name=wandb_name)
        samples.append(sample), log_weights.append(log_weight)
        if (tmax is not None) and (time() > max_time): break
    samples = resample_using_importance_weights(samples, log_weights, wandb_name=wandb_name)
    return samples


def get_SMC_samples(ast:dict, num_samples:int, run_name='start', wandb_name=None, verbose=False):
    '''
    Generate a set of samples via Sequential Monte Carlo from a HOPPL program
    '''
    # Initialisation for all calculations
    identity = lambda x: x # Identity function to evaluate program
    logZs = [] # To accumulate evidence

    # Loop over the observes
    run = True; iobserve = 0
    while run:

        # Loop over 'particles'
        samples = []; log_weights = []
        for isample in range(num_samples):

            # Initialisation
            if iobserve == 0: # Starting initialisation
                sig = {'location': run_name, 'logW': tc.tensor(0.), 'address': ''}
                env = standard_env()
                exp = eval(ast, sig, env, verbose)(run_name, identity)
            else: # Further re-initialisation (after observes)
                exp = resamples[isample]
                sig = exp[2]
                sig['location'] = 'continue' # Reset location
                sig['logW'] = tc.tensor(0.)  # Reset weight

            # If there are continuations then exp will be a tuple and a re-evaluation needs to occur
            while (sig['location'] != 'observe') and (type(exp) is tuple):
                func, args, sig = exp
                exp = func(*args)

            # Collect samples and weights (observe or end of program)
            sample, log_weight = exp, sig['logW']
            samples.append(sample); log_weights.append(log_weight)

        # Decide how to proceed
        if sig['location'] == 'observe': # Resample using importance weights
            iobserve += 1 # Add to observe counter
            print('Observe:', iobserve)
            check_addresses(samples)
            logZ = calculate_log_evidence(log_weights)
            print('Contribution to log evidence:', logZ)
            logZs.append(logZ)
            resamples = resample_using_importance_weights(samples, log_weights, verbose=True)
            if verbose:
                for isample, sample in enumerate(samples):
                    print('observe', iobserve, 'sample:', isample, sample[2]['logW'])
                for iresample, resample in enumerate(resamples):
                    print('observe', iobserve, 'resample:', iresample, resample[2]['logW'])
        else: # Program has finished, so return samples
            log_samples_to_wandb(samples, wandb_name=wandb_name)
            run = False

    # Calculate the final evidence and return
    logZ = tc.sum(tc.tensor(logZs))
    print('Final log evidence:', logZ)
    return samples