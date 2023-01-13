# Standard imports
import numpy as np
import torch as tc
from time import time

# Project imports
from evaluation_based_sampling import evaluate_program
from graph_based_sampling import evaluate_graph
from utils import burn_chain, log_sample, flatten_sample


def get_sample(ast_or_graph, mode, verbose=False):
    if mode == 'desugar':
        ret, sig, _ = evaluate_program(ast_or_graph, verbose=verbose)
    elif mode == 'graph':
        ret, sig, _ = evaluate_graph(ast_or_graph, verbose=verbose)
    else:
        raise ValueError('Mode not recognised')
    ret = flatten_sample(ret)
    return ret, sig


def prior_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a FOPPL program
    '''
    samples = []; weights = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, sig = get_sample(ast_or_graph, mode, verbose)
        weight = sig['logW'] # Importance weight
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)
        samples.append(sample); weights.append(weight)
        if (tmax is not None) and time() > max_time: break
    return samples, weights


def calculate_effective_sample_size(weights, verbose=False):
    '''
    Calculate the effective sample size via the importance weights
    '''
    weights /= weights.sum()
    ESS = 1./(weights**2).sum()
    ESS = ESS.type(tc.float)
    if verbose:
        print('Effective sample size:', ESS)
        print('Fractional sample size:', ESS/len(weights))
        print('Sum of weights:', weights.sum())
    return ESS


def resample_using_importance_weights(samples, log_weights, normalize=True, wandb_name=None):
    '''
    Use the (log) importance weights to resample so as to generate posterior samples 
    '''
    nsamples = samples.shape[0]
    if normalize: log_weights = log_weights-max(log_weights) # Makes the max log weight 0 for numerical stability
    weights = tc.exp(log_weights).type(tc.float64) # NOTE: float64 is necessary here or weights do not sum to unity
    _ = calculate_effective_sample_size(weights, verbose=True)
    indices = np.random.choice(nsamples, size=nsamples, replace=True, p=weights)
    if samples.dim() == 1:
        new_samples = samples[indices]
    else:
        new_samples = samples[indices, :]
    if wandb_name is not None:
        for i, sample in enumerate(new_samples):
            log_sample(sample, i, wandb_name, resample=True)
    return new_samples


def Metropolis_Hastings_samples(ast_or_graph, mode, num_samples, tmax=None, burn_frac=None, wandb_name=None, verbose=False):
    '''
    This does 'independent Metropolis Hastings' as per Section 4.2.1 of the book
    '''
    accepted_steps = 0; num_steps = 0
    samples = []; weights = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, sig = get_sample(ast_or_graph, mode, verbose)
        prob = tc.exp(sig['logW'])
        if i != 0:
            acceptance = min(1., prob/old_prob)
            accept = (tc.rand(size=(1,)) < acceptance)
        else:
            accept = True
        if accept:
            new_sample = sample; new_prob = prob
            accepted_steps += 1
        else:
            new_sample = old_sample; new_prob = old_prob
        num_steps += 1
        if wandb_name is not None: log_sample(sample, i, wandb_name)
        samples.append(new_sample); weights.append(tc.tensor(1.))
        old_sample = new_sample; old_prob = new_prob
        if (tmax is not None) and time() > max_time: break
    print('Acceptance fraction:', accepted_steps/num_steps)
    samples, weights = burn_chain(samples, weights, burn_frac=burn_frac)
    return samples, weights