# Standard imports
import numpy as np
import torch as tc
from time import time
import wandb
import hydra

# Project imports
from daphne import load_program
from tests import is_tol, run_probabilistic_test, load_truth
from general_sampling import get_sample, prior_samples, Metropolis_Hastings_samples, resample_using_importance_weights
from evaluation_based_sampling import abstract_syntax_tree
from graph_based_sampling import graph, Gibbs_samples, HMC_samples, VI_samples
from utils import wandb_plots_homework2, wandb_plots_homework3, wandb_plots_homework4

def create_class(ast_or_graph, mode):
    if mode == 'desugar':
        return abstract_syntax_tree(ast_or_graph)
    elif mode == 'graph':
        return graph(ast_or_graph)
    else:
        raise ValueError('Input type not recognised')


def run_tests(tests, mode, test_type, base_dir, daphne_dir, num_samples=int(1e4), max_p_value=1e-4, compile=True, verbose=False,):

    # File paths
    # NOTE: This path should be with respect to the daphne path
    test_dir = base_dir+'/programs/tests/'+test_type+'/'
    daphne_test = lambda i: test_dir+'test_%d.daphne'%(i)
    json_test = lambda i: test_dir+'test_%d_%s.json'%(i, mode)
    truth_file = lambda i: test_dir+'test_%d.truth'%(i)

    # Loop over tests
    print('Running '+test_type+' tests')
    for i in tests:
        print('Test %d starting'%i)
        print('Evaluation scheme:', mode)
        ast_or_graph = load_program(daphne_dir, daphne_test(i), json_test(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)
        truth = load_truth(truth_file(i))
        if verbose: print('Test truth:', truth)
        if test_type == 'deterministic':
            ret, _ = get_sample(ast_or_graph, mode, verbose=verbose)
            if verbose: print('Test result:', ret)
            try:
                assert(is_tol(ret, truth))
            except AssertionError:
                raise AssertionError('Return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast_or_graph))
        elif test_type == 'probabilistic':
            samples = []
            for _ in range(num_samples):
                sample, _ = get_sample(ast_or_graph, mode, verbose=verbose)
                samples.append(sample)
            p_val = run_probabilistic_test(samples, truth)
            print('p value:', p_val)
            assert(p_val > max_p_value)
        else:
            raise ValueError('Test type not recognised')
        print('Test %d passed'%i, '\n')
    print('All '+test_type+' tests passed\n')


def run_programs(programs, mode, prog_set, base_dir, daphne_dir, 
    num_samples=int(1e3), num_samples_per_step=int(1e2), num_steps=int(1e3), learning_rate=1e-1, zero_b=False,
    tmax=None, inference=None, compile=True, wandb_run=False, verbose=False,):

    # File paths
    prog_dir = base_dir+'/programs/'+prog_set+'/'
    daphne_prog = lambda i: prog_dir+'%d.daphne'%(i)
    json_prog = lambda i: prog_dir+'%d_%s.json'%(i, mode)
    if inference is not None:
        results_file = lambda i: 'data/%s/%d_%s_%s.dat'%(prog_set, i, mode, inference)
        sample_file = lambda i: 'data/%s/%d_%s_samples.dat'%(prog_set, i, mode)
        params_file = lambda i: 'data/%s/%d_%s_params.dat'%(prog_set, i, mode)
        loss_file = lambda i: 'data/%s/%d_%s_loss.dat'%(prog_set, i, mode)
    else:
        results_file = lambda i: 'data/%s/%d_%s.dat'%(prog_set, i, mode)

    for i in programs:
        if (inference == 'MHG') and (mode == 'desugar'): continue
        if (inference == 'HMC') and (mode == 'desugar'): continue
        if (inference == 'HMC') and (i in [3, 4]): continue
        if (inference == 'VI') and (mode == 'desugar'): continue
        t_start = time()
        wandb_name = 'Program %s'%i if wandb_run else None
        print('Running: '+prog_set+':' ,i)
        print('Maximum samples [log10]:', np.log10(num_samples))
        print('Maximum time [s]:', tmax)
        print('Inference method:', inference)
        print('Evaluation scheme:', mode)
        ast_or_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)
        if inference == 'MH':
            samples, weights = Metropolis_Hastings_samples(ast_or_graph, mode, num_samples, tmax=tmax, burn_frac=0.5, wandb_name=wandb_name, verbose=verbose)
        elif inference == 'MHG':
            if mode != 'graph': raise ValueError('Metropolis-Hastings within Gibbs only supported for graphs')
            samples, weights = Gibbs_samples(ast_or_graph, num_samples, tmax=tmax, burn_frac=0.5, wandb_name=wandb_name, verbose=verbose)
        elif inference == 'HMC':
            if mode != 'graph': raise ValueError('Hamiltonian Monte Carlo only supported for graphs')
            samples, weights = HMC_samples(ast_or_graph, num_samples, tmax=tmax, burn_frac=0.5, wandb_name=wandb_name, verbose=verbose)
        elif inference == 'VI':
            if mode != 'graph': raise ValueError('Variational Inference only supported for graphs')
            samples, weights, parameters, losses = VI_samples(
                ast_or_graph, num_samples, num_samples_per_step, num_steps, learning_rate, tmax=tmax, zero_b=zero_b,
                wandb_name=wandb_name, verbose=verbose)
            if i in [1, 2, 5]:
                np.savetxt(params_file(i), np.array(parameters))
            #print(losses)
            np.savetxt(loss_file(i), np.array(losses))
        elif inference in [None, 'IS']:
            samples, weights = prior_samples(ast_or_graph, mode, num_samples, tmax=tmax, verbose=verbose)
        else:
            raise ValueError('Inference method not recognised')
        samples = tc.stack(samples).type(tc.float); weights = tc.stack(weights).detach().type(tc.float)
        if inference in ['IS', 'VI']:
            np.savetxt(sample_file(i), samples)
            samples = resample_using_importance_weights(samples, weights, wandb_name=wandb_name)
        np.savetxt(results_file(i), samples)

        # Calculate some properties of the data
        print('Samples shape:', samples.shape)
        print('First sample:', samples[0])
        print('First weight [log]:', weights[0])
        print('Sample mean:', samples.mean(axis=0))
        print('Sample standard deviation:', samples.std(axis=0))

        if wandb_run and (prog_set == 'homework_2'): wandb_plots_homework2(samples, i)
        if wandb_run and (prog_set == 'homework_3'): wandb_plots_homework3(samples, i)
        if wandb_run and (prog_set == 'homework_4'): wandb_plots_homework4(samples, i)

        t_finish = time()
        print('Time taken [s]:', t_finish-t_start)
        print('Number of samples:', len(samples))
        print('Finished program {}\n'.format(i))


@hydra.main(version_base=None, config_path='', config_name='config')
def run_all(cfg):

    # Configuration
    wandb_run = cfg['wandb_run']
    mode = cfg['mode']
    inference = cfg['inference']
    num_samples = int(cfg['num_samples'])
    num_steps = int(cfg['num_steps'])
    num_samples_per_step = int(cfg['num_samples_per_step'])
    learning_rate = cfg['learning_rate']
    zero_b = cfg['zero_b']
    tmax = cfg['tmax']
    compile = cfg['compile']
    base_dir = cfg['base_dir']
    daphne_dir = cfg['daphne_dir']
    seed = cfg['seed']

    # Seed
    if (seed != 'None'):
        tc.manual_seed(seed)

    if inference == 'None': inference = None

    # Deterministic tests
    tests = cfg['deterministic_tests']
    run_tests(tests, mode=mode, test_type='deterministic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile, verbose=False)

    # Probabilistic tests
    tests = cfg['probabilistic_tests']
    run_tests(tests, mode=mode, test_type='probabilistic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile, verbose=False)

    # Homework 2
    programs = cfg['homework2_programs']
    if wandb_run and (len(programs) != 0): wandb.init(project='', entity='cs532-2022')
    run_programs(programs, mode=mode, prog_set='homework_2', base_dir=base_dir, daphne_dir=daphne_dir, 
        num_samples=num_samples, 
        compile=compile, wandb_run=wandb_run, verbose=False)
    if wandb_run and (len(programs) != 0): wandb.finish()

    # Homework 3
    programs = cfg['homework3_programs']
    if wandb_run and (len(programs) != 0): wandb.init(project='test_homework3', entity='cs532-2022')
    run_programs(programs, mode=mode, prog_set='homework_3', base_dir=base_dir, daphne_dir=daphne_dir, 
        num_samples=num_samples, tmax=tmax, inference=inference, 
        compile=compile, wandb_run=wandb_run, verbose=False)
    if wandb_run and (len(programs) != 0): wandb.finish()

    # Homework 4
    programs = cfg['homework4_programs']
    if wandb_run and (len(programs) != 0): wandb.init(project='HW4', entity='cs532-2022')
    run_programs(programs, mode=mode, prog_set='homework_4', base_dir=base_dir, daphne_dir=daphne_dir, 
        num_samples=num_samples, num_samples_per_step=num_samples_per_step, num_steps=num_steps, learning_rate=learning_rate, zero_b=zero_b,
        tmax=tmax, inference='VI', compile=compile, wandb_run=wandb_run, verbose=False)
    if wandb_run and (len(programs) != 0): wandb.finish()

    # TTV
    programs = cfg['TTV_programs']
    #if wandb_run and (len(programs) != 0): wandb.init(project='HW4', entity='cs532-2022')
    run_programs(programs, mode=mode, prog_set='TTV', base_dir=base_dir, daphne_dir=daphne_dir, 
        num_samples=num_samples, tmax=tmax, inference=inference, 
        compile=compile, wandb_run=wandb_run, verbose=False)
    #if wandb_run and (len(programs) != 0): wandb.finish()

if __name__ == '__main__':
    run_all()