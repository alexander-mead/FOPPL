# Standard imports
import torch as tc
from time import time
from copy import deepcopy
from graphlib import TopologicalSorter
from pprint import pprint

# Project imports
from primitives import primitives
from evaluation_based_sampling import eval
from utils import burn_chain, log_sample, log_params, log_loss, flatten_sample, covariance

class graph:

    def __init__(self, graph_json):

        # Definitions
        self.functions = graph_json[0] # NOTE: Not used
        self.nodes = graph_json[1]['V'] # (Will be) ordered set of nodes
        self.arrows = graph_json[1]['A'] # Dictionary with node keys
        self.expressions = graph_json[1]['P'] # Dictionary with node keys
        self.observe = graph_json[1]['Y'] # NOTE: Not used
        self.program = graph_json[2] # The actual program to evaluate
        
        # Calculations
        self.nodes = self.topological_sort()

    def __str__(self):
        things = {
            'Definitions': self.functions,
            'Nodes': self.nodes,
            'Arrows': self.arrows,
            'Expressions': self.expressions,
            'Observe': self.observe,
            'Deterministic': self.program,
        }
        for name, thing in things.items():
            print(name, thing, type(thing), len(thing))

    def topological_sort(self, verbose=False):
        '''
        Sort the nodes in topological order; order of dependancies
        Organises nodes from top to bottom of graph
        NOTE: Ordering for a specific graph is not necessarily unique
        '''

        # Add nodes with no arrows to graph
        for node in self.nodes:
            if node not in self.arrows.keys():
                self.arrows[node] = [] # Empty list for nodes with no outgoing arrows
        if verbose: print('arrows:', self.arrows)

        # Use topological sort to order the graph
        sorter = TopologicalSorter(self.arrows)
        sorted_list = list(sorter.static_order())
        return list(reversed(sorted_list))

    def split_nodes_into_sample_observe(self):
        '''
        Split the nodes by type into either sample (X) or observe (Y)
        '''
        sample_nodes = []; observe_nodes = []
        for node in self.nodes:
            if 'sample' in node:
                sample_nodes.append(node)
            elif 'observe' in node:
                observe_nodes.append(node)
            else:
                raise ValueError('Node present that is neither sample nor observe')
        return sample_nodes, observe_nodes

### Evaluation ###

def evaluate_node(node, exp, sig, l, fixed_dists={}, fixed_nodes={}, fixed_probs={}, verbose=False):
    '''
    Evaluate an individual graph node. Probabilities in sig will be updated
    '''
    if verbose: print('Node:', node)
    if node in fixed_dists.keys():
        result = fixed_dists[node].sample()
        p_log_prob = eval(exp[1], sig, l).log_prob(result) # Probability of sample under prior
        q_log_prob = fixed_dists[node].log_prob(result) # Probability of sample under Q
        sig['logP'] += q_log_prob # TODO: Probably unnecessary
        sig['logW'] += p_log_prob-q_log_prob # This is the general importance weight (when not sampling from prior)
    elif (node in fixed_nodes.keys()) and (node in fixed_probs.keys()):
        # Node probability and value are both fixed
        result = fixed_nodes[node]
        log_prob = fixed_probs[node]
        sig['logP'] += log_prob
        if 'observe' in node: sig['logW'] += log_prob
    elif node in fixed_nodes.keys():
        # Node value if fixed, but probability should be updated
        result = fixed_nodes[node]
        log_prob = eval(exp[1], sig, l).log_prob(result)
        sig['logP'] += log_prob
        if 'observe' in node: sig['logW'] += log_prob
    else:
        # Evaluate the node normally
        result = eval(exp, sig, l, verbose=verbose)
    if verbose: print('Value:', result)
    return result


def evaluate_graph(graph, fixed_dists={}, fixed_nodes={}, fixed_probs={}, verbose=False):
    '''
    This function does ancestral (parent -> children) sampling from the graph
    NOTE: The graph ought to have been topologically sorted prior to this
    NOTE: I do not use Y here at all, all the observe information is within P anyway...
    '''
    if verbose: print(graph)

    # Evaluate the nodes in order (results stored in enviornment: l)
    sig = {'logW': 0., 'logP': 0.}; l = {}
    for node in graph.nodes: # Loop over all nodes (which should have been ordered)
        exp = graph.expressions[node]
        original_logP = sig['logP']
        result = evaluate_node(node, exp, sig, l, fixed_dists=fixed_dists, fixed_nodes=fixed_nodes, fixed_probs=fixed_probs, verbose=verbose)
        l[node] = result # Update the local environment with the value
        l[node+'_logP'] = sig['logP']-original_logP # Update the local environment with the probability

    # Evaluate the final deterministic expression (just look-up things in environment)
    result = eval(graph.program, sig, l, verbose=verbose) 
    if verbose: print('Result:', result)
    return result, sig, l

### ###

### Gibbs sampling ###

def Gibbs_samples(graph, num_samples, tmax=None, burn_frac=None, wandb_name=None, debug=False, verbose=False):

    # Split the nodes by type into either sample (X) or observe (Y)
    sample_nodes, _ = graph.split_nodes_into_sample_observe()

    # Loop over samples
    samples = []; weights = []
    accepted_small_steps = 0; num_small_steps = 0; num_big_steps = 0
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        if i == 0:
            # Evaluate the graph in the standard way on the first iteration
            result, sig, l = evaluate_graph(graph, verbose=verbose)
        else:

            # Loop over the sample nodes
            for resample_node in sample_nodes: 

                # Re-sample the primary node
                resample_logP = l[resample_node+'_logP']
                sig_here = deepcopy(sig); l_here = deepcopy(l) # NOTE: Take care to copy to avoid making pointers
                d = eval(graph.expressions[resample_node], sig_here, l_here) # Generate a new sample for the node NOTE: Careful! This will update sig, l
                resample_logP_new = sig_here['logP']-sig['logP']
                fixed_nodes = {resample_node: d}; fixed_probs = {resample_node: resample_logP_new}
                if debug:
                    print('Original node value:', l[resample_node])
                    print('Original node logP:', resample_logP)
                    print('Resampled node value:', d)
                    print('Resampled node logP:', resample_logP_new)

                # Set fixed values for some nodes and evaluate the graph again
                # TODO: This is not as efficient as it could be
                for node in graph.nodes:
                    if node != resample_node:
                        fixed_nodes[node] = l[node] # Fix values for all the nodes
                        if node not in graph.arrows[resample_node]:
                            fixed_probs[node] = l[node+'_logP'] # Fix probabilities for all but child nodes
                if debug:
                    print('Fixed nodes:', fixed_nodes)
                    print('Fixed probabilities:', fixed_probs)
                result_new, sig_new, l_new = evaluate_graph(graph, fixed_nodes=fixed_nodes, fixed_probs=fixed_probs, verbose=verbose)
                if debug:
                    print('Old sig:', sig)
                    print('New sig:', sig_new)
                    print('Old environment:', l)
                    print('New environment:', l_new)

                # Calculate the acceptance probabilities
                acceptance = tc.exp(sig_new['logP']-sig['logP']-resample_logP_new+resample_logP)
                alpha = min(1., acceptance)
                accept = (tc.rand(size=(1,)) < alpha)
                if accept: result, sig, l = result_new, sig_new, l_new; accepted_small_steps += 1
                if wandb_name is not None: log_sample(result, i, wandb_name)
                num_small_steps += 1
                if debug:
                    print('Acceptance probability:', acceptance)
                    print('Step accepted:', accept)
            if debug: exit()

        # Update for this step of Gibbs
        num_big_steps += 1
        samples.append(result); weights.append(tc.tensor(1.))
        if (tmax is not None) and time() > max_time: break

    # Finalize
    print('Acceptance fraction:', accepted_small_steps/num_small_steps)
    print('Number of samples:', num_big_steps)
    if burn_frac is not None:
        print('Burn fraction:', burn_frac)
        nburn = int(burn_frac*num_big_steps)
        print('Burning up to:', nburn)
        samples = samples[nburn:]; weights = weights[nburn:]
    return samples, weights

### ###

### Hamiltonian Monte Carlo ###

def generate_IC(graph, verbose=False):
    '''
    Generates a set of initial conditions for Hamiltonian Monte Carlo
    '''
    _, _, l = evaluate_graph(graph, verbose=verbose)
    start = tc.tensor([l[node] for node in graph.nodes if 'sample' in node])
    if verbose: print('Initial conditions:', start)
    return start


def log_joint(graph, x, verbose=False):
    '''
    Evaluates a graph with fixed (sample) node values and returns ln(P(x,y))
    This is the target function for Hamiltonian Monte Carlo
    '''
    # Fill the fixed nodes dictionary
    fixed_nodes = {}; i = 0
    for node in graph.nodes:
        if 'sample' in node:
            fixed_nodes[node] = x[i]; i += 1
    # Evaluate the graph and recover the log joint probability
    _, sig, _ = evaluate_graph(graph, fixed_nodes=fixed_nodes, verbose=verbose)
    log_joint = sig['logP']
    return log_joint


def HMC_samples(graph, num_samples, tmax=None, burn_frac=None, M=1., dt=0.1, T=1., wandb_name=None, verbose=False):
    '''
    Hamiltonian Monte Carlo with m chains of length n
    @params:
        lnf: ln(f(x)) natural logarithm of the target function
        start: starting location in parameter space
        n_chains: Number of independent chains
        n_points: Number of points per chain
        burn_frac: Fraction of the beginning of the chain to remove
        M: Mass for the 'particles' TODO: Make matrix
        dt: Time-step for the particles
        T: Integration time per step for the particles
    '''
    # Functions for leap-frog integration
    def get_gradient(x, lnf):
        # NOTE: Be very careful to stop computation graph exploding
        x = x.detach()
        x.requires_grad_()
        lnf(x).backward()
        dlnfx = x.grad
        x = x.detach() # NOTE: Maybe unnecessary
        return dlnfx
    def forward_Euler_step(x, p, lnf, M, dt):
        dlnfx = get_gradient(x, lnf)
        x_full = x+p*dt/M
        p_full = p+dlnfx*dt
        return x_full, p_full
    def leap_frog_step(x, p, lnf, M, dt):
        dlnfx = get_gradient(x, lnf)
        p_half = p+0.5*dlnfx*dt
        x_full = x+p_half*dt/M
        dlnfx = get_gradient(x_full, lnf)
        p_full = p_half+0.5*dlnfx*dt
        return x_full, p_full
    def leap_frog_integration(x_init, p_init, lnf, M, dt, T, method='leapfrog'):
        N_steps = int(T/dt)
        x, p = tc.clone(x_init), tc.clone(p_init)
        step = leap_frog_step if method=='leapfrog' else forward_Euler_step
        for _ in range(N_steps):
            x, p = step(x, p, lnf, M, dt)
        return x, p
    def Hamiltonian(x, p, lnf, M):
        T = 0.5*tc.dot(p, p)/M
        V = -lnf(x)
        return T+V

    # Wire generic routine into graph
    start = generate_IC(graph, verbose=verbose)
    lnf = lambda x: log_joint(graph, x)

    # MCMC step
    samples = []; weights = []
    x_old = tc.clone(start); n = len(start); accepted_steps = 0; num_steps = 0
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        p_old = tc.normal(0., 1., size=(n,))
        if i == 0: H_old = 0.
        x_new, p_new = leap_frog_integration(x_old, p_old, lnf, M, dt, T)
        H_new = Hamiltonian(x_new, p_new, lnf, M)
        acceptance = 1. if (i == 0) else min(tc.exp(H_old-H_new), 1.) # Acceptance probability
        accept = tc.rand((1,)) < acceptance
        if accept: x_old, H_old = x_new, H_new; accepted_steps += 1
        num_steps += 1
        if wandb_name is not None: log_sample(x_old, i, wandb_name)
        samples.append(x_old); weights.append(tc.tensor([1.]))
        if (tmax is not None) and time() > max_time: break
    print('Number of steps:', num_steps)
    print('Acceptance fraction:', accepted_steps/num_steps)
    samples, weights = burn_chain(samples, weights, burn_frac=burn_frac)
    return samples, weights

### ###

### Variational Inference ###

def get_variational_distributions(graph: graph, use_prior=False, verbose=False) -> dict:
    '''
    Get the variational distributions corresponding to each sample node
    @params:
        graph: Graph class
        use_prior: Boolean; use prior to set the initial parameters for the variational distribution?
        verbose: Boolean verbosity
    '''
    variational_distributions = {}; sig = {}; l = {}
    for node in graph.nodes:
        if 'sample' in node:
            exp = graph.expressions[node][1] # The first element in this list is 'sample', the second is distribution
            if verbose: print('exp:', exp)
            if use_prior:
                exp[0] = exp[0]+'-guide' # Append _grad to get the appropriate variational distribution
                variational_distributions[node] = eval(exp, sig, l) # Use prior for starting parameters
            else:
                dist = exp[0]+'-guide'
                parameters = primitives[exp[0]+'-params']
                parameters  = tuple([param.clone() for param in parameters]) # NOTE: Must clone parameters here!
                variational_distributions[node] = primitives[dist](*parameters)
    if verbose: print('Variational distributions:', variational_distributions)
    return variational_distributions


def VI_evaluate_graph(graph: graph, variationals: dict, verbose=False):
    '''
    Evaluate a graph for BBVI
    @params:
        graph: Graph class
        variationals: Dictionary of variational distributions corresponding to each node
        verbose: Boolean verbosity
    '''
    sig = {'logW': 0., 'logP': 0., 'logQ': {}}; l = {}
    for node in graph.nodes:
        exp = graph.expressions[node]
        if 'sample' in node:
            variational = variationals[node]
            result, sig = VI_evaluate_sample_node(node, exp, sig, l, variational, verbose=verbose)
        elif 'observe' in node:
            result, sig = VI_evaluate_observe_node(exp, sig, l, verbose=verbose)
        else:
            raise ValueError('Node present that is neither sample nor observe')
        l[node] = result

    result = eval(graph.program, sig, l, verbose=verbose) 
    if verbose: print('Result:', result)
    return result, sig, l


def VI_evaluate_sample_node(node: str, exp: list, sig: dict, l: dict, variational: tc.distributions.Distribution, verbose=False):
    '''
    Evaluate a sample node for BBVI
    Algorithm 11 case (sample v e) from the book
    @params:
        node: String name of node
        exp: List expression corresponding to node
        sig: Dictionary of program side effects (contains grad_log_prob and importance weights)
        l: Dictionary of local enviornment
        variational: Variational distribution corresponding to node
        verbose: Boolean verbosity
    '''
    d = eval(exp[1], sig, l, verbose=verbose)
    c = variational.sample()
    logP = d.log_prob(c) # NOTE: Detach not necessary here
    logQ = variational.log_prob(c) # NOTE: Do not detach here
    sig['logQ'][node] = logQ
    sig['logW'] += logP-logQ.detach()
    return c, sig


def VI_evaluate_observe_node(exp: list, sig: dict, l: dict, verbose=False):
    '''
    Evaluate an observe node for BBVI
    Algorithm 11 case (observe v e1 e2) from the book
    @params:
        exp: List expression corresponding to node
        sig: Dictionary of program side effect (contains grad_log_prob and importance weights)
        l: Dictionary of local enviornment
        verbose: Boolean verbosity
    '''
    d = eval(exp[1], sig, l, verbose=verbose)
    c = eval(exp[2], sig, l, verbose=verbose)
    logP = d.log_prob(c) # NOTE: Detach not necessary here
    sig['logW'] += logP
    return c, sig


def VI_samples(graph: graph, num_samples=int(1e3), num_samples_per_step=int(1e2), num_steps=int(1e3), learning_rate=1e-1, 
    tmax=None, zero_b=False, wandb_name=None, verbose=False):
    '''
    Draws samples from a general proposal distribution and returns the samples and importance weights
    'BBVI' function from Algorithm 12 in the book
    @params:
        graph: Graph class
        num_samples: Integer total number of samples to draw for variational inference
        num_samples_per_step: Integer number of samples per each training step
        learning_rate: Learning rate
        tmax: Maximum time to run algorithm for
        wandb_name: Weights and biases name
        verbose: Boolean verbosity
    '''
    # TODO: Use zero_b

    # Parameters
    rejection_sampling = True # Might be necessary for Program 5
    #bad_weight = None
    bad_weight = tc.tensor(-1000.) # A large negative weight
    #bad_weight = tc.tensor(0.) # Zero weight
    print_diagnostics = True

    # Calculations
    variationals = get_variational_distributions(graph, use_prior=False, verbose=False)
    sample_nodes, _ = graph.split_nodes_into_sample_observe()

    # Get VI samples
    parameters = []; losses = []; i = 0
    optimizer = intialize_optimizer(variationals, learning_rate)
    if (tmax is not None): max_time = time()+tmax
    for step in range(num_steps):
        logWs = []; logQs = []
        for _ in range(num_samples_per_step):
            sample, sig, _ = VI_evaluate_graph(graph, variationals, verbose=verbose)
            logW, logQ = sig['logW'], sig['logQ']
            sample = flatten_sample(sample); logW = flatten_sample(logW)
            if (bad_weight is not None) and (tc.isinf(logW) or tc.isnan(logW)): logW = bad_weight
            if rejection_sampling and (tc.isinf(logW) or tc.isnan(logW)): continue
            i += 1
            logWs.append(logW); logQs.append(logQ) # logW and logQ pertaining to the current update
            if wandb_name is not None: log_sample(sample, i, wandb_name)
            if (tmax is not None) and time() > max_time: break
        ELBO = update_parameters(sample_nodes, variationals, logQs, logWs, optimizer, zero_b=zero_b)
        if wandb_name is not None: 
            log_params(variationals, step, wandb_name)
            log_loss(ELBO, step, wandb_name)
        parameters = save_parameters(parameters, variationals)
        losses.append(ELBO)
        if print_diagnostics: # Diagnostic information
            print('Distribution:')
            pprint(variationals)
            print('ELBO:', float(ELBO))
            print('Step:', step)

    # Generate samples and get weights from final Q distributions
    samples = []; logWs = []
    for _ in range(num_samples):
        sample, sig, _ = VI_evaluate_graph(graph, variationals, verbose=verbose)
        logW = sig['logW']
        sample = flatten_sample(sample); logW = flatten_sample(logW)
        if rejection_sampling and (tc.isinf(logW) or tc.isnan(logW)): continue
        samples.append(sample); logWs.append(logW)

    return samples, logWs, parameters, losses


def save_parameters(parameters, variationals):
    '''
    Add a line to a long list of distribution parameters corresponding to each update step
    @params:
        parameters: list of distribution parameters to be appended to
        variational: dict of variational distributions corresponding to each node
    '''
    params_here = []
    for dist in variationals.values():
        params = [p.clone().detach().numpy() for p in dist.params()]
        params_here.extend(params)
    parameters.append(params_here)
    return parameters


def calculate_b(node: str, variational, logQs: tc.Tensor, logWs: tc.Tensor, zero=False) -> tc.Tensor:
    '''
    Calculate the factor that minimizes the variance of the ELBO gradient estimator
    TODO: Seems not to work with a batch size of 1
    @params:
        node: The node for which we are calculating b
        variational: variational distribution corresponding to node
        logQs: List of dictionaries of probabilities of sample from each Q
        logWs: List of importance weights corresponding to each run through the graph
        zero: Boolean; should b be fixed to zero?
    '''
    if zero:
        b = 0.
    else:
        Fs = []; Gs = []
        for logQ, logW in zip(logQs, logWs):
            Q = logQ[node]
            Q.backward(retain_graph=True) # Necessary to allow further backward passes later
            grads = [param.grad for param in variational.optim_params()]
            if len(grads) == 1: # TODO: This is lazy
                G = grads[0]
            else:
                G = tc.tensor(grads)
            for param in variational.optim_params():
                param.grad.zero_() # Necessary to stop gradients being accumulated downstream
            F = G*logW
            Fs.append(F); Gs.append(G)
        Fs = tc.stack(Fs); Gs = tc.stack(Gs)
        cov_FG = tc.sum(covariance(Fs, Gs)) # TODO: Sum? That's what it says in the book
        var_GG = tc.sum(covariance(Gs, Gs))
        b = cov_FG/var_GG
    return b


def update_parameters(nodes: list, variationals, logQs: list, logWs: list, optimizer, zero_b=False):
    '''
    @params:
        nodes: List of sample node names
        variationals: Dictionary of variational distributions corresponding to each node
        logQs: List of dictionaries of probabilities of sample from each Q
        logWs: List of importance weights corresponding to each run through the graph
        optimizer: Instance of torch optmizer class
    '''
    total_ELBO = 0.; total_loss = 0.; batch_size = len(logQs)
    for node in nodes:
        b = calculate_b(node, variationals[node], logQs, logWs, zero=zero_b)
        ELBO = 0.; loss = 0.
        for logQ, logW in zip(logQs, logWs): # Loop over the batch
            ELBO -= logQ[node]*logW
            loss -= logQ[node]*((logW-b).detach())
        ELBO /= batch_size; loss /= batch_size # Average over batch
        total_ELBO += ELBO; total_loss += loss
    total_loss.backward()
    optimizer.step(); optimizer.zero_grad()
    return total_ELBO.clone().detach()


def intialize_optimizer(variationals: dict, learning_rate: float) -> dict:
    '''
    Returns a dictionary of initialised optimizers corresponding to each sample node
    @params:
       variationals: dictionary of node names (keys) and variational distributions (values)
       learning_rate: learning rate
    '''
    all_parameters = []
    for dist in variationals.values():
        parameters = dist.optim_params()
        all_parameters.extend(parameters)
    optimizer = tc.optim.Adam(all_parameters, lr=learning_rate)
    return optimizer

### ###