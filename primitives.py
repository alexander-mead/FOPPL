# Standard imports
import torch as tc
import distributions as dists

# Project imports
from TTV import oneplanet

def vector(*x):
    # NOTE: This must support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except: # TODO: This except is horrible, but necessary for list/vector ambiguity
        # TODO: Be very careful that e.g., rest([2 3]) = [3] rather than 3
        result = list(x)
    return result


def fix_index(container, index):
    # Sort the index out to be appropriate for vectors, lists and dictionaries
    if type(container) in [tc.Tensor, list]: index = int(index) # Indices for vectors/lists should be integers
    if type(index) is tc.Tensor: index = float(index) # Keys for dictionaries cannot be tensors
    return index


def get(*x):
    # NOTE: This must work for tensors, lists and dictionaries
    container = x[0]; index = x[1]
    index = fix_index(container, index)
    return container[index]


def put(*x):
    # NOTE: This must work for tensors, lists and dictionaries
    container = x[0]; index = x[1]; new_value = x[2]
    index = fix_index(container, index)
    container[index] = new_value
    return container


def append(*x):
    # NOTE: This should work for tensors and lists
    container = x[0]; value = x[1]
    if type(container) is tc.Tensor:
        value = tc.atleast_1d(value)
        result = tc.cat((container, value))
    elif type(container) is list:
        result = container.append(value)
    else:
        print('Container:', container)
        print('Type:', type(container))
        raise ValueError('Append not defined for this container')
    return result


def hashmap(*x):
    # NOTE: This is a dictionary
    _keys = [key for key in x[0::2]]
    keys = []
    for key in _keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        keys.append(key)
    values = [value for value in x[1::2]]
    return dict(zip(keys, values))


class delta_distribution:
    def __init__(self, x):
        self.x = x
    def sample(self):
        return self.x
    def log_prob(self, x0):
        return tc.tensor(tc.inf) if (self.x == x0) else tc.tensor(-tc.inf)


def dirac_delta_distribution(*x, scheme='normal'):
    if scheme == 'normal':
        return tc.distributions.normal.Normal(loc=x[0], scale=0.1)
    elif scheme == 'uniform':
        return tc.distributions.uniform.Uniform(low=x[0]-0.05, high=x[0]+0.05)
    elif scheme == 'delta':
        return delta_distribution(*x)
    else:
        raise ValueError('Dirac delta scheme not recognised')


# Primative function dictionary
primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    '>': tc.gt,
    '>=': tc.ge,
    '=': tc.eq,
    #'!=': None, # TODO: Should be tc.ne I guess...
    'and': tc.logical_and,
    'or': tc.logical_or,

    # Maths
    '+': tc.add,
    '-': tc.sub,
    '*': tc.mul,
    '/': tc.div,
    'exp': tc.exp,
    'sqrt': tc.sqrt,
    'abs': tc.abs,

    # Containers
    'vector': vector,
    'get': get,
    'put': put,
    'append': append,
    'remove': None,
    #'cons': None, # TODO: Should prepend to a list
    #'conj': None, # TODO: Should prepend to a list and append to a vector
    'first': lambda *x: x[0][0],
    'second': lambda *x: x[0][1],
    #'nth': None, #lambda *x: x[0][x[1]], # TODO: Should it be lambda x* here?
    'last': lambda *x: x[0][-1],
    'rest': lambda x: x[1:], # TODO: Should it be lambda x* here? Test 21 fails if so... not sure!
    #'list': None, # TODO: Note that this does not appear in any daphne files
    'hash-map': hashmap,

    # Matrices
    'mat-transpose': lambda *x: x[0].T,
    'mat-add': tc.add,
    'mat-mul': tc.matmul,
    'mat-tanh': tc.tanh,
    'mat-repmat': lambda *x: x[0].repeat((int(x[1]), int(x[2]))),

    # Distributions
    'normal': tc.distributions.Normal,
    'beta': tc.distributions.Beta,
    'exponential': tc.distributions.Exponential,
    'uniform-continuous': tc.distributions.Uniform,
    'discrete': tc.distributions.Categorical,
    'bernoulli': tc.distributions.Bernoulli,
    'gamma': tc.distributions.Gamma,
    'dirichlet': tc.distributions.Dirichlet,
    'flip': tc.distributions.Bernoulli, # NOTE: This is the same as Bernoulli
    'dirac': dirac_delta_distribution,

    # Distributions with gradients (called guide functions in Pyro)
    # NOTE: These are much slower than the native pytorch distributions
    # NOTE: Transforming parameters to optim parameters is what takes time
    'normal-guide': dists.Normal,
    'beta-guide': dists.Beta,
    'exponential-guide': dists.Exponential,
    'uniform-continuous-guide': dists.Gamma, # TODO: This is not a general map for uniform, but works for H4Q5!
    #'uniform-continuous-guide': dists.Normal, # TODO: This is not a general map for uniform!
    #'uniform-continuous-guide': dists.Beta, # TODO: This is not a general map for uniform!
    'discrete-guide': dists.Categorical,
    'bernoulli-guide': dists.Bernoulli,
    'gamma-guide': dists.Gamma,
    'dirichlet-guide': dists.Dirichlet,
    'flip-guide': dists.Bernoulli,

    # Starting parameters for distributions (for variational inference)
    'normal-params': (tc.tensor(0.), tc.tensor(1.)),
    'beta-params': (tc.tensor(1.), tc.tensor(1.)),
    'exponential-params': (tc.tensor(1.),),
    'uniform-continuous-params': (tc.tensor(1.), tc.tensor(1.)), # NOTE: This maps to gamma and is not general
    #'uniform-continuous-params': (tc.tensor(0.), tc.tensor(1.)), # NOTE: This maps to normal and is not general
    #'uniform-continuous-params': (tc.tensor(1.), tc.tensor(1.)), # NOTE: This maps to beta and is not general
    'discrete-params': (tc.tensor([1./3., 1./3., 1./3.]),), # TODO: 3 logits is not general (H4Q3 specific)
    'bernoulli-params': (tc.tensor(0.5),),
    'gamma-params': (tc.tensor(1.), tc.tensor(1.)),
    'dirichlet-params': (tc.tensor([1., 1., 1.]),), # TODO: 3 concentration parameters is not general (H4Q3 specific)
    'flip-params': (tc.tensor(0.5),),

    # TTV
    'oneplanet': oneplanet,

}

# A list of all the supported distributions
distributions = [
    'normal',
    'beta',
    'exponential',
    'uniform-continuous',
    'discrete',
    'bernoulli',
    'gamma',
    'dirichlet',
    'flip',
    'dirac',
]