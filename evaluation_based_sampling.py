# Standard imports
import torch as tc
from copy import deepcopy

# Project imports
from primitives import primitives, distributions

class abstract_syntax_tree:
    def __init__(self, ast_json):
        self.functions = ast_json[:-1]
        self.program = ast_json[-1]


def eval(e, sig, l, rho={}, verbose=False):
    '''
    e: expression
    sig: side-effects
    l: local environment
    rho: global environment
    '''
    if verbose: print('Expression (before):', e)

    if type(e) in [float, int, bool]: # 'case c' with constants (float, int, bool)
        result = tc.tensor(e, dtype=tc.float)

    elif type(e) is str: # 'case v' look-up variable in local environment
        result = l[e]

    elif type(e) is list: # Usually e will be a list, which then needs to be evaluated

        if e[0] == 'defn': # NOTE: functions should all be stored in rho?
            raise ValueError('This defn case should never happen!')

        elif e[0] == 'let': # 'let' case needs to bind variable: (let [v1 e1] e0)
            expression = e[1][1]; name = e[1][0]
            c1 = eval(expression, sig, l, rho)
            l[name] = c1
            result = eval(e[2], sig, l, rho)

        elif e[0] == 'if': # 'if' case needs to do lazy evaluation (if e1 e2 e3)
            e1 = eval(e[1], sig, l, rho)
            result = eval(e[2], sig, l, rho) if e1 else eval(e[3], sig, l, rho)

        elif e[0] in ['sample', 'sample*']:
            d = eval(e[1], sig, l, rho)
            s = d.sample()
            log_prob = d.log_prob(s) # TODO: Wasteful if not doing MCMC
            sig['logP'] += log_prob
            result = s

        elif e[0] in ['observe', 'observe*']:
            d = eval(e[1], sig, l, rho)
            y = eval(e[2], sig, l, rho)
            log_prob = d.log_prob(y) # TODO: Wasteful if not doing MCMC or importance sampling?
            sig['logP'] += log_prob 
            sig['logW'] += log_prob
            result = y

        else: # case (e0 e1 . . . en) 
            
            # Loop over all elements and evaluate except the zeroth
            cs = []
            for element in e[1:]: # NOTE: Not the zeroth element
                c = eval(element, sig, l, rho)
                cs.append(c)

            if type(e[0]) is list: # NOTE: This should never happen
                print('List:', e[0])
                raise ValueError('This list case should never happen!')

            elif (type(e[0]) is str) and (e[0] in rho.keys()): # User-defined function
                variables, function_body = rho[e[0]] # Get the function from the global environment
                func_env = deepcopy(l) # NOTE: This is super important so as not to pollute environment
                for variable, exp in zip(variables, cs):
                    func_env[variable] = exp # Bind the function variables in the local environment
                func_env[e[0]] = function_body
                result = eval(function_body, sig, func_env, rho)

            elif (type(e[0]) is str) and (e[0] in distributions) and (e[0] in primitives.keys()):
               # Force distributions to not validate arguments (necessary for H4Q5)
               result = primitives[e[0]](*cs, validate_args=False)

            elif (type(e[0]) is str) and (e[0] in primitives.keys()): # Primitive function
                result = primitives[e[0]](*cs)

            else:
                print('List expression not recognised:', e)
                raise ValueError('List expression not recognised')
    else:
        print('Expression not recognised:', e)
        print('Expression not recognised')
    if verbose: 
        print('Expression (after):', e)
        print('Result:', result, type(result))
    return result


def bind_functions(ast):
    '''
    Put all function/procedure definitions into an environment
    Args:
        ast: abstract syntax tree; json FOPPL program
    Returns: A global function environment
    '''
    rho = {}
    for e in ast.functions:
        if e[0] == 'defn':
            rho[e[1]] = (e[2], e[3])
    return rho


def evaluate_program(ast, verbose=False):
    '''
    Evaluate a FOPPL program as desugared by daphne and generate a sample from the prior
    Args:
        ast: abstract syntax tree; json FOPPL program
    Returns: A single sample from the prior of the abstract syntax tree
    '''
    sig = {'logW': 0., 'logP': 0.}; l = {}; rho = bind_functions(ast)
    e = eval(ast.program, sig, l, rho, verbose)
    return e, sig, l