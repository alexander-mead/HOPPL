# Standard imports
import torch as tc
from pyrsistent._pmap import PMap
from pyrsistent._plist import PList, _EmptyPList as EmptyPList
from pyrsistent import pmap, plist

# Options
use_pyrsistent = True

def fix_index(container, index):
    # Sort the index out to be appropriate for vectors, lists and dictionaries
    if type(container) in [tc.Tensor, list, PList]: index = int(index) # Indices for vectors/lists should be integers
    if type(index) is tc.Tensor: index = float(index) # Keys for dictionaries cannot be tensors
    return index


def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except: # NOTE: This except is horrible, but necessary for list/vector ambiguity
        result = plist(x) if use_pyrsistent else list(x)
    return result


def get(*x):
    # This needs to work for tensors, lists and dictionaries
    container, index = x[0], x[1]
    index = fix_index(container, index)
    return container[index]


def put(*x):
    # This needs to work for tensors, lists, and dictionaries
    container, index, new_value = x[0], x[1], x[2]
    index = fix_index(container, index)
    if type(container) in [PMap]:
        container = container.set(index, new_value)
    else:
        container[index] = new_value
    return container


def first(*x):
    # Return the first element in the container
    container = x[0]
    return container[0]


def second(*x):
    # Return the second element in the container
    container = x[0]
    return container[1]


def last(*x):
    # Return the final element in the container
    container = x[0]
    return container[-1]


def rest(*x):
    # Return all but the first element in a container of the same type
    container = x[0]
    return container[1:]


def append(*x):
    # Append value to end should work for tensors and lists
    container, value = x[0], x[1]
    if type(container) is tc.Tensor:
        value = tc.atleast_1d(value)
        result = tc.cat((container, value))
    elif type(container) in [list, PList, EmptyPList]:
        result = container.append(value)
    else:
        print('Container:', container)
        print('Type:', type(container))
        raise ValueError('Append not defined for this container')
    return result


def conj(*x):
    # Conjoin should prepend to a list and append to a vector
    container, value = x[0], x[1]
    if type(container) is tc.Tensor:
        result = tc.cat((tc.atleast_1d(value), tc.atleast_1d(container)))
    elif type(container) in [list, PList, EmptyPList]:
        if ((not container) or (container == plist([]))) and (type(value) is tc.Tensor):
            result = value # Returns a tensor if the container is empty but the value is tensorial
        else:
            result = [value]+container
    else:
        print('Value:', value)
        print('Container:', container)
        raise ValueError('Conj error')
    return result


def peek(*x):
    # For a list, same as first, for a vector, same as last.
    container = x[0]
    if type(container) in [list, PList]:
        result = container[0]
    elif type(container) is tc.Tensor:
        result = container[-1]
    else:
        print('Container:', container)
        raise ValueError('Container type not recognised')
    return result


def hashmap(*x):
    # This is a dictionary
    keys, values = x[0::2], x[1::2]
    checked_keys = []
    for key in keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        checked_keys.append(key)
    dictionary = dict(zip(checked_keys, values))
    hashmap = pmap(dictionary) if use_pyrsistent else dictionary
    return hashmap


def push_address(*x):
    # Concatenate two addresses to produce a new, unique address
    previous_address, current_addreess, continuation = x[0], x[1], x[2]
    return continuation(previous_address+'-'+current_addreess)


# Primative function dictionary
primitives = {

    # HOPPL
    'push-address': push_address,

    # Comparisons
    '<': lambda *x: x[-1](tc.lt(*x[1:-1])),
    '<=': lambda *x: x[-1](tc.le(*x[1:-1])),
    '>': lambda *x: x[-1](tc.gt(*x[1:-1])),
    '>=': lambda *x: x[-1](tc.ge(*x[1:-1])),
    '=': lambda *x: x[-1](tc.eq(*x[1:-1])),
    '!=': lambda *x: x[-1](tc.ne(*x[1:-1])),
    'and': lambda *x: x[-1](tc.logical_and(*x[1:-1])),
    'or': lambda *x: x[-1](tc.logical_or(*x[1:-1])),

    # Maths
    '+': lambda *x: x[-1](tc.add(*x[1:-1])),
    '-': lambda *x: x[-1](tc.sub(*x[1:-1])),
    '*': lambda *x: x[-1](tc.mul(*x[1:-1])),
    '/': lambda *x: x[-1](tc.div(*x[1:-1])),
    'exp': lambda *x: x[-1](tc.exp(*x[1:-1])),
    'log': lambda *x: x[-1](tc.log(*x[1:-1])),
    'sqrt': lambda *x: x[-1](tc.sqrt(*x[1:-1])),
    'abs': lambda *x: x[-1](tc.abs(*x[1:-1])),

    # Containers
    'vector': lambda *x: x[-1](vector(*x[1:-1])),
    'get': lambda *x: x[-1](get(*x[1:-1])),
    'put': lambda *x: x[-1](put(*x[1:-1])),
    'append': lambda *x: x[-1](append(*x[1:-1])),
    #'remove': None, # TODO: Add this
    #'cons': None, # TODO: Add this; should prepend to a list
    'conj': lambda *x: x[-1](conj(*x[1:-1])),
    'first': lambda *x: x[-1](first(*x[1:-1])),
    'second': lambda *x: x[-1](second(*x[1:-1])),
    #'nth': None, #lambda *x: x[0][x[1]], # TODO: Add this
    'last': lambda *x: x[-1](last(*x[1:-1])),
    'rest': lambda *x: x[-1](rest(*x[1:-1])),
    #'list': None, # TODO: Add this
    'empty?': lambda *x: x[-1](len(x[1]) == 0),
    'peek': lambda *x: x[-1](peek(*x[1:-1])),
    'hash-map': lambda *x: x[-1](hashmap(*x[1:-1])),

    # Matrices
    #'mat-transpose': lambda *x: x[1].T, # TODO: Looks dodgy...?
    'mat-transpose': lambda *x: x[-1](tc.transpose(*x[1:-1], 0, 1)),
    'mat-add': lambda *x: x[-1](tc.add(*x[1:-1])),
    'mat-mul': lambda *x: x[-1](tc.matmul(*x[1:-1])),
    'mat-tanh': lambda *x: x[-1](tc.tanh(*x[1:-1])),
    'mat-repmat': lambda *x: x[-1](x[1].repeat((int(x[2]), int(x[3])))), # TODO: Looks dodgy...? Use torch function

    # Distributions
    'normal': lambda *x: x[-1](tc.distributions.Normal(*x[1:-1])),
    'beta': lambda *x: x[-1](tc.distributions.Beta(*x[1:-1])),
    'exponential': lambda *x: x[-1](tc.distributions.Exponential(*x[1:-1])),
    'uniform-continuous': lambda *x: x[-1](tc.distributions.Uniform(*x[1:-1])),
    'discrete': lambda *x: x[-1](tc.distributions.Categorical(*x[1:-1])),
    'bernoulli': lambda *x: x[-1](tc.distributions.Bernoulli(*x[1:-1])),
    'gamma': lambda *x: x[-1](tc.distributions.Gamma(*x[1:-1])),
    'dirichlet': lambda *x: x[-1](tc.distributions.Dirichlet(*x[1:-1])),
    'flip': lambda *x: x[-1](tc.distributions.Bernoulli(*x[1:-1])), # NOTE: This is the same as Bernoulli

}