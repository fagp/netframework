import torch
import copy
from os.path import exists, commonprefix

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]

def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]

def create_target_direction(net, net2, dir_type='states'):
    assert (net2 is not None)
    # direction between net2 and net
    if dir_type == 'weights':
        w = get_weights(net)
        w2 = get_weights(net2)
        direction = get_diff_weights(w, w2)
    elif dir_type == 'states':
        s = net.state_dict()
        s2 = net2.state_dict()
        direction = get_diff_states(s, s2)

    return direction

def set_states(net, states, dx=None, dy=None, step=None):
    assert step is not None, 'If direction is provided then the step must be specified as well'
    changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
    
    new_states = copy.deepcopy(states)
    assert (len(new_states) == len(changes))
    for (k, v), d in zip(new_states.items(), changes):
        d = torch.tensor(d)
        v.add_(d.type(v.type()))
    net.load_state_dict(new_states)

################################################################################
#                        Normalization Functions
################################################################################
def normalize_direction(direction, weights, norm='filter'):
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_states(direction, states, norm='filter', ignore='biasbn'):
    assert(len(direction) == len(states))
    for (_,d), (k, w) in zip(direction.items(), states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)