import torch

from abc import ABC, abstractmethod
from functools import singledispatch


class BaseScalarization(ABC):
    def __init__(self, **kwargs):
            pass
    @abstractmethod
    def scalarize(self):
        pass

class SumScalarization(BaseScalarization):

    def scalarize(self, scores, weights):
        weighted_scores = scores * weights.unsqueeze(0)
        #sum_weights = torch.sum(weights) 
        return weighted_scores.sum(dim=1)  #/ sum_weights # is needed for static mpo when static sum weights > 1


class ProductScalarization(BaseScalarization):

    def __init__(self, epsilon=0.001):
        self.epsilon = epsilon

    def scalarize(self, scores, weights):
        scores = torch.clamp_min(scores, self.epsilon)
        weighted_log_scores = torch.log(scores) * weights.unsqueeze(0)
        sum_weighted_log_scores = weighted_log_scores.sum(dim=1)
        return torch.exp(sum_weighted_log_scores)
    

class ChebyshevScalarization(BaseScalarization):

    def scalarize(self, scores, weights, targets):
        weighted_abs = abs(scores - targets) * weights.unsqueeze(0)
        return (1 - weighted_abs.max(dim=1).values)
    

class MinkowskiDistance(BaseScalarization):

    def scalarize(self, scores, weights, targets, p):
        weighted_abs = abs(scores - targets) * weights.unsqueeze(0)
        minkowski_distance = weighted_abs.pow(p).sum(dim=1).pow(1/p)
        return 1 - (minkowski_distance / targets.shape[-1]**(1/p))
    

class ScalarizationFactory:
    _methods = {
        'sum' : SumScalarization,
        'product' : ProductScalarization,
        'chebyshev' : ChebyshevScalarization,
        'minkowski' : MinkowskiDistance
    }

    @staticmethod
    def get_scalarization_method(method, **kwargs):
        scalarization_method = ScalarizationFactory._methods.get(method)
        if not scalarization_method:
            raise ValueError(f'Unknown scalarization method: {method}')
        return scalarization_method(**kwargs)

@singledispatch
def _dispatch_scalarize(scalarizer, scores, weights, **kwargs):
    return scalarizer.scalarize(scores, weights)

@_dispatch_scalarize.register(ChebyshevScalarization)
def _(scalarizer, scores, weights, targets, **kwargs):
    return scalarizer.scalarize(scores, weights, targets)

@_dispatch_scalarize.register(MinkowskiDistance)
def _(scalarizer, scores, weights, targets, p, **kwargs):
    return scalarizer.scalarize(scores, weights, targets, p)

def scalarize_reward(scalarizer, scores, weights, **kwargs):
    normalized_weights = weights / torch.sum(weights)
    return _dispatch_scalarize(scalarizer, scores, normalized_weights, **kwargs)