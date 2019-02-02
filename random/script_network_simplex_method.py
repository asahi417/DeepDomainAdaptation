""" Test optimal transport library

"""

import ot
import numpy as np


def test(source_samples,
         target_samples,
         weight_function):
    """
    :param source_samples: array (n_source, feature)
    :param target_samples: array (n_target, feature)
    :param weight_function: function determine distance between two samples
    :return:
    """

    assert source_samples.shape[1] == target_samples.shape[1]

    # Employ uniform distribution over all data as empirical distribution (not a histogram)
    source_dist = np.ones((len(source_samples), )) / len(source_samples)
    target_dist = np.ones((len(target_samples), )) / len(target_samples)
    # print('source:', source_dist.shape, np.sum(source_dist))
    # print('target:', target_dist.shape, np.sum(target_dist))

    # build cost matrix (n_source, n_target)
    cost_matrix = np.array([[float(weight_function(__i, __o)) for __i in target_samples] for __o in source_samples])
    print('cost :\n', cost_matrix, cost_matrix.shape)
    # derive optimal transport based on network simplex algorithm
    # Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011, December).
    # Displacement interpolation using Lagrangian mass transport.
    # In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.
    optimal_transport = ot.emd(a=source_dist, b=target_dist, M=cost_matrix)
    return optimal_transport


if __name__ == '__main__':
    np.random.seed(0)
    source_data = np.random.randint(5, 10, (2, 2))
    target_data = np.random.randint(0, 5, (4, 2))
    print('source :\n', source_data, source_data.shape)
    print('target :\n', target_data, target_data.shape)

    def distance(x, y): return np.abs(np.sum(x-y))

    o = test(source_data, target_data, distance)

    print('ot :\n', o)
    # print(o * source_data)


