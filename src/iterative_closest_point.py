import numpy as np


class IterativeClosestPoint:

  @staticmethod
  def run(source: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param source:
    :type source: np.ndarray
    :param target:
    :type target: np.ndarray
    :return: Returns a tuple consisting of the rotational matrix and the translation matrix
    :rtype: tuple
    """
    rotation, translation = IterativeClosestPoint.compute(source, target)

    return rotation, translation

  @staticmethod
  def compute(source: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray):
    # find the closest point in the target cloud for each point in the source point set
    closest_targets = IterativeClosestPoint.get_nearest_targets(source, target)

    assert source.shape == closest_targets.shape

    # use the SVD method to find the rotation and translation matrices
    source_weighted_mean = np.sum(source, axis=0) / len(source)
    target_weighted_mean = np.sum(closest_targets, axis=0) / len(closest_targets)

    mean_reduced_source = source - source_weighted_mean
    mean_reduced_target = closest_targets - target_weighted_mean
    cross_covariance = np.dot(mean_reduced_source.T, mean_reduced_target)

    u, d, v_t = np.linalg.svd(cross_covariance)
    rotation = np.dot(v_t.T, u.T)
    translation = target_weighted_mean.T - np.dot(rotation, source_weighted_mean.T)

    return rotation, translation

  @staticmethod
  def get_nearest_targets(source: np.ndarray, target: np.ndarray):
    max_range = len(target) if len(source) > len(target) else len(source)
    closest = list()

    for i in range(len(source)):
      distances = list()

      for j in range(max_range):
        distances.append(np.linalg.norm(target[j] - source[i]))

      closest_point = target[np.argmin(np.array(distances))]
      closest.append(closest_point)

    return np.array(closest)
