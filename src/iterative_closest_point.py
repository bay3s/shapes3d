import numpy as np
import open3d as o3d


class IterativeClosestPoint:

  @staticmethod
  def run(source: np.ndarray, target: np.ndarray) -> list:
    """
    :param source:
    :type source: np.ndarray
    :param target:
    :type target: np.ndarray
    :return: Returns a tuple consisting of the rotational matrix and the translation matrix
    :rtype: tuple
    """
    self = IterativeClosestPoint
    error_progression = list()

    # find the closest point in the target cloud for each point in the source point set
    assigned_target = self.assign_targets(source, target)

    for i in range(1000):
      rotation, translation = self.compute_rotation_translation(source, assigned_target)
      transformed = np.matmul(source, rotation) + translation

      error = self.calculate_error(transformed, assigned_target)
      error_progression.append(error)
      print(error)

      if error < 0.05 and i > 0:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(transformed)
        o3d.visualization.draw_geometries([point_cloud])

      source = transformed

  @staticmethod
  def compute_rotation_translation(source: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray):
    self = IterativeClosestPoint

    assert source.shape == target.shape

    # use the SVD method to find the rotation and translation matrices
    source_weighted_mean = np.sum(source, axis=0) / len(source)
    target_weighted_mean = np.sum(target, axis=0) / len(target)

    mean_reduced_source = source - source_weighted_mean
    mean_reduced_target = target - target_weighted_mean
    cross_covariance = np.dot(mean_reduced_source.T, mean_reduced_target)

    u, d, v_t = np.linalg.svd(cross_covariance)

    rotation = np.dot(v_t.T, u.T)
    translation = target_weighted_mean.T - np.dot(rotation, source_weighted_mean.T)

    return rotation, translation

  @staticmethod
  def assign_targets(source: np.ndarray, target: np.ndarray):
    closest_points = list()

    for i in range(len(source)):
      distances = np.sum((target - source[i]) ** 2, axis = 1)
      current_min = np.argmin(distances)
      closest_point = target[current_min]
      closest_points.append(closest_point)

    return np.array(closest_points)

  @staticmethod
  def calculate_error(calculated: np.ndarray, target: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((calculated - target) ** 2) ** 0.5))
