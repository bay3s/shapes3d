import numpy as np
import open3d as o3d
import os
from src.utils.data.sources import Sources


class Loader:

  @staticmethod
  def get(data_source: Sources) -> (np.ndarray, np.ndarray):
    if not isinstance(data_source, Sources):
      raise TypeError('data_source must be an instance of DataSources Enum')

    if data_source == Sources.DATA_SOURCE_OPEN3D:
      # @todo update to return data from open3d
      source, target = None, None
    elif data_source == Sources.DATA_SOURCE_RABBITS:
      source, target = Loader.get_bunny_data()
    elif data_source == Sources.DATA_SOURCE_WAVES:
      source, target = Loader.get_wave_data()

    return source, target

  @staticmethod
  def open3d_example(data_path):
    point_cloud_data = o3d.io.read_point_cloud(f'{data_path}data/0000000000.pcd')

    point_cloud_data_arr = np.asarray(point_cloud_data.points)

    # @todo need to clean the point cloud using a threshold
    point_cloud_data_arr_cleaned = point_cloud_data_arr

    # visualization from ndarray
    point_cloud_data_visualization = o3d.geometry.PointCloud()
    point_cloud_data_visualization.points = o3d.utility.Vector3dVector(point_cloud_data_arr_cleaned)
    o3d.visualization.draw_geometries([point_cloud_data_visualization])

  @staticmethod
  def get_wave_data() -> (np.ndarray, np.ndarray):
    data_path = Loader.get_data_path()
    source = np.load(os.path.join(data_path, 'wave_source.npy'))
    target = np.load(os.path.join(data_path, 'wave_target.npy'))

    return source, target

  @staticmethod
  def get_bunny_data() -> (np.ndarray, np.ndarray):
    data_path = Loader.get_data_path()
    source = np.load(os.path.join(data_path, 'bunny_source.npy'))
    target = np.load(os.path.join(data_path, 'bunny_target.npy'))

    return source, target

  @staticmethod
  def get_data_path() -> str:
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, './../../../data')

    return data_path
