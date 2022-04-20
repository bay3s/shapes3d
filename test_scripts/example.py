from src.utils.data.loader import Loader as loader
from src.utils.data.sources import Sources as sources
from src.iterative_closest_point import IterativeClosestPoint as ICP

import numpy as np
import open3d as o3d


def open3d_example():
    pcd = o3d.io.read_point_cloud("../data/data/0000000000.pcd")
    # ## convert into ndarray

    pcd_arr = np.asarray(pcd.points)

    # ***  you need to clean the point cloud using a threshold ***
    pcd_arr_cleaned = pcd_arr

    # visualization from ndarray
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)

    o3d.visualization.draw_geometries([vis_pcd])


###### 0. (adding noise)


###### 1. initialize R= I , t= 0

###### go to 2. unless RMS is unchanged(<= epsilon)

###### 2. using different sampling methods

###### 3. transform point cloud with R and t

###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach

###### 5. Calculate RMS

###### 6. Refine R and t using SVD


############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################
source_rabbits, target_rabbits = loader.get(sources.DATA_SOURCE_RABBITS)
rotations, translations = ICP.run(source_rabbits.T, target_rabbits.T)

pass
