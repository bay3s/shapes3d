from src.utils.data.loader import Loader as loader
from src.utils.data.sources import Sources as sources
from src.iterative_closest_point import IterativeClosestPoint as icp

source_rabbits, target_rabbits = loader.get(sources.DATA_SOURCE_RABBITS)
rotations, translations = icp.run(source_rabbits.T, target_rabbits.T)

pass

############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################


###### 0. (adding noise)


###### 1. initialize R= I , t= 0


###### go to 2. unless RMS is unchanged(<= epsilon)


###### 2. using different sampling methods


###### 3. transform point cloud with R and t


###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach


###### 5. Calculate RMS


###### 6. Refine R and t using SVD
