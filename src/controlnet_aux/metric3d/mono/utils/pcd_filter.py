import open3d as o3d
import numpy as np

def downsample_and_filter(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file, max_bound_div = 750, neighbor_num = 8)
    point_num = len(pcd.points)
    if (point_num > 10000000):
        voxel_down_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, int(point_num / 10000000)+1)
    else:
        voxel_down_pcd = pcd
    max_bound = voxel_down_pcd.get_max_bound()
    ball_radius = np.linalg.norm(max_bound) / max_bound_div
    pcd_filter, _ = voxel_down_pcd.remove_radius_outlier(neighbor_num, ball_radius)
    print('filtered size', len(pcd_filter.points), 'pre size:', len(pcd.points))
    o3d.io.write_point_cloud(pcd_file[:-4] + '_filtered.ply', pcd_filter)


if __name__ == "__main__":
    import os
    dir_path = './data/demo_pcd'
    for pcd_file in os.listdir(dir_path):
        #if 'jonathan' in pcd_file: set max_bound_div to 300 and neighbot_num to 8
        downsample_and_filter(os.path.join(dir_path, pcd_file))
        