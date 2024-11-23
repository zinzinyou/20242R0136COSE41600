# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# pcd 파일 불러오기, 필요에 맞게 경로 수정
scenario = "01_straight_walk"
# scenario = "02_straight_duck_walk"
# scenario = "03_straight_crawl"
scenario = "04_zigzag_walk"

pcd_dir = f"./data/{scenario}/pcd/"
pcd_files = sorted([os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith(".pcd")])

####################################################################################################################################

# pcd 전처리
def process_pcd(file_path):
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling 수행
    voxel_size = 0.25 # 필요에 따라 voxel 크기를 조정하세요.
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=2000)

    # 도로에 속하지 않는 포인트 (outliers) 추출
    final_point = ror_pcd.select_by_index(inliers, invert=True)
    return final_point

def get_moving_objects(pcd_prev, pcd_curr, threshold):
    # KDTree 생성
    prev_tree = o3d.geometry.KDTreeFlann(pcd_prev)

    moving_points = []
    prev_points = np.asarray(pcd_prev.points)

    for point in np.asarray(pcd_curr.points):
        # KDTree로 최근접 이웃 탐색
        [_, idx, distances] = prev_tree.search_knn_vector_3d(point, 1)
        if distances[0] > threshold:
            moving_points.append(point)

    # 움직이는 포인트 클라우드 생성
    moving_pcd = o3d.geometry.PointCloud()
    moving_pcd.points = o3d.utility.Vector3dVector(np.array(moving_points))
    return moving_pcd



# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

####################################################################################################################################
prev_pcd = None
merged_pcd = o3d.geometry.PointCloud()
clusters = [] # cluster 개수
all_bbox = []


# pcd 파일 연속적으로 불러오기
for idx, file_path in tqdm(enumerate(pcd_files), desc="Processing PCD files"):
    # if idx>10: break
    if idx%5!=0:
        continue

    curr_pcd = process_pcd(file_path)

    if prev_pcd is not None:
        # 움직이는 객체 탐지
        # moving_pcd = get_moving_objects(prev_pcd, curr_pcd, threshold=0.2)
        moving_pcd = get_moving_objects(prev_pcd, curr_pcd, threshold=0.2)

        # DBSCAN 클러스터링 적용
        if len(moving_pcd.points)>0:
            # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(moving_pcd.cluster_dbscan(eps=0.35, min_points=10, print_progress=False))
            num_cluster = labels.max()+1
            clusters.append(num_cluster)

            # 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
            colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
            colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

            moving_pcd.colors = o3d.utility.Vector3dVector(colors)

            # 통합 PointCloud에 추가
            merged_pcd += moving_pcd

            # 필터링 기준 설정
            min_points_in_cluster = 10   # 클러스터 내 최소 포인트 수
            max_points_in_cluster = 100  # 클러스터 내 최대 포인트 수
            min_z_value = -1.0          # 클러스터 내 최소 Z값
            max_z_value = 5.0           # 클러스터 내 최대 Z값
            min_height = 0.5            # Z값 차이의 최소값
            max_height = 3.0            # Z값 차이의 최대값
            max_distance = 50.0         # 원점으로부터의 최대 거리

            # 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
            # bboxes_1234 = []
            for i in range(labels.max() + 1):
                cluster_indices = np.where(labels == i)[0]
                if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
                    cluster_pcd = moving_pcd.select_by_index(cluster_indices)
                    points = np.asarray(cluster_pcd.points)
                    z_values = points[:, 2]
                    z_min = z_values.min()
                    z_max = z_values.max()
                    if min_z_value <= z_min and z_max <= max_z_value:
                        height_diff = z_max - z_min
                        if min_height <= height_diff <= max_height:
                            distances = np.linalg.norm(points, axis=1)
                            if distances.max() <= max_distance:
                                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                                bbox.color = (1, 0, 0) 
                                # bboxes_1234.append(bbox)
                                all_bbox.append(bbox)

        # 시각화 (포인트 크기를 원하는 크기로 조절 가능)
        # visualize_with_bounding_boxes(moving_pcd, bboxes_1234, point_size=2.0)
    
    prev_pcd = curr_pcd

# 통합 결과 시각화
visualize_with_bounding_boxes(merged_pcd, all_bbox, window_name="Merged Clusters with Bounding Boxes", point_size=1.0)
# o3d.visualization.draw_geometries([merged_pcd], window_name="Merged Clusters")
print("average number of clusters:",np.mean(clusters))