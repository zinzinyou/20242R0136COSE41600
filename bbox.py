# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2


# pcd 파일 불러오기, 필요에 맞게 경로 수정
scenario = "01_straight_walk"
scenario = "02_straight_duck_walk"
scenario = "03_straight_crawl"
scenario = "04_zigzag_walk"
scenario = "05_straight_duck_walk"
scenario = "06_straight_crawl"
scenario = "07_straight_walk"
pcd_dir = f"./data/{scenario}/pcd/"
pcd_files = sorted([os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith(".pcd")])

# 모든 PCD 파일 로드
print("Loading PCD files...")
pcd_list = [o3d.io.read_point_cloud(file) for file in tqdm(pcd_files)]


# 시각화 프레임 저장 경로
output_dir = f"./output_frames/{scenario}/"
os.makedirs(output_dir, exist_ok=True)

# 동영상 저장 경로
output_video = f"./output_video/{scenario}.mp4"
os.makedirs(os.path.dirname(output_video), exist_ok=True)

# 누적 결과 저장 경로
output_image = f"./output_image/{scenario}_output.png"
os.makedirs(os.path.dirname(output_image), exist_ok=True)

####################################################################################################################################

# pcd 전처리
def process_pcd(original_pcd):
    # Voxel Downsampling 수행
    voxel_size = 0.2 # 필요에 따라 voxel 크기를 조정하세요.
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=5, radius=1.0)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.05,
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
    vis.poll_events()
    vis.update_renderer()

    # 결과 저장
    vis.capture_screen_image(output_image)
    print(f"Result image saved at {output_image}")

    vis.run()
    vis.destroy_window()

# Open3D 시각화 저장 함수
def save_frame(pcd, bounding_boxes, frame_id, output_dir):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 창 표시 비활성화
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = 1.0
    vis.poll_events()
    vis.update_renderer()

    # 프레임 저장
    frame_path = os.path.join(output_dir, f"frame_{frame_id:04d}.png")
    vis.capture_screen_image(frame_path, do_render=True)
    vis.destroy_window()

####################################################################################################################################
prev_pcd = None
merged_pcd = o3d.geometry.PointCloud()
clusters = [] 
all_bbox = []
curr_bbox = []


# pcd 파일 연속적으로 불러오기
for idx, pcd in tqdm(enumerate(pcd_list), desc="Processing PCD files"):

    curr_pcd = process_pcd(pcd)
    if idx%5==0:
        curr_bbox = []

    if prev_pcd is not None and idx%5==0:
        # 이전 pcd와 비교하여 움직이는 point 탐지
        moving_pcd = get_moving_objects(prev_pcd, curr_pcd, threshold=0.15)

        # # 포인트 개수 비교
        # total_points = len(curr_pcd.points)
        # moving_points = len(moving_pcd.points)
        # moving_ratio = (moving_points / total_points) * 100 if total_points > 0 else 0
        # print(f"Frame {idx}:")
        # print(f"  Total points: {total_points}")
        # print(f"  Moving points: {moving_points} ({moving_ratio:.2f}%)")
        
        if len(moving_pcd.points)>0:
            # 움직이는 point들만 대상으로 DBSCAN 클러스터링 적용
            labels = np.array(moving_pcd.cluster_dbscan(eps=0.4, min_points=5, print_progress=False))

            # 시각화 결과에서 노이즈 제거
            # valid_indices = labels >= 0
            # if np.sum(valid_indices) == 0:
            #     print(f"Frame {idx}: No valid clusters found.")
            #     prev_pcd = curr_pcd
            #     continue
            # moving_pcd = moving_pcd.select_by_index(np.where(valid_indices)[0])

            # labels = labels[valid_indices]  # 노이즈 제외 후 라벨 갱신

            # cluster 개수 계산
            num_cluster = labels.max()+1
            clusters.append(num_cluster)

            # 각 cluster에 고유 색상 할당
            # max_label = num_cluster-1
            # cmap = plt.get_cmap("tab20")  
            # colors = np.zeros((len(labels), 3))  # 기본 검정색
            # for i in range(max_label + 1):
            #     cluster_color = cmap(i / (max_label if max_label > 0 else 1))[:3]  # RGB 값만 추출
            #     colors[labels == i] = cluster_color

            # 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
            colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
            colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

            moving_pcd.colors = o3d.utility.Vector3dVector(colors)

            # 통합 PointCloud에 추가
            merged_pcd += moving_pcd

            # 필터링 기준 설정
            min_points_in_cluster = 7   # 클러스터 내 최소 포인트 수
            max_points_in_cluster = 100 # 클러스터 내 최대 포인트 수
            min_z_value = -1.0          # 클러스터 내 최소 Z값
            max_z_value = 4.0           # 클러스터 내 최대 Z값
            min_height = 0.3            # Z값 차이의 최소값
            max_height = 2.5            # Z값 차이의 최대값
            max_distance = 50.0         # 원점으로부터의 최대 거리

            # X, Y 필터링 기준 추가
            min_width = 0.2             # X값 차이의 최소값
            max_width = 1.0             # X값 차이의 최대값
            min_depth = 0.2             # Y값 차이의 최소값
            max_depth = 1.0             # Y값 차이의 최대값

            # 클러스터 필터링 및 바운딩 박스 생성
            for i in range(num_cluster):
                cluster_indices = np.where(labels == i)[0]
                if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
                    cluster_pcd = moving_pcd.select_by_index(cluster_indices)
                    points = np.asarray(cluster_pcd.points)
                    x_values = points[:,0]
                    y_values = points[:,1]
                    z_values = points[:,2]

                    x_min, x_max = x_values.min(), x_values.max()
                    y_min, y_max = y_values.min(), y_values.max()
                    z_min, z_max = z_values.min(), z_values.max()

                    if min_z_value <= z_min and z_max <= max_z_value:
                        width_diff = x_max - x_min
                        depth_diff = y_max - y_min
                        height_diff = z_max - z_min
                        if (min_height <= height_diff <= max_height and
                            min_width <= width_diff <= max_width and
                            min_depth <= depth_diff <= max_depth):
                            distances = np.linalg.norm(points, axis=1)
                            if distances.max() <= max_distance:
                                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                                bbox.color = (1, 0, 0) 
                                curr_bbox.append(bbox)
                                all_bbox.append(bbox)
    save_frame(curr_pcd, curr_bbox, idx, output_dir)
    if(idx%5==0):
        prev_pcd = curr_pcd

# 통합 결과 시각화
# visualize_with_bounding_boxes(merged_pcd, all_bbox, window_name="Moving Objects", point_size=1.0)
# print("average number of clusters:",np.mean(clusters))

# 프레임을 동영상으로 변환
frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")])
frame_size = cv2.imread(frame_files[0]).shape[1::-1]
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, frame_size)

for frame_file in tqdm(frame_files, desc="Creating Video"):
    frame = cv2.imread(frame_file)
    out.write(frame)

out.release()
print(f"Video saved at {output_video}")