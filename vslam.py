import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import open3d as o3d
from scipy.optimize import least_squares

class Frame:
    def __init__(self, image, pose=None):
        self.image = image
        self.pose = pose if pose is not None else np.eye(4)
        self.kp = None
        self.des = None
        self.points3D = []
        self.is_keyframe = False

def match_features(des1, des2):
    if des1 is None or des2 is None:
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return []
    
    good_matches = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def compute_parallax(matches, kp1, kp2):
    if len(matches) < 8:
        return 0
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    distances = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))
    return np.mean(distances)

def reprojection_loss_function(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3,4)
    
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]
        rep_error.append(pt_2d - reprojected_pt[0:2])

    return np.array(rep_error).ravel()

def bundle_adjustment(points_3d, points_2d, projection_matrix):
    opt_variables = np.hstack((projection_matrix.ravel(), points_3d.ravel(order="F")))
    num_points = len(points_2d[0])
    corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(points_2d,num_points))
    P = corrected_values.x[0:12].reshape(3,4)
    points_3d = corrected_values.x[12:].reshape((num_points, 4))
    return P, points_3d

def triangulate_points(kp1, kp2, matches, R, T, camera_matrix):
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = np.hstack((R, T))
    proj1 = camera_matrix @ proj1
    proj2 = camera_matrix @ proj2
    
    points_4d = cv2.triangulatePoints(proj1, proj2, points1.T, points2.T)

    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    return points_3d

def scale_points_to_height(points_3d, reference_height=21.0, z_threshold=100.0):
    valid_points = points_3d[:, 2] > 0
    if np.sum(valid_points) == 0:
        raise ValueError("No valid 3D points found.")

    current_height = np.mean(points_3d[valid_points, 2])
    scale_factor = reference_height / current_height
    scaled_points = points_3d * scale_factor
    return scaled_points

def estimate_motion(matches, kp1, kp2, K):
    if len(matches) < 8:
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
    
    return R, t

def is_keyframe(curr_frame, prev_keyframe, min_parallax=20):
    if prev_keyframe is None:
        return True
        
    matches = match_features(prev_keyframe.des, curr_frame.des)
    if len(matches) < 10:
        return True
        
    parallax = compute_parallax(matches, prev_keyframe.kp, curr_frame.kp)
    return parallax > min_parallax

def save_trajectory_point(pose, frame_idx):
    with open('output/trajectory/trajectory.txt', 'a') as f:
        translation = pose[:3, 3]
        f.write(f'{frame_idx} {translation[0]} {translation[1]} {translation[2]}\n')

def numerical_sort(value):
    import re
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

def create_camera_visualization(poses):
    if not poses:
        return None
    
    points = []
    lines = []
    colors = []
    
    for pose in poses:
        points.append(pose[:3, 3])  
    
    for i in range(len(points) - 1):
        lines.append([i, i + 1])
        colors.append([1, 0, 0])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return line_set

def create_camera_mesh(scale=0.1):
    points = scale * np.array([
        [0, 0, 0],          
        [-1, -1, 2],        
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2]
    ])
    
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  
        [1, 2], [2, 3], [3, 4], [4, 1]  
    ])
    
    colors = np.array([[0, 1, 0] for _ in range(len(lines))])
    
    camera = o3d.geometry.LineSet()
    camera.points = o3d.utility.Vector3dVector(points)
    camera.lines = o3d.utility.Vector2iVector(lines)
    camera.colors = o3d.utility.Vector3dVector(colors)
    
    return camera

def final_visualization(map_points, camera_poses):
    if len(map_points) == 0:
        return
    
    pcd = o3d.geometry.PointCloud()
    points = np.array(map_points)
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points = np.asarray(pcd.points)
    
    camera_trajectory = create_camera_visualization(camera_poses)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.add_geometry(pcd)
    if camera_trajectory is not None:
        vis.add_geometry(camera_trajectory)
    
    for pose in camera_poses:
        camera_mesh = create_camera_mesh()
        camera_mesh.transform(pose)
        vis.add_geometry(camera_mesh)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.show_coordinate_frame = True
    
    vis.poll_events()
    vis.update_renderer()
    
    vis.run()
    vis.destroy_window()

def process_frame(frame_folder, camera_matrix, dist_coeffs): 
    open('output/trajectory/trajectory.txt', 'w').close()
    
    frame_files = sorted(os.listdir(frame_folder), key=numerical_sort)
    orb = cv2.ORB_create(1000,
         scaleFactor=1.2,                
        nlevels=8,
        edgeThreshold=19,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20)

    frames = []
    keyframes = []
    map_points = []
    
    current_pose = np.eye(4) 
    plt.figure(figsize=(10, 10))
    
    camera_poses = [] 
    
    i = 0
    for idx, frame_file in enumerate(frame_files):
        if idx >= 200:
            break
            
        print(f"Processing frame {idx}/{len(frame_files)}")
        
        
        new_frame = cv2.imread(os.path.join(frame_folder, frame_file),0)
        if new_frame is None:
            print(f"Frame {frame_file} can't be processed, skipping.")
            continue

        new_frame = cv2.resize(new_frame, (900, 600))

        new_frame = cv2.bilateralFilter(
                new_frame, d=9, sigmaColor=75, sigmaSpace=75)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        new_frame = clahe.apply(new_frame)
        
        frame = Frame(new_frame)
        frame.kp, frame.des = orb.detectAndCompute(new_frame, None)
        
        if idx > 0:
            good_matches = match_features(frames[-1].des, frame.des)
            
            if len(good_matches) > 0:
                R, t = estimate_motion(good_matches, frames[-1].kp, frame.kp, camera_matrix)
                Rt = np.eye(4)
                Rt[:3,:3] = R
                Rt[:3,3] = t.ravel()
                if Rt is not None:
                    current_pose = current_pose @ np.linalg.inv(Rt)
                    
                    frame.pose = current_pose.copy()
                    camera_poses.append(current_pose.copy())
                
                if len(keyframes) == 0 or is_keyframe(frame, keyframes[-1]):
                    i=i+1
                    frame.is_keyframe = True
                    keyframes.append(frame)
                    

                    if len(keyframes) > 1:
                        keyframe_matches = match_features(keyframes[-2].des, keyframes[-1].des)
                        
                        if len(keyframe_matches) > 0:
                            try:
                                points1 = np.float32([frames[-1].kp[m.queryIdx].pt for m in good_matches])
                                points2 = np.float32([frame.kp[m.trainIdx].pt for m in good_matches])

                                proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                                proj2 = np.hstack((R, t))
                                proj1 = camera_matrix @ proj1
                                proj2 = camera_matrix @ proj2

                                points_4d = cv2.triangulatePoints(proj1, proj2, points1.T, points2.T)                               
                                points_3d = points_4d[:3] / points_4d[3]
                                scaled_points_3d = scale_points_to_height(points_3d)
                                points_3d = scaled_points_3d.T
                                valid_points = points_3d[:, 2] > 0
                                filtered_points = points_3d[valid_points]
                                map_points.extend(points_3d)

                                if len(map_points) > 0:
                                    final_visualization(filtered_points, camera_poses)
                            except IndexError as e:
                                print(f"Matching error: {e}")
                                continue
                    
                    save_trajectory_point(current_pose, idx)
                
                img_matches = cv2.drawMatches(frames[-1].image, frames[-1].kp, 
                                            frame.image, frame.kp, good_matches, None, flags=2)
                cv2.imshow('Feature Matches', img_matches)
                cv2.imwrite(f'output/matches/matches_{idx:04d}.png', img_matches)
            
        frames.append(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    plt.close()

focal_length = [6089.317, 5839.989] 
principal_point = [ 967.128594, 539.250594] 

camera_matrix = np.array([[focal_length[0], 0, principal_point[0]],
              [0, focal_length[1], principal_point[1]],
              [0, 0, 1]])

dist_coeffs = np.array([[-0.594934073, -172.936636, -0.041924952,
                         -0.092356184, 7354.07094]])


if __name__ == "__main__":
    process_frame('frames', camera_matrix, dist_coeffs)