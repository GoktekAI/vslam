# Monocular Visual SLAM (Simultaneous Localization and Mapping)

This repository contains a Python implementation of a monocular visual SLAM system. It takes a sequence of images from a single camera, estimates the camera's trajectory, and builds a sparse 3D map of the environment.

## Features

* **Feature Detection and Matching:** Uses ORB (Oriented FAST and Rotated BRIEF) for robust feature detection and matching between consecutive frames.
* **Motion Estimation:** Employs essential matrix decomposition and homography estimation with RANSAC to estimate camera motion (rotation and translation).
* **Triangulation:** Reconstructs 3D points by triangulating matched features across multiple views.
* **Keyframe Selection:** Implements a parallax-based keyframe selection strategy to ensure sufficient baseline for accurate triangulation and reduce computational cost.
* **Bundle Adjustment (BA):** Refines camera poses and 3D point locations by minimizing reprojection errors, improving map and trajectory consistency.
* **Scale Estimation:** Includes a function to scale the 3D points based on a reference height. Useful when other scale information is unavailable.
* **Visualization:** Uses Open3D to visualize the reconstructed 3D points and the estimated camera trajectory.
* **Trajectory Saving:** Saves the estimated camera trajectory to a text file.
* **Image Preprocessing:** Includes bilateral filtering and CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image quality and improve feature detection.

## Requirements

* Python 3.x
* NumPy
* OpenCV (cv2)
* Matplotlib
* Open3D
* SciPy

## Usage
* Data: Place your image sequence in a folder named frames in the same directory as the script. Images should be named numerically (e.g., 0001.png, 0002.png).

* Calibration: Update the camera_matrix and dist_coeffs variables in the script with your camera's intrinsic parameters. These are crucial for accurate 3D reconstruction. See the code for placeholder values.

* Run: Execute the script: python your_script_name.py

* The script will process the images, estimate the camera trajectory, and visualize the results.

# Output
* Trajectory: The estimated camera trajectory is saved to output/trajectory/trajectory.txt. Each line represents a frame and its 3D position (x, y, z).

* Feature Matches: Visualizations of feature matches are saved to output/matches/.

* 3D Visualization: An Open3D window displays the reconstructed sparse 3D point cloud and camera trajectory.

# Important Considerations
* Scale: The scale_points_to_height function uses a reference height of 21.0. Adjust this to your scene.

* Image Format: The code assumes grayscale images for ORB. Convert color images to grayscale before processing.

* Keyframe Selection: The is_keyframe function's minimum parallax threshold (20) controls keyframe frequency. Adjust for a balance between accuracy and computation.

* Processing Limit: The code processes the first 200 frames by default. Remove or modify if idx >= 200: break to process the entire sequence.

# Future Work
* Loop Closure: Implement loop closure detection and optimization.

* Feature Options: Add support for other feature detectors/descriptors (SIFT, SURF).

* Scale Estimation: Explore more robust scale estimation methods.

# Contributors

* Nursen Marancı

* Sena Varıcı
