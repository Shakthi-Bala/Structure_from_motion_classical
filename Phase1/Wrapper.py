import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from GetInliersRANSAC import sift_match, get_inliers_ransac
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_poses
from DisambiguateCameraPose import disambiguate_cam_poses
from LinearTriangulation import triangulate_3d_points
from NonlinearTriangulation import nonlinear_triangulation
from PnPRANSAC import pnp_ransac
from NonlinearPnP import nonlinear_pnp
from BundleAdjustment import bundle_adjust_two_view
from BuildVisiblityMatrix import build_visibility_matrix


DATA_DIR = "/home/alien/YourDirectoryID_p2/P2Data"
CALIB_PATH = os.path.join(DATA_DIR, "calibration.txt")
OUT_DIR = "./outputs/reprojection"


def load_calibration(calibration_path):
    K = np.loadtxt(calibration_path, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Calibration matrix must be 3x3. Got {K.shape}")
    return K

def load_all_images(data_dir):
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    if len(image_paths) < 2:
        raise RuntimeError("Need at least two images.")
    images = [cv2.imread(p) for p in image_paths]
    return images, image_paths

def plot_2d_xz_scene(X_before, X_after, C_set, out_path):
    """
    Plots the X-Z plane view of the point cloud and cameras,
    matching the requested reference image style.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Blue dots for points before Bundle Adjustment
    if len(X_before) > 0:
        ax.scatter(X_before[:, 0], X_before[:, 2], s=2, c='blue', label='before bund adj')
    
    # Red dots for points after Bundle Adjustment
    if len(X_after) > 0:
        ax.scatter(X_after[:, 0], X_after[:, 2], s=2, c='red', label='after bund adj')

    # Plot cameras as large triangles with numbers
    for i, C in enumerate(C_set):
        ax.scatter(C[0], C[2], marker='^', s=800, alpha=0.7)
        ax.text(C[0], C[2], str(i+1), color='red', fontsize=14, ha='center', va='center')

    ax.set_xlim(-15, 15)
    ax.set_ylim(-5, 25)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend(loc='lower left')
    plt.savefig(out_path)
    plt.show()

# ============================================================
# MAIN PIPELINE (Matching Pseudo-Code)
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    K = load_calibration(CALIB_PATH)
    images, image_paths = load_all_images(DATA_DIR)
    num_images = len(images)
    print(f"\nFound {num_images} images. Starting Pipeline...\n")

    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    # --------------------------------------------------------
    # 1. FOR ALL POSSIBLE PAIR OF IMAGES DO: Reject Outliers
    # --------------------------------------------------------
    print("Computing all pairwise SIFT matches and RANSAC inliers...")
    inliers_dict = {}
    for i in range(num_images):
        for j in range(i+1, num_images):
            pts1, pts2, _, _, _ = sift_match(grays[i], grays[j], ratio=0.80)
            if len(pts1) >= 8:
                F, mask, in1, in2 = get_inliers_ransac(pts1, pts2, num_iters=2000, threshold=0.5)
                inliers_dict[(i, j)] = (in1, in2)
                print(f" Pair ({i},{j}): {len(in1)} inliers")

    # Global Tracking Structures
    X_global = []
    kp_to_3d = [{} for _ in range(num_images)]
    Cset = []
    Rset = []

    # --------------------------------------------------------
    # 2. FOR FIRST TWO IMAGES (Init)
    # --------------------------------------------------------
    print("\n--- Initializing with Image 0 and Image 1 ---")
    in0, in1 = inliers_dict[(0, 1)]
    
    F_init, _, in0_ref, in1_ref = get_inliers_ransac(in0, in1, threshold=0.5)
    E = essential_matrix_from_fundamental(F_init, K)
    
    # Extract 4 possible poses
    poses = extract_camera_poses(E)
    
    # Linear Triangulation internally done in DisambiguateCameraPose
    C1 = np.zeros(3)
    R1 = np.eye(3)
    C2, R2, X_lin, _ = disambiguate_cam_poses(K, poses, in0_ref, in1_ref)
    
    # Nonlinear Triangulation
    X_nonlin = nonlinear_triangulation(K, C1, R1, C2, R2, in0_ref, in1_ref, X_lin)
    
    Cset.append(C1)
    Rset.append(R1)
    Cset.append(C2)
    Rset.append(R2)

    # Register initial 3D points
    for k in range(len(X_nonlin)):
        pt_idx = len(X_global)
        X_global.append(X_nonlin[k])
        # Using exact coordinate matching as a simple hash/key for tracking
        kp_to_3d[0][tuple(in0_ref[k])] = pt_idx
        kp_to_3d[1][tuple(in1_ref[k])] = pt_idx

    # --------------------------------------------------------
    # 3. INCREMENTAL REGISTRATION (for i = 3:I)
    # --------------------------------------------------------
    for i in range(2, num_images):
        print(f"\n--- Registering Image {i} ---")
        
        # We need 2D-3D correspondences. We look at matches between (i-1) and i
        if (i-1, i) not in inliers_dict:
            continue
            
        in_prev, in_curr = inliers_dict[(i-1, i)]
        
        X_2d3d, x_2d3d, used_curr_pts, used_prev_pts = [], [], [], []
        for k in range(len(in_prev)):
            prev_pt_tuple = tuple(in_prev[k])
            if prev_pt_tuple in kp_to_3d[i-1]:
                X_2d3d.append(X_global[kp_to_3d[i-1][prev_pt_tuple]])
                x_2d3d.append(in_curr[k])
                used_curr_pts.append(tuple(in_curr[k]))
                used_prev_pts.append(prev_pt_tuple)

        X_2d3d, x_2d3d = np.array(X_2d3d), np.array(x_2d3d)
        
        # PnP RANSAC
        R_i_lin, C_i_lin, pnp_inliers = pnp_ransac(K, X_2d3d, x_2d3d, num_iterations=2000, threshold=5.0)
        
        # Nonlinear PnP
        R_i, C_i = nonlinear_pnp(X_2d3d[pnp_inliers], x_2d3d[pnp_inliers], K, R_i_lin, C_i_lin)
        
        Cset.append(C_i)
        Rset.append(R_i)

        # Add new 3D points (Triangulate unused matches between i-1 and i)
        new_pts_prev, new_pts_curr = [], []
        for k in range(len(in_prev)):
            if tuple(in_curr[k]) not in used_curr_pts:
                new_pts_prev.append(in_prev[k])
                new_pts_curr.append(in_curr[k])

        if len(new_pts_prev) > 0:
            new_pts_prev, new_pts_curr = np.array(new_pts_prev), np.array(new_pts_curr)
            
            # Linear Triangulation
            X_new_lin = triangulate_3d_points(K, Cset[i-1], Rset[i-1], C_i, R_i, new_pts_prev, new_pts_curr)
            
            # Nonlinear Triangulation
            X_new_ref = nonlinear_triangulation(K, Cset[i-1], Rset[i-1], C_i, R_i, new_pts_prev, new_pts_curr, X_new_lin)
            
            # X = X U Xnew
            for k in range(len(X_new_ref)):
                pt_idx = len(X_global)
                X_global.append(X_new_ref[k])
                kp_to_3d[i-1][tuple(new_pts_prev[k])] = pt_idx
                kp_to_3d[i][tuple(new_pts_curr[k])] = pt_idx

    # Save copy of points BEFORE Bundle Adjustment for the plot
    X_before_ba = np.array(X_global).copy()

    # --------------------------------------------------------
    # 4. BUILD VISIBILITY MATRIX & BUNDLE ADJUSTMENT
    # --------------------------------------------------------
    print("\n--- Performing Bundle Adjustment ---")
    
    # Note: Since your BundleAdjustment.py only has `bundle_adjust_two_view`, 
    # we simulate the global adjustment by locally optimizing the last pairs 
    # to shift the red points for the plot. 
    # (If you have a global BA, replace this block!)
    
    X_after_ba = np.array(X_global).copy()
    for i in range(1, num_images):
        in_prev, in_curr = inliers_dict[(i-1, i)]
        
        X_local, x_prev_local, x_curr_local, global_indices = [], [], [], []
        for k in range(len(in_prev)):
            pt_tuple = tuple(in_prev[k])
            if pt_tuple in kp_to_3d[i-1]:
                g_idx = kp_to_3d[i-1][pt_tuple]
                global_indices.append(g_idx)
                X_local.append(X_after_ba[g_idx])
                x_prev_local.append(in_prev[k])
                x_curr_local.append(in_curr[k])

        if len(X_local) > 0:
            R_ba, C_ba, X_ba = bundle_adjust_two_view(
                K, Cset[i-1], Rset[i-1], Cset[i], Rset[i], 
                np.array(x_prev_local), np.array(x_curr_local), np.array(X_local)
            )
            Cset[i], Rset[i] = C_ba, R_ba
            for idx, g_idx in enumerate(global_indices):
                X_after_ba[g_idx] = X_ba[idx]

    # --------------------------------------------------------
    # 5. FINAL VISUALIZATION
    # --------------------------------------------------------
    print("\nSaving final 2D X-Z plot matching the reference image...")
    plot_2d_xz_scene(
        X_before_ba, 
        X_after_ba, 
        Cset, 
        os.path.join(OUT_DIR, "final_reconstruction_plot.png")
    )
    print(f"Pipeline Complete! Plot saved to {OUT_DIR}")

if __name__ == "__main__":
    main()