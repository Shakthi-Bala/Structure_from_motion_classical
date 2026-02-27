import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_poses
from DisambiguateCameraPose import disambiguate_cam_poses
from NonlinearTriangulation import nonlinear_triangulation
from PnPRANSAC import linearpnp
from NonlinearPnP import nonlinear_pnp
from BundleAdjustment import bundle_adjust_two_view
from ReprojectionReport import ReprojectionReporter, evaluate_and_visualize_step


def load_calibration(calibration_path):
	K = np.loadtxt(calibration_path, dtype=np.float64)
	if K.shape != (3, 3):
		raise ValueError(f"Calibration matrix must be 3x3. Got shape: {K.shape}")
	return K


def load_images(data_dir):
	image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
	if len(image_paths) < 2:
		raise RuntimeError(f"Need at least two images in {data_dir}")

	img1 = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
	img2 = cv2.imread(image_paths[1], cv2.IMREAD_COLOR)
	if img1 is None or img2 is None:
		raise RuntimeError("Could not load input images.")
	return img1, img2, image_paths[0], image_paths[1]


def main():
	phase1_dir = os.path.dirname(os.path.abspath(__file__))
	project_dir = os.path.dirname(phase1_dir)
	data_dir = os.path.join(project_dir, "P2Data")
	calibration_path = os.path.join(data_dir, "calibration.txt")
	out_dir = os.path.join(phase1_dir, "outputs", "reprojection")
	os.makedirs(out_dir, exist_ok=True)

	K = load_calibration(calibration_path)
	img1, img2, img_path1, img_path2 = load_images(data_dir)

	print(f"Using images: {img_path1} and {img_path2}")

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	pts1, pts2, _, _, _ = sift_match(gray1, gray2, ratio=0.75)
	if pts1.shape[0] < 8:
		raise RuntimeError("Not enough matches to estimate F. Need >= 8.")

	F, inlier_mask, in1, in2 = get_inliers_ransac(
		pts1,
		pts2,
		num_iters=3000,
		threshold=1e-3,
		seed=42,
	)

	print(f"Inliers: {int(np.sum(inlier_mask))} / {pts1.shape[0]}")

	E = essential_matrix_from_fundamental(F, K)
	poses = extract_camera_poses(E)

	C1 = np.zeros(3, dtype=np.float64)
	R1 = np.eye(3, dtype=np.float64)

	C2_lin, R2_lin, X_lin, _ = disambiguate_cam_poses(K, poses, in1, in2)

	reporter = ReprojectionReporter()

	evaluate_and_visualize_step(
		reporter,
		"Linear Triangulation",
		"Cam1",
		img1,
		X_lin,
		in1,
		K,
		R1,
		C1,
		os.path.join(out_dir, "01_linear_triangulation_cam1.png"),
	)
	evaluate_and_visualize_step(
		reporter,
		"Linear Triangulation",
		"Cam2",
		img2,
		X_lin,
		in2,
		K,
		R2_lin,
		C2_lin,
		os.path.join(out_dir, "02_linear_triangulation_cam2.png"),
	)

	X_nonlin = nonlinear_triangulation(K, C1, R1, C2_lin, R2_lin, in1, in2, X_lin)

	evaluate_and_visualize_step(
		reporter,
		"Nonlinear Triangulation",
		"Cam1",
		img1,
		X_nonlin,
		in1,
		K,
		R1,
		C1,
		os.path.join(out_dir, "03_nonlinear_triangulation_cam1.png"),
	)
	evaluate_and_visualize_step(
		reporter,
		"Nonlinear Triangulation",
		"Cam2",
		img2,
		X_nonlin,
		in2,
		K,
		R2_lin,
		C2_lin,
		os.path.join(out_dir, "04_nonlinear_triangulation_cam2.png"),
	)

	_, R2_pnp_lin, C2_pnp_lin = linearpnp(X_nonlin, in2, K)

	evaluate_and_visualize_step(
		reporter,
		"Linear PnP",
		"Cam2",
		img2,
		X_nonlin,
		in2,
		K,
		R2_pnp_lin,
		C2_pnp_lin,
		os.path.join(out_dir, "05_linear_pnp_cam2.png"),
	)

	R2_pnp_nonlin, C2_pnp_nonlin = nonlinear_pnp(X_nonlin, in2, K, R2_pnp_lin, C2_pnp_lin)

	evaluate_and_visualize_step(
		reporter,
		"Nonlinear PnP (Before BA)",
		"Cam2",
		img2,
		X_nonlin,
		in2,
		K,
		R2_pnp_nonlin,
		C2_pnp_nonlin,
		os.path.join(out_dir, "06_nonlinear_pnp_before_ba_cam2.png"),
	)

	R2_ba, C2_ba, X_ba = bundle_adjust_two_view(
		K,
		C1,
		R1,
		C2_pnp_nonlin,
		R2_pnp_nonlin,
		in1,
		in2,
		X_nonlin,
	)

	evaluate_and_visualize_step(
		reporter,
		"Bundle Adjustment (After BA)",
		"Cam1",
		img1,
		X_ba,
		in1,
		K,
		R1,
		C1,
		os.path.join(out_dir, "07_after_ba_cam1.png"),
	)
	evaluate_and_visualize_step(
		reporter,
		"Bundle Adjustment (After BA)",
		"Cam2",
		img2,
		X_ba,
		in2,
		K,
		R2_ba,
		C2_ba,
		os.path.join(out_dir, "08_after_ba_cam2.png"),
	)

	reporter.print_table()
	csv_path = os.path.join(out_dir, "reprojection_report.csv")
	reporter.save_csv(csv_path)

	print(f"Saved reprojection report CSV: {csv_path}")
	print(f"Saved reprojection overlays in: {out_dir}")


if __name__ == "__main__":
	main()