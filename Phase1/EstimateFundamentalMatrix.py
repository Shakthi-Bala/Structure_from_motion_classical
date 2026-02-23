import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac, estimate_fundamental_matrix



def _load_two_images_from_folder(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)

    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images in folder: %s" % folder)

    img1 = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    img2 = cv2.imread(paths[1], cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise RuntimeError("Failed to read one of the images: %s, %s" % (paths[0], paths[1]))

    return img1, img2, paths[0], paths[1]


def _draw_inlier_matches(img1, img2, kp1, kp2, matches, inlier_mask, max_draw=200):
    # Build inlier match list
    inlier_matches = [m for m, keep in zip(matches, inlier_mask.tolist()) if keep]
    inlier_matches = sorted(inlier_matches, key=lambda m: m.distance)
    if len(inlier_matches) > max_draw:
        inlier_matches = inlier_matches[:max_draw]

    vis = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis


def _draw_epipolar_lines(img1, img2, F, pts1, pts2, num=15):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1c = img1.copy()
    img2c = img2.copy()

    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    if pts1.shape[0] == 0:
        return img1c, img2c

    # sample a few points
    idx = np.linspace(0, pts1.shape[0] - 1, min(num, pts1.shape[0])).astype(int)
    p1s = pts1[idx]
    p2s = pts2[idx]

    # lines in img2 from pts1: l' = F x
    x1 = np.hstack([p1s, np.ones((p1s.shape[0], 1))])
    lines2 = (F.dot(x1.T)).T  # (N,3)

    # lines in img1 from pts2: l = F^T x'
    x2 = np.hstack([p2s, np.ones((p2s.shape[0], 1))])
    lines1 = (F.T.dot(x2.T)).T  # (N,3)

    # draw helper
    def draw_lines(img, lines, pts, w, h):
        out = img.copy()
        for (a, b, c), (x, y) in zip(lines, pts):
            # line: aX + bY + c = 0
            if abs(b) > 1e-12:
                y0 = int(round((-c - a * 0) / b))
                yW = int(round((-c - a * (w - 1)) / b))
                pt1 = (0, y0)
                pt2 = (w - 1, yW)
            else:
                # vertical line
                x0 = int(round((-c) / (a + 1e-12)))
                pt1 = (x0, 0)
                pt2 = (x0, h - 1)

            # clip not strictly needed; OpenCV will handle, but keep sane
            cv2.line(out, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
        return out

    img2c = draw_lines(img2c, lines2, p2s, w2, h2)
    img1c = draw_lines(img1c, lines1, p1s, w1, h1)
    return img1c, img2c

def main():
    folder = "./images"  
    img1, img2, p1, p2 = _load_two_images_from_folder(folder)
    print("Using images:\n  1) %s\n  2) %s" % (p1, p2))

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 1) SIFT matching
    pts1, pts2, matches, kp1, kp2 = sift_match(g1, g2, ratio=0.75)
    print("Raw matches:", pts1.shape[0])

    if pts1.shape[0] < 8:
        raise RuntimeError("Not enough matches to estimate F. Need >= 8.")


    F_best, inlier_mask, in1, in2 = get_inliers_ransac(
        pts1, pts2,
        num_iters=3000,
        threshold=1e-3,
        seed=42
    )
    print("Inliers:", int(np.sum(inlier_mask)), "/", pts1.shape[0])
    print("F (RANSAC best/refit):\n", F_best)

    if in1.shape[0] >= 8:
        F_inliers = estimate_fundamental_matrix(in1, in2)
        print("F (re-estimated from inliers):\n", F_inliers)
    else:
        F_inliers = F_best

    vis_matches = _draw_inlier_matches(img1, img2, kp1, kp2, matches, inlier_mask, max_draw=250)
    cv2.imshow("Inlier Matches (after RANSAC)", vis_matches)

    epi1, epi2 = _draw_epipolar_lines(img1, img2, F_inliers, in1, in2, num=15)
    cv2.imshow("Epipolar lines in Image 1", epi1)
    cv2.imshow("Epipolar lines in Image 2", epi2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()