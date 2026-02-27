import os
import cv2
import numpy as np

from PnPRANSAC import project_points


class ReprojectionReporter:
    def __init__(self):
        self.rows = []

    def add(self, step, cam_name, x_obs, x_proj):
        x_obs = np.asarray(x_obs, dtype=np.float64)
        x_proj = np.asarray(x_proj, dtype=np.float64)
        err = np.linalg.norm(x_obs - x_proj, axis=1)

        self.rows.append(
            {
                "step": step,
                "camera": cam_name,
                "n_points": int(err.shape[0]),
                "mean": float(np.mean(err)),
                "median": float(np.median(err)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "max": float(np.max(err)),
            }
        )

    def print_table(self):
        if not self.rows:
            print("No reprojection rows to report.")
            return

        headers = ["Step", "Camera", "N", "Mean(px)", "Median(px)", "RMSE(px)", "Max(px)"]
        data_rows = []
        for r in self.rows:
            data_rows.append(
                [
                    r["step"],
                    r["camera"],
                    str(r["n_points"]),
                    f"{r['mean']:.4f}",
                    f"{r['median']:.4f}",
                    f"{r['rmse']:.4f}",
                    f"{r['max']:.4f}",
                ]
            )

        widths = [len(h) for h in headers]
        for row in data_rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        sep = "-+-".join("-" * w for w in widths)

        print("\nReprojection Error Summary")
        print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        print(sep)
        for row in data_rows:
            print(" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)))

    def save_csv(self, path):
        if not self.rows:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("step,camera,n_points,mean,median,rmse,max\n")
            for r in self.rows:
                f.write(
                    f"{r['step']},{r['camera']},{r['n_points']},{r['mean']:.8f},{r['median']:.8f},{r['rmse']:.8f},{r['max']:.8f}\n"
                )


def save_reprojection_overlay(image, x_obs, x_proj, out_path, title_text=None):
    canvas = image.copy()
    x_obs = np.asarray(x_obs, dtype=np.float64)
    x_proj = np.asarray(x_proj, dtype=np.float64)

    for p_obs, p_proj in zip(x_obs, x_proj):
        uo, vo = int(round(p_obs[0])), int(round(p_obs[1]))
        up, vp = int(round(p_proj[0])), int(round(p_proj[1]))

        cv2.circle(canvas, (uo, vo), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (up, vp), 3, (0, 0, 255), -1)
        cv2.line(canvas, (uo, vo), (up, vp), (255, 0, 0), 1)

    if title_text is not None:
        cv2.putText(
            canvas,
            title_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)


def evaluate_and_visualize_step(
    reporter,
    step_name,
    cam_name,
    image,
    X_world,
    x_obs,
    K,
    R,
    C,
    out_path,
):
    x_proj = project_points(X_world, R, C, K)
    reporter.add(step_name, cam_name, x_obs, x_proj)
    save_reprojection_overlay(image, x_obs, x_proj, out_path, title_text=f"{step_name} - {cam_name}")
    return x_proj
