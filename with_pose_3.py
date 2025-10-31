import os
import json
import cv2
import math
import numpy as np
from skimage.color import rgb2lab
from ultralytics import YOLO


BBOX_EXPAND   = 0.15   
POSE_MIN_PIX  = 80     
ANGLE_STAND   = 20.0
ANGLE_SIT     = 60.0
DEBUG         = False 

def get_dominant_color(image_bgr):
   
    if image_bgr is None or image_bgr.size == 0:
        return "unknown"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _   = image_rgb.shape
    scale     = 150.0 / max(h, w)
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    image_lab = rgb2lab(image_rgb).reshape(-1, 3)
    hist, edges = np.histogramdd(image_lab, bins=(8, 8, 8))
    idx = np.unravel_index(np.argmax(hist), hist.shape)
    l_mid = (edges[0][idx[0]] + edges[0][idx[0] + 1]) / 2
    a_mid = (edges[1][idx[1]] + edges[1][idx[1] + 1]) / 2
    b_mid = (edges[2][idx[2]] + edges[2][idx[2] + 1]) / 2
    if l_mid < 30:  return "black"
    if l_mid > 80:  return "white"
    if a_mid > 20 and b_mid < 20:       return "red"
    if a_mid < 0 and b_mid < 0:         return "green"
    if b_mid > 20 and a_mid < 20:       return "yellow"
    if b_mid < -10:                     return "blue"
    return "gray"



def compute_iou(b1, b2):
 
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter  = max(0, xb - xa) * max(0, yb - ya)
    union  = w1 * h1 + w2 * h2 - inter
    return 0 if union == 0 else inter / union

def posture_from_keypoints(kpt, conf_th=0.2):

    need = [5, 6, 11, 12]  # L/R shoulder + L/R hip
    if any(kpt[i, 2] < conf_th for i in need):
        return "unknown"
    sx, sy = kpt[[5, 6], 0].mean(), kpt[[5, 6], 1].mean()
    hx, hy = kpt[[11, 12], 0].mean(), kpt[[11, 12], 1].mean()
    dx, dy = sx - hx, hy - sy
    angle  = abs(math.degrees(math.atan2(dx, dy)))
    if angle <= ANGLE_STAND:
        return "standing"
    elif angle <= ANGLE_SIT:
        return "sitting"
    else:
        return "lying"


def main():
    video_path  = "_test.mp4"
    output_json = "step_1.json"
    skip_frame  = 1
    det_conf    = 0.4
    pose_conf   = 0.25

   
    det_model  = YOLO("yolo/yolo11m.pt").to("cuda")

    pose_model = YOLO("yolov8n-pose.pt").to("cuda")

    print("Torch =", __import__("torch").__version__,
          "  CUDA =", __import__("torch").cuda.is_available())

    allowed = {"person": "person", "car": "car", "truck": "car",
               "bus": "car", "motorcycle": "car", "bicycle": "car"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return

    results_json, frame_idx = {}, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_frame != 0:
            frame_idx += 1
            continue

 
        det_res = det_model.track(
            frame, conf=det_conf, device="cuda",
            persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        pose_res = pose_model.predict(
            frame, conf=pose_conf, device="cuda", verbose=False)[0]

    
        pose_list = []
        for b, k in zip(pose_res.boxes, pose_res.keypoints):
            pb = b.xyxy[0].cpu().numpy()       # xyxy
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            pose_list.append({
                "bbox_xywh": [pb[0], pb[1], pw, ph],
                "kpt": k.xy[0].cpu().numpy(),       # (17,2)
                "kpt_conf": k.conf[0].cpu().numpy() # (17,)
            })

        frame_out = []
        for box in det_res.boxes:
            cls_id = int(box.cls[0])
            name   = det_model.names[cls_id]
            if name not in allowed:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h           = x2 - x1, y2 - y1
            conf           = float(box.conf[0])
            tid            = int(box.id[0]) if box.id is not None else -1
            main_label     = allowed[name]

     
            ex = int(BBOX_EXPAND * w)
            ey = int(BBOX_EXPAND * h)
            x1e = max(0, x1 - ex); y1e = max(0, y1 - ey)
            x2e = min(frame.shape[1], x2 + ex)
            y2e = min(frame.shape[0], y2 + ey)
            roi = frame[y1e:y2e, x1e:x2e]
            color = get_dominant_color(roi)

    
            posture = "standing"
            if main_label == "person":
                best_p, best_iou = None, 0
                det_xywh = [x1, y1, w, h]
                for p in pose_list:
                    iou = compute_iou(det_xywh, p["bbox_xywh"])
                    if iou > best_iou:
                        best_iou, best_p = iou, p
                if best_iou > 0.5:
                    kpt_xy   = best_p["kpt"]
                    kpt_conf = best_p["kpt_conf"].reshape(-1, 1)
                    kpt_full = np.concatenate([kpt_xy, kpt_conf], axis=1)
                    posture  = posture_from_keypoints(kpt_full)
                elif DEBUG:
                    print(f"[DEBUG] frame {frame_idx}: no matching pose IoU>0.5")

            frame_out.append({
                "track_id":  tid,
                "label":     main_label,
                "confidence":round(conf, 3),
                "bbox":     [int(x1), int(y1), int(w), int(h)],
                "color":    color,
                "posture":  posture
            })

        results_json[f"frame_{frame_idx}"] = frame_out
        frame_idx += 1

    cap.release()


    track_counts = {}
    for objs in results_json.values():
        for obj in objs:
            tid = obj["track_id"]
            track_counts[tid] = track_counts.get(tid, 0) + 1

    filtered_results = {}
    for frame, objs in results_json.items():
        keep = [obj for obj in objs if track_counts.get(obj["track_id"], 0) >= 20]
        if keep:
            filtered_results[frame] = keep
    results_json = filtered_results


    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    main()

