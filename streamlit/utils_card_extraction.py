import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional, List, Dict, Any, Union

# =========================
# Geometry helpers
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points to tl, tr, br, bl."""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def inset_quad(quad: np.ndarray, ratio: float) -> np.ndarray:
    """Move quad points slightly toward center to suppress background."""
    if ratio <= 0:
        return quad.astype(np.float32)
    q = quad.astype(np.float32)
    c = q.mean(axis=0, keepdims=True)
    return c + (q - c) * (1.0 - float(ratio))


def line_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    """Intersection of 2 param lines: p1+t*d1 and p2+u*d2. Return point or None if parallel."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float32)
    b = (p2 - p1).astype(np.float32)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-6:
        return None
    t_u = np.linalg.solve(A, b)
    t = float(t_u[0])
    return (p1 + t * d1).astype(np.float32)


def point_line_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from pt to infinite line through a-b."""
    v = b - a
    w = pt - a
    denom = float(np.hypot(v[0], v[1]) + 1e-9)
    return float(abs(np.cross(v, w)) / denom)


# =========================
# Mask / contour helpers
# =========================
def polygon_to_mask(poly_xy: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """poly_xy: (N,2) float -> binary mask uint8 {0,255}."""
    H, W = hw
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = poly_xy.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def clean_mask(mask: np.ndarray, close_ksize: int = 7) -> np.ndarray:
    """Remove holes/noise; keep largest connected component."""
    m = mask.copy()
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    cnt = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(m)
    cv2.fillPoly(out, [cnt], 255)
    return out


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# =========================
# Quad estimation
# =========================
def approx_quad_from_contour(cnt: np.ndarray, eps_ratio: float) -> Optional[np.ndarray]:
    peri = cv2.arcLength(cnt, True)
    eps = float(eps_ratio) * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    return None


def box_from_min_area_rect(cnt: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return order_points(box)


def fit_edge_lines_and_intersections(cnt: np.ndarray, init_box: np.ndarray, dist_thresh: float) -> Optional[np.ndarray]:
    """
    Use init_box edges as guidance:
    - For each of 4 edges, collect contour points close to that edge
    - Fit line with cv2.fitLine
    - Intersect adjacent lines -> refined corners
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    box = init_box.astype(np.float32)  # tl,tr,br,bl
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]

    fitted = []
    for (a, b) in edges:
        dists = np.array([point_line_distance(p, a, b) for p in pts], dtype=np.float32)
        sel = pts[dists < dist_thresh]
        if sel.shape[0] < 40:
            k = min(200, pts.shape[0])
            idx = np.argsort(dists)[:k]
            sel = pts[idx]
        if sel.shape[0] < 10:
            return None

        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        d = np.array([vx, vy], dtype=np.float32)
        p0 = np.array([x0, y0], dtype=np.float32)
        n = float(np.hypot(d[0], d[1]) + 1e-9)
        d = d / n
        fitted.append((p0, d))

    corners = []
    for i in range(4):
        p1, d1 = fitted[i]
        p2, d2 = fitted[(i + 1) % 4]
        inter = line_intersection(p1, d1, p2, d2)
        if inter is None:
            return None
        corners.append(inter)

    quad = np.stack(corners, axis=0).astype(np.float32)
    return order_points(quad)


def estimate_quad_from_mask(mask: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Returns (quad, method). quad is ordered tl,tr,br,bl in image coords.
    """
    cnt = largest_contour(mask)
    if cnt is None:
        return None, "no_contour"

    q = approx_quad_from_contour(cnt, eps_ratio=0.012)
    if q is not None:
        return order_points(q), "approx4_eps0.012"

    box = box_from_min_area_rect(cnt)
    w = float(np.linalg.norm(box[1] - box[0]))
    h = float(np.linalg.norm(box[3] - box[0]))
    dist_thresh = max(3.0, 0.01 * min(w, h))
    q2 = fit_edge_lines_and_intersections(cnt, box, dist_thresh=dist_thresh)
    if q2 is not None:
        return q2, f"fitLine_edges_dist{dist_thresh:.1f}"

    return box, "minAreaRect"


# =========================
# Warp + crop
# =========================
def warp_by_quad(img_bgr: np.ndarray, mask_u8: np.ndarray, quad: np.ndarray, pad: int, border_val: int = 255):
    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))
    out_w = int(round(max(wA, wB)))
    out_h = int(round(max(hA, hB)))

    out_w = max(out_w, 64)
    out_h = max(out_h, 64)

    dst = np.array(
        [[pad, pad], [out_w - 1 + pad, pad], [out_w - 1 + pad, out_h - 1 + pad], [pad, out_h - 1 + pad]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)

    warped_img = cv2.warpPerspective(
        img_bgr,
        M,
        (out_w + 2 * pad, out_h + 2 * pad),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_val, border_val, border_val),
    )
    warped_mask = cv2.warpPerspective(
        mask_u8,
        M,
        (out_w + 2 * pad, out_h + 2 * pad),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_img, warped_mask


def tight_crop_by_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, erode_px: int, margin_px: int) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
    m = mask_u8.copy()
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, k, iterations=1)

    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, (0, 0, img_bgr.shape[1], img_bgr.shape[0]), True

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(img_bgr.shape[1] - 1, x2 + margin_px)
    y2 = min(img_bgr.shape[0] - 1, y2 + margin_px)

    cropped = img_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    return cropped, (x1, y1, x2, y2), False


# =========================
# Upright by classification
# =========================
def rot90_ccw(img: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))


def get_rot0_class_id(cls_model: YOLO) -> int:
    names = getattr(cls_model, "names", None)
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == "rot0":
                return int(k)
    if isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            if str(v).lower() == "rot0":
                return int(i)
    return 0


def choose_upright_by_cls(img_bgr: np.ndarray, cls_model: YOLO, imgsz: int, device) -> Tuple[np.ndarray, int, float]:
    rot0_id = get_rot0_class_id(cls_model)

    best_k = 0
    best_p = -1.0
    best_img = img_bgr

    for k in range(4):
        cand = rot90_ccw(img_bgr, k)
        r = cls_model.predict(cand, imgsz=imgsz, device=device, verbose=False)[0]
        p = float(r.probs.data[rot0_id].item())
        if p > best_p:
            best_p = p
            best_k = k
            best_img = cand

    return best_img, best_k, best_p


# =========================
# Main exports
# =========================
def load_seg_model(model_path: str):
    return YOLO(model_path)


def load_cls_model(model_path: str):
    return YOLO(model_path)


def extract_card_seg(
    image_np: np.ndarray,
    seg_model: YOLO,
    cls_model: YOLO,
    conf_seg: float = 0.25,
    iou_seg: float = 0.5,
    min_area_px: int = 20000,
    device: str = "0",
    imgsz_seg: int = 960,
    imgsz_cls: int = 384
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts, each containing 'crop_img' (upright) and 'debug_info'.
    """
    H, W = image_np.shape[:2]
    # Prediction
    results = seg_model.predict(
        source=image_np,
        imgsz=imgsz_seg,
        conf=conf_seg,
        iou=iou_seg,
        retina_masks=True,
        verbose=False,
        device=device
    )[0]

    extracted_cards = []

    if results.masks is None or results.boxes is None:
        return []

    polys = results.masks.xy
    boxes = results.boxes.xyxy.detach().cpu().numpy()
    
    # Sort by area
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)

    for j in order:
        if areas[j] < min_area_px:
            continue
            
        poly = np.array(polys[j], dtype=np.float32)
        if poly.shape[0] < 4:
            continue
            
        # Get mask
        mask = polygon_to_mask(poly, (H, W))
        mask = clean_mask(mask, close_ksize=7)
        
        # Estimate quad
        quad, method = estimate_quad_from_mask(mask)
        if quad is None:
            continue
            
        # Refine quad (inset)
        quad = inset_quad(quad, 0.004)
        
        # Warp
        warp1, wmask1 = warp_by_quad(image_np, mask, quad, pad=6, border_val=255)
        
        # Second refine (optional but enabled in v5)
        wmask_clean = clean_mask(wmask1, close_ksize=7)
        quad2, _ = estimate_quad_from_mask(wmask_clean)
        if quad2 is not None:
            quad2 = inset_quad(quad2, 0.002) # half of 0.004
            warp2, wmask2 = warp_by_quad(warp1, wmask_clean, quad2, pad=0, border_val=255)
        else:
            warp2, wmask2 = warp1, wmask1
            
        # Tight crop
        crop_img, _, _ = tight_crop_by_mask(warp2, wmask2, erode_px=3, margin_px=0)
        
        # Upright
        upright_img, best_k, best_p = choose_upright_by_cls(
            crop_img, cls_model, imgsz=imgsz_cls, device=device
        )
        
        extracted_cards.append({
            "crop_img": upright_img,
            "mask": mask, # Original mask on full image
            "quad": quad,
            "rotation_k": best_k,
            "rotation_conf": best_p
        })
        
    return extracted_cards
