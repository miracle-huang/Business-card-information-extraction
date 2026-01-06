# 尝试只使用 OpenCV 来检测并提取场景中的名片
# 原理: 用 MSER 找文字/图案区域 -> 按位置聚类

from pathlib import Path
import cv2
import numpy as np


# =========================
# 固定路径配置
# =========================
IMAGE_PATH = r"data/synth_output/synth_20260104_122112_00000.jpg"   # ← 输入图片路径
OUT_DIR = r"data/cards_out"                        # ← 输出目录
EXPAND = 1.15                                 # 框放大系数
DEBUG = True                                  # 是否输出 debug 图


def order_points(pts: np.ndarray) -> np.ndarray:
    """把4个点排序成: tl, tr, br, bl"""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def warp_quad(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = order_points(quad)
    (tl, tr, br, bl) = rect
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    W = int(round(max(wA, wB)))
    H = int(round(max(hA, hB)))
    W = max(W, 50)
    H = max(H, 50)

    dst = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H))
    return warped


def detect_4_cards_by_mser(img_bgr: np.ndarray, expand: float = 1.15):
    """
    用 MSER 找文字/图案区域 -> 按位置聚类成4组 -> 每组 minAreaRect 得到一张卡的四点框
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(8000)

    regions, bboxes = mser.detectRegions(gray)
    centers = []
    sel_idx = []

    for i, (x, y, w, h) in enumerate(bboxes):
        area = w * h
        if area < 200 or area > 20000:
            continue
        centers.append([x + w / 2, y + h / 2])
        sel_idx.append(i)

    if len(centers) < 50:
        raise RuntimeError("MSER 特征太少，无法稳定分成4张卡")

    centers = np.array(centers, np.float32)
    K = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(
        centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    clusters = [[] for _ in range(K)]
    for j, i in enumerate(sel_idx):
        clusters[int(labels[j])].append(regions[i])

    quads = []
    for cl in clusters:
        pts = np.vstack(cl).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), ang = rect
        rect = ((cx, cy), (w * expand, h * expand), ang)
        quad = cv2.boxPoints(rect)
        quads.append(quad.astype(np.float32))

    return quads


def sort_quads_reading_order(quads):
    """按从上到下、从左到右排序"""
    def center(q):
        return float(q[:, 0].mean()), float(q[:, 1].mean())

    items = [(q, *center(q)) for q in quads]
    items.sort(key=lambda t: (t[2], t[1]))  # y → x
    return [t[0] for t in items]


def main():
    img_path = Path(IMAGE_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"读不到图片: {img_path}")

    quads = detect_4_cards_by_mser(img, expand=EXPAND)
    quads = sort_quads_reading_order(quads)

    # 导出每张回正名片
    for i, q in enumerate(quads):
        card = warp_quad(img, q)

        # 统一横向
        if card.shape[0] > card.shape[1]:
            card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(str(out_dir / f"card_{i}.jpg"), card)

    if DEBUG:
        dbg = img.copy()
        for i, q in enumerate(quads):
            qi = q.astype(np.int32)
            cv2.polylines(dbg, [qi], True, (0, 0, 255), 3)
            cx, cy = int(q[:, 0].mean()), int(q[:, 1].mean())
            cv2.putText(
                dbg, str(i), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )
        cv2.imwrite(str(out_dir / "debug_boxes.jpg"), dbg)

    print(f"Done. 输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
