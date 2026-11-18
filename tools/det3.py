"""Detector v3 prototype (to be ported to Kotlin).

1. "Paper map": morphological closing removes text, table lines, stamps -> only paper vs. background.
2. Candidate outlines from: paper-map edges (contours), bright paper blob, and straight lines (Hough)
   combined with the frame borders (for pages that extend beyond the frame).
3. Each side is scored by STEP CONTRAST on the paper map (paper on one side, something else on the
   other). Printed tables/frames have paper on both sides -> rejected.
4. Result: FULL page (all four sides real) or PARTIAL (a side is the frame border -> "move back").
"""
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/claude/tools")
import det as v1  # noqa: E402  (geometry helpers)

STEP_MIN = 14.0
SIDE_GOOD_FRAC = 0.45   # min fraction of samples on any side showing the step (binding side is weak)
MEAN_GOOD = 0.72        # min average over the four sides
BORDER_PX = 4           # a side within this distance of the frame border is a "frame side"


def paper_map(gray):
    h, w = gray.shape
    k = max(5, int(round(min(w, h) / 50)) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.GaussianBlur(paper, (5, 5), 0)


def bil(img, x, y):
    h, w = img.shape
    if x < 0 or y < 0 or x > w - 1.001 or y > h - 1.001:
        return None
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    a = float(img[y0, x0]) * (1 - fx) + float(img[y0, x0 + 1]) * fx
    b = float(img[y0 + 1, x0]) * (1 - fx) + float(img[y0 + 1, x0 + 1]) * fx
    return a * (1 - fy) + b * fy


def on_frame_border(a, b, w, h):
    tol = BORDER_PX
    return ((a[0] <= tol and b[0] <= tol) or (a[1] <= tol and b[1] <= tol) or
            (a[0] >= w - 1 - tol and b[0] >= w - 1 - tol) or (a[1] >= h - 1 - tol and b[1] >= h - 1 - tol))


def side_quality(paper, q, n=48):
    """Per side: ('frame' | fraction of samples with a paper->other step), and mean paper brightness."""
    h, w = paper.shape
    cx = sum(p[0] for p in q) / 4
    cy = sum(p[1] for p in q) / 4
    d = max(3.0, min(w, h) / 120)
    r = max(3.0, min(w, h) * 0.015)
    prec_tol = max(2.0, r / 3)
    precs = []
    meds = []
    res = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        if on_frame_border(a, b, w, h):
            res.append("frame")
            continue
        L = v1.dist(a, b)
        dx, dy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        nx, ny = -dy, dx
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if (cx - mx) * nx + (cy - my) * ny < 0:
            nx, ny = -nx, -ny
        good = vis = precise = 0
        steps = []
        for s in range(n):
            t = 0.04 + 0.92 * s / (n - 1)
            px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            if bil(paper, px, py) is None:
                continue
            vis += 1
            # tolerant search: strongest step within +-r of the expected edge position
            best = 0.0
            best_o = 0.0
            o = -r
            while o <= r:
                vin = bil(paper, px + nx * (o + d), py + ny * (o + d))
                vout = bil(paper, px + nx * (o - d), py + ny * (o - d))
                if vin is not None and vout is not None:
                    st = abs(vin - vout)
                    if st > best:
                        best, best_o = st, o
                o += 1.0
            steps.append(best)
            if best >= STEP_MIN:
                good += 1
                if abs(best_o) <= prec_tol:
                    precise += 1
        if vis < n * 0.5:
            res.append("frame")  # mostly outside the image
        else:
            res.append(good / vis)
            precs.append(precise / vis)
            meds.append(float(np.median(steps)))
    side_quality.last_precision = float(np.mean(precs)) if len(precs) == 4 else 0.0
    side_quality.last_strength = min(meds) if len(meds) == 4 else 0.0
    return res


def frame_sides_on_paper(blob, q, n=40):
    """A side lying on the frame border is only plausible if paper continues there."""
    h, w = blob.shape
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        if not on_frame_border(a, b, w, h):
            continue
        hit = 0
        for s in range(n):
            t = 0.05 + 0.9 * s / (n - 1)
            x = int(round(min(max(a[0] + (b[0] - a[0]) * t, 2), w - 3)))
            y = int(round(min(max(a[1] + (b[1] - a[1]) * t, 2), h - 3)))
            if blob[y, x]:
                hit += 1
        if hit < 0.7 * n:
            return False
    return True


def clamp_quad(q, w, h):
    return [(min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)) for x, y in q]


def contour_quads(binary, min_area, out, tag, eps_list=(0.01, 0.02, 0.03, 0.045, 0.06, 0.08)):
    cs, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hulls = []
    for c in cs:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh < min_area:
            continue
        hl = cv2.convexHull(c).astype(np.float32)
        a = cv2.contourArea(hl)
        if a >= min_area:
            hulls.append((a, hl))
    hulls.sort(key=lambda t: -t[0])
    for _, hl in hulls[:5]:
        peri = cv2.arcLength(hl, True)
        for e in eps_list:
            ap = cv2.approxPolyDP(hl, e * peri, True)
            if len(ap) == 4:
                out.append((v1.order(ap.reshape(4, 2)), tag))
                break
            if len(ap) < 4:
                break


def lines_of(edges, w, h):
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30,
                           minLineLength=int(min(w, h) * 0.18), maxLineGap=int(min(w, h) * 0.06))
    out = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in segs.reshape(-1, 4).tolist():
        th = math.atan2(y2 - y1, x2 - x1)
        nx, ny = -math.sin(th), math.cos(th)
        rho = x1 * nx + y1 * ny
        if rho < 0:
            nx, ny, rho = -nx, -ny, -rho
        out.append(dict(nx=nx, ny=ny, rho=rho, len=math.hypot(x2 - x1, y2 - y1), seg=(x1, y1, x2, y2)))
    merged = []
    for l in sorted(out, key=lambda l: -l["len"]):
        for m in merged:
            if l["nx"] * m["nx"] + l["ny"] * m["ny"] > math.cos(math.radians(4)) and abs(l["rho"] - m["rho"]) < 6:
                m["len"] += l["len"]
                break
        else:
            merged.append(dict(l))
    return merged


def frame_lines(w, h):
    return [dict(nx=0.0, ny=1.0, rho=0.0, len=0, frame=True),            # top  y=0
            dict(nx=0.0, ny=1.0, rho=h - 1.0, len=0, frame=True),        # bottom
            dict(nx=1.0, ny=0.0, rho=0.0, len=0, frame=True),            # left x=0
            dict(nx=1.0, ny=0.0, rho=w - 1.0, len=0, frame=True)]        # right


def intersect(l1, l2):
    det = l1["nx"] * l2["ny"] - l1["ny"] * l2["nx"]
    if abs(det) < 1e-6:
        return None
    return ((l1["rho"] * l2["ny"] - l2["rho"] * l1["ny"]) / det, (l1["nx"] * l2["rho"] - l2["nx"] * l1["rho"]) / det)


def line_quads(edges, w, h, per_family=7):
    L = lines_of(edges, w, h)
    fr = frame_lines(w, h)
    hz = sorted([l for l in L if abs(l["ny"]) > abs(l["nx"])], key=lambda l: -l["len"])[:per_family] + fr[:2]
    vt = sorted([l for l in L if abs(l["nx"]) >= abs(l["ny"])], key=lambda l: -l["len"])[:per_family] + fr[2:]
    out = []
    tol = 0.03
    for i in range(len(hz)):
        for j in range(i + 1, len(hz)):
            for k in range(len(vt)):
                for m in range(k + 1, len(vt)):
                    pts = [intersect(hz[i], vt[k]), intersect(hz[i], vt[m]), intersect(hz[j], vt[k]), intersect(hz[j], vt[m])]
                    if any(p is None for p in pts):
                        continue
                    if any(p[0] < 0 or p[0] > w - 1 or p[1] < 0 or p[1] > h - 1 for p in pts):
                        continue  # complete pages only: every corner inside the frame
                    out.append((v1.order(pts), "line"))
    return out, L


def detect(gray, verbose=False):
    h, w = gray.shape
    fa = float(w * h)
    paper = paper_map(gray)
    otsu, blob = cv2.threshold(paper, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(paper, max(8, otsu * 0.25), max(20, otsu * 0.6))
    edges_d = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    raw = []
    contour_quads(edges_d, fa * 0.12, raw, "edge")
    contour_quads(blob, fa * 0.12, raw, "blob")
    lq, L = line_quads(edges, w, h)
    raw += lq

    seen = []
    cands = []
    for q, tag in raw:
        if any(v1.dist(a, b) < 4 for s in seen for a, b in [(s[0], q[0])] if all(v1.dist(s[i], q[i]) < 4 for i in range(4))):
            continue
        seen.append(q)
        ar = v1.area(q) / fa
        if ar < 0.15 or not v1.convex(q):
            continue
        if any(a < 55 or a > 125 for a in v1.angles(q)):
            continue
        if v1.opp_ratio(q) < 0.55:
            continue
        sq = side_quality(paper, q)
        frame_sides = sum(1 for s in sq if s == "frame")
        if frame_sides:
            continue  # only complete pages are capture candidates
        real = [s for s in sq if s != "frame"]
        if frame_sides > 2 or any(s < SIDE_GOOD_FRAC for s in real) or (real and np.mean(real) < MEAN_GOOD):
            if verbose and ar > 0.3:
                print("    rej %-5s area=%.2f sides=%s" % (tag, ar, ["F" if s == "frame" else round(s, 2) for s in sq]))
            continue
        # paper must be inside: inner region brighter than average
        mask = np.zeros_like(paper)
        cv2.fillConvexPoly(mask, np.array([[round(x), round(y)] for x, y in q], np.int32), 255)
        inside = cv2.mean(paper, mask)[0]
        quality = float(np.mean(real)) if real else 0.0
        precision = side_quality.last_precision
        strength = side_quality.last_strength
        score = 0.50 * min(ar / 0.7, 1.0) + 0.30 * quality + 0.20 * (1 - frame_sides / 2) + 0.002 * (inside - 128) / 1.27 + 0.25 * precision + 0.30 * min(strength / 60.0, 1.0)
        cands.append(dict(q=q, tag=tag, area=ar, sides=sq, frame=frame_sides, score=score, inside=inside, precision=precision))
    cands.sort(key=lambda c: -c["score"])
    if verbose:
        print("  lines=%d raw=%d kept=%d" % (len(L), len(raw), len(cands)))
        for c in cands[:5]:
            print("   %-5s score=%.3f area=%.2f frame=%d sides=%s inside=%.0f q=%s" % (
                c["tag"], c["score"], c["area"], c["frame"], ["F" if s == "frame" else round(s, 2) for s in c["sides"]],
                c["inside"], [(round(x), round(y)) for x, y in c["q"]]))
    return cands, dict(paper=paper, edges=edges, lines=L, too_close=too_close(blob))


def too_close(blob):
    """Paper runs off the frame: the page is cut -> ask the user to move back."""
    h, w = blob.shape
    if (blob > 0).mean() < 0.45:
        return False
    m = 2
    borders = [blob[m, :], blob[h - 1 - m, :], blob[:, m], blob[:, w - 1 - m]]
    return max((b > 0).mean() for b in borders) > 0.40


def draw(gray, cands, path):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if cands:
        c = cands[0]
        col = (0, 220, 0) if c["frame"] == 0 else (0, 165, 255)
        cv2.polylines(vis, [np.array([[round(x), round(y)] for x, y in c["q"]], np.int32)], True, col, 3)
        cv2.putText(vis, "FULL" if c["frame"] == 0 else "PARTIAL-move back", (10, 30), 0, 0.8, col, 2)
    else:
        cv2.putText(vis, "NONE", (10, 30), 0, 0.8, (0, 0, 255), 2)
    cv2.imwrite(path, vis)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        g, _ = v1.load_preview(p)
        print(p)
        c, info = detect(g, verbose=True)
        print("  -> ", ("FULL" if c else "none"), "too_close=", info["too_close"])
        draw(g, c, p.rsplit(".", 1)[0] + "_v3.png")
