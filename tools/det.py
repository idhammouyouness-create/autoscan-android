"""Python mirror of DocumentDetector.kt (v1.0.4) for fast experiments on real photos."""
import math
import sys

import cv2
import numpy as np

MIN_AREA = 0.10
MAX_AREA = 0.985
MIN_SUPPORT = 0.55
SAMPLES = 32
EPS = [0.015, 0.025, 0.035, 0.05, 0.07, 0.09]


def order(pts):
    pts = [tuple(map(float, p)) for p in pts]
    cx = sum(p[0] for p in pts) / 4
    cy = sum(p[1] for p in pts) / 4
    s = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    st = min(range(4), key=lambda i: s[i][0] + s[i][1])
    return [s[(st + i) % 4] for i in range(4)]


def area(q):
    s = 0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        s += a[0] * b[1] - b[0] * a[1]
    return abs(s) / 2


def convex(q):
    sign = 0
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < 1e-12:
            return False
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def angles(q):
    out = []
    for i in range(4):
        p, c, n = q[(i + 3) % 4], q[i], q[(i + 1) % 4]
        v1 = (p[0] - c[0], p[1] - c[1])
        v2 = (n[0] - c[0], n[1] - c[1])
        d = math.hypot(*v1) * math.hypot(*v2)
        out.append(math.degrees(math.acos(max(-1, min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / d)))) if d else 0)
    return out


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def opp_ratio(q):
    t, b = dist(q[0], q[1]), dist(q[3], q[2])
    l, r = dist(q[0], q[3]), dist(q[1], q[2])
    return min(min(t, b) / max(t, b, 1e-9), min(l, r) / max(l, r, 1e-9))


def support_of(sup, q):
    h, w = sup.shape
    hit = tot = 0
    weakest = 1.0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        sh = 0
        for s in range(SAMPLES):
            t = 0.06 + 0.88 * s / (SAMPLES - 1)
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t))
            tot += 1
            if x < 3 or y < 3 or x >= w - 3 or y >= h - 3:
                continue
            if sup[y, x]:
                hit += 1
                sh += 1
        weakest = min(weakest, sh / SAMPLES)
    ov = hit / tot
    return ov * 0.5 if weakest < 0.3 else ov


def quads_from(binary, min_area, out, tag):
    cs, _ = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cs if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] >= min_area]
    hulls = [cv2.convexHull(c).astype(np.float32) for c in big]
    hulls = [(h, cv2.contourArea(h)) for h in hulls]
    hulls = sorted([x for x in hulls if x[1] >= min_area], key=lambda x: -x[1])[:6]
    for h, _ in hulls:
        peri = cv2.arcLength(h, True)
        for e in EPS:
            ap = cv2.approxPolyDP(h, e * peri, True)
            if len(ap) == 4:
                out.append((order(ap.reshape(4, 2)), tag))
                break
            if len(ap) < 4:
                break


def maps(gray):
    bl = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, otsu_bin = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eA = cv2.Canny(bl, max(10, otsu * 0.5), max(30, otsu))
    eq = cv2.createCLAHE(3.0, (8, 8)).apply(bl)
    eB = cv2.Canny(eq, 25, 75)
    grad = cv2.morphologyEx(bl, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, eC = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k3 = np.ones((3, 3), np.uint8)
    dA = cv2.dilate(eA, k3)
    dB = cv2.dilate(eB, k3)
    blob = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    union = dA | dB | eC
    sup = cv2.dilate(union, k3)
    return dict(dA=dA, dB=dB, eC=eC, blob=blob, sup=sup)


def candidates(gray, verbose=False):
    h, w = gray.shape
    fa = w * h
    m = maps(gray)
    raw = []
    for tag in ["dA", "dB", "eC", "blob"]:
        quads_from(m[tag], fa * MIN_AREA, raw, tag)
    res = []
    for q, tag in raw:
        ar = area(q) / fa
        why = None
        if ar < MIN_AREA or ar > MAX_AREA:
            why = "area %.2f" % ar
        elif not convex(q):
            why = "nonconvex"
        elif any(a < 45 or a > 135 for a in angles(q)):
            why = "angles"
        elif opp_ratio(q) < 0.45:
            why = "opp"
        sup = support_of(m["sup"], q)
        if why is None and sup < MIN_SUPPORT:
            why = "support %.2f" % sup
        if verbose:
            print("  %-4s area=%.3f sup=%.2f %s %s" % (tag, ar, sup, why or "OK", [(round(x), round(y)) for x, y in q]))
        if why is None:
            res.append(dict(q=q, score=sup * 0.65 + min(ar / 0.6, 1) * 0.35, sup=sup, area=ar, tag=tag))
    res.sort(key=lambda c: -c["score"])
    return res, m


def load_preview(path, long_side=640):
    im = cv2.imread(path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = long_side / max(h, w)
    return cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), im


if __name__ == "__main__":
    for p in sys.argv[1:]:
        g, _ = load_preview(p)
        print(p, g.shape)
        c, _ = candidates(g, verbose=True)
        print(" BEST:", c[0] if c else None)
