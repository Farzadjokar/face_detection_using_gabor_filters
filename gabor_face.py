import numpy as np
import cv2
from scipy.special import logsumexp

WIN = 20            # scanning window size (pixels)
HALF = 5            # Gabor kernel half-size -> 11x11 kernels; valid responses of a 20x20 window = 10x10
ORIENTATIONS = (0, 45, 90, 135)

# ----------------------------------------------------------------------------
# Gabor kernels
# ----------------------------------------------------------------------------
def gabor_kernel(freq, theta, sigma_x, sigma_y, half=None):
    """Complex 2-D Gabor kernel  g(x,y) = exp(-.5(x'^2/sx^2 + y'^2/sy^2)) * exp(2*pi*j*freq*x').

    freq   : spatial frequency in cycles / pixel
    theta  : orientation in radians
    The DC component of the real part is removed so the filter ignores brightness offsets.
    """
    if half is None:
        half = int(np.ceil(3 * max(sigma_x, sigma_y)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-0.5 * (xr ** 2 / sigma_x ** 2 + yr ** 2 / sigma_y ** 2))
    k = env * np.exp(2j * np.pi * freq * xr)
    k = k - env * (k.real.sum() / env.sum())          # zero-mean (removes DC from real part)
    return k / np.abs(k).sum()


def face_filters(freq=1 / 6.0, sigma=2.0):
    """The four 11x11 Gabor filters (0/45/90/135 degrees) used for face detection."""
    return [gabor_kernel(freq, np.deg2rad(t), sigma, sigma, half=HALF) for t in ORIENTATIONS]


def manjunath_params(S=4, K=6, Ul=0.05, Uh=0.4):
    """Filter-width parameters that tile the frequency plane (slide 13, Manjunath & Ma 1996)."""
    a = (Uh / Ul) ** (1.0 / (S - 1))
    su = (a - 1) * Uh / ((a + 1) * np.sqrt(2 * np.log(2)))
    inner = 2 * np.log(2) - ((2 * np.log(2)) ** 2 * su ** 2) / Uh ** 2
    sv = np.tan(np.pi / (2 * K)) * (Uh - 2 * np.log(2) * su ** 2 / Uh) * inner ** -0.5
    return a, su, sv


def manjunath_bank(S=4, K=6, Ul=0.05, Uh=0.4):
    """Gabor wavelet family: S scales x K orientations (theta_n = n*pi/K). Returns [scale][orientation]."""
    a, su, sv = manjunath_params(S, K, Ul, Uh)
    sx0, sy0 = 1 / (2 * np.pi * su), 1 / (2 * np.pi * sv)
    bank = []
    for m in range(S):
        row = []
        for n in range(K):
            row.append(gabor_kernel(Uh * a ** (-m), n * np.pi / K, sx0 * a ** m, sy0 * a ** m))
        bank.append(row)
    return bank


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def _corner_mask():
    """Keep-mask for a 10x10 map with 8 corner pixels removed -> 92 values per map."""
    m = np.ones((10, 10), bool)
    for r in (0, 9):
        m[r, [0, 1, 8, 9]] = False
    return m


CORNER_MASK = _corner_mask()
N_FEATURES = 4 * int(CORNER_MASK.sum())          # 368


def magnitude_maps(gray, kernels):
    """|image * gabor| for every kernel (gray: float32 in [0,1])."""
    maps = []
    for k in kernels:
        re = cv2.filter2D(gray, cv2.CV_32F, np.ascontiguousarray(k.real, np.float32),
                          borderType=cv2.BORDER_REFLECT)
        im = cv2.filter2D(gray, cv2.CV_32F, np.ascontiguousarray(k.imag, np.float32),
                          borderType=cv2.BORDER_REFLECT)
        maps.append(np.sqrt(re * re + im * im))
    return maps


def scan_features(gray, kernels, stride=2, min_std=0.04, norm="logl2"):
    """Features for every WIN x WIN window of `gray` on a regular grid.

    Returns (features [n, 368], ys [n], xs [n]) - flat windows (no contrast) are skipped.
    """
    H, W = gray.shape
    if H < WIN or W < WIN:
        return np.zeros((0, N_FEATURES), np.float32), np.zeros(0, int), np.zeros(0, int)
    maps = magnitude_maps(gray, kernels)
    ys = np.arange(0, H - WIN + 1, stride)
    xs = np.arange(0, W - WIN + 1, stride)
    rows = ys[:, None] + HALF + np.arange(10)[None, :]         # (ny,10) kernel centres fully inside the window
    cols = xs[:, None] + HALF + np.arange(10)[None, :]         # (nx,10)
    feats = []
    for M in maps:
        g = M[rows[:, None, :, None], cols[None, :, None, :]]  # (ny,nx,10,10)
        feats.append(g[:, :, CORNER_MASK])                     # (ny,nx,92)
    if norm == "map":                                          # normalise every orientation map separately
        feats = [f / (np.linalg.norm(f, axis=2, keepdims=True) + 1e-8) / 2.0 for f in feats]
    F = np.concatenate(feats, axis=2)                          # (ny,nx,368)
    if norm == "l2":
        F = F / (np.linalg.norm(F, axis=2, keepdims=True) + 1e-8)  # contrast invariance
    elif norm == "logl2":                                      # compress dynamic range, then normalise
        F = np.log1p(F / (F.mean(axis=2, keepdims=True) + 1e-8))
        F = F - F.mean(axis=2, keepdims=True)
        F = F / (np.linalg.norm(F, axis=2, keepdims=True) + 1e-8)

    # reject flat windows using local std of the grey image
    mean = cv2.boxFilter(gray, -1, (WIN, WIN), anchor=(0, 0), normalize=True)
    sq = cv2.boxFilter(gray * gray, -1, (WIN, WIN), anchor=(0, 0), normalize=True)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))[np.ix_(ys, xs)]
    ok = std > min_std
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return F[ok].astype(np.float32), yy[ok], xx[ok]


def patch_features(patch, kernels, norm="logl2"):
    """Features of a single 20x20 patch (float32, [0,1]) - exactly the same code path as scanning."""
    f, _, _ = scan_features(np.ascontiguousarray(patch, np.float32), kernels, stride=1, min_std=-1, norm=norm)
    return f[0]


# ----------------------------------------------------------------------------
# PCA + PNN classifier
# ----------------------------------------------------------------------------
class PCAPNN:
    """PCA projection followed by a Probabilistic Neural Network (Parzen window, Gaussian kernels).

    Each class density is  p(z|c) = sum_i w_i * exp(-|z - z_i|^2 / (2 sigma^2)) / sum_i w_i.
    The training samples can be compressed into weighted k-means prototypes to make scanning fast.
    """
    def __init__(self, n_components=60, sigma=0.15, norm="logl2"):
        self.n_components, self.sigma, self.norm = n_components, sigma, norm

    def fit(self, X, y, prototypes=(500, 1200), seed=0):
        X = np.asarray(X, np.float64)
        self.mean_ = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        Z = (X - self.mean_) @ self.components_.T
        self.Zf_, self.wf_ = self._compress(Z[y == 1], prototypes[0], seed)
        self.Zn_, self.wn_ = self._compress(Z[y == 0], prototypes[1], seed)
        return self

    @staticmethod
    def _compress(Z, k, seed):
        if k is None or len(Z) <= k:
            return Z, np.ones(len(Z))
        from sklearn.cluster import MiniBatchKMeans          # training-time only dependency
        km = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=2048).fit(Z)
        w = np.bincount(km.labels_, minlength=k).astype(float)
        keep = w > 0
        return km.cluster_centers_[keep], w[keep]

    def _logdens(self, Z, ref, w):
        d2 = (Z ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2 * Z @ ref.T
        return logsumexp(-np.maximum(d2, 0) / (2 * self.sigma ** 2) + np.log(w)[None, :], axis=1) - np.log(w.sum())

    TEMP = 8.0     # confidence = sigmoid(log-likelihood-ratio / TEMP): keeps scores from saturating at 1.0

    def predict_proba(self, X, chunk=2000):
        """Face confidence in [0,1]: sigmoid of the PNN log-likelihood ratio (equal class priors)."""
        X = np.asarray(X, np.float64)
        out = np.empty(len(X))
        for i in range(0, len(X), chunk):
            Z = (X[i:i + chunk] - self.mean_) @ self.components_.T
            lf = self._logdens(Z, self.Zf_, self.wf_)
            ln = self._logdens(Z, self.Zn_, self.wn_)
            out[i:i + chunk] = 1 / (1 + np.exp(-np.clip(lf - ln, -200, 200) / self.TEMP))
        return out

    def save(self, path):
        np.savez_compressed(path, mean=self.mean_, comps=self.components_, Zf=self.Zf_, wf=self.wf_,
                            Zn=self.Zn_, wn=self.wn_, sigma=self.sigma, norm=self.norm)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(len(d["comps"]), float(d["sigma"]), str(d["norm"]))
        m.mean_, m.components_ = d["mean"], d["comps"]
        m.Zf_, m.wf_, m.Zn_, m.wn_ = d["Zf"], d["wf"], d["Zn"], d["wn"]
        return m


# ----------------------------------------------------------------------------
# Multi-scale detector
# ----------------------------------------------------------------------------
def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def contained(small, big, frac=0.7):
    """True if at least `frac` of the small box lies inside the big box."""
    ix = max(0, min(small[0] + small[2], big[0] + big[2]) - max(small[0], big[0]))
    iy = max(0, min(small[1] + small[3], big[1] + big[3]) - max(small[1], big[1]))
    return ix * iy >= frac * small[2] * small[3]


def nms(dets, overlap=0.3):
    """Non-maximum suppression: among overlapping candidates keep the one with the highest face likelihood.
    A box that sits inside a larger kept box (an eye, a cheek ...) is also dropped; a larger box that
    contains a kept box replaces it when its likelihood is nearly as high."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if any(iou(d, k) >= overlap for k in keep):
            continue
        inside = [k for k in keep if d[2] > k[2] and contained(k, d)]
        if inside and d[4] >= 0.9 * max(k[4] for k in inside):
            for k in inside:
                keep.remove(k)
            keep.append(d)
        elif any(k[2] > d[2] and contained(d, k) for k in keep):
            continue
        else:
            keep.append(d)
    return keep


class GaborFaceDetector:
    def __init__(self, model, kernels=None):
        self.model = model
        self.kernels = kernels or face_filters()

    def candidates(self, gray, threshold=0.5, scale_step=1.2, stride=2, min_face=40, max_dim=640,
                   progress=None):
        """Slow part: scan the image pyramid. Returns every window (x, y, w, h, score) with
        score >= threshold, in original-image pixels. Cache this and call `filter` for fast re-thresholding."""
        H0, W0 = gray.shape
        pre = min(1.0, max_dim / float(max(H0, W0)))
        g = cv2.resize(gray, None, fx=pre, fy=pre, interpolation=cv2.INTER_AREA) if pre < 1 else gray
        g = g.astype(np.float32)
        dets, scales, s = [], [], 1.0
        while min(g.shape) * s >= WIN:          # image pyramid: shrink by `s` each level
            scales.append(s)
            s /= scale_step
        # a 20 px window on a level with factor s covers 20/(s*pre) px of the original image
        scales = [sc for sc in scales if WIN / (sc * pre) >= min_face * 0.999] or scales[:1]
        for i, sc in enumerate(scales):
            level = g if sc == 1.0 else cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            F, ys, xs = scan_features(level, self.kernels, stride=stride, norm=self.model.norm)
            if len(F):
                p = self.model.predict_proba(F)
                k = 1.0 / (sc * pre)
                for pi, y, x in zip(p, ys, xs):
                    if pi >= threshold:
                        dets.append((x * k, y * k, WIN * k, WIN * k, float(pi)))
            if progress:
                progress((i + 1) / len(scales))
        return dets

    @staticmethod
    def filter(cands, threshold=0.8, overlap=0.3, min_neighbors=1):
        """Fast part: threshold -> NMS (highest likelihood wins) -> require `min_neighbors`
        overlapping hits to confirm a face. Returns the final boxes."""
        cc = [d for d in cands if d[4] >= threshold]
        return [d for d in nms(cc, overlap) if sum(1 for c in cc if iou(c, d) > 0.3) >= min_neighbors]

    def detect(self, gray, threshold=0.8, overlap=0.3, min_neighbors=1, **kw):
        cands = self.candidates(gray, min(threshold, 0.5), **kw)
        return self.filter(cands, threshold, overlap, min_neighbors), cands
