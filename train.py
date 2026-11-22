import argparse, glob, os
import numpy as np, cv2
import skimage.data as skd
from sklearn.metrics import roc_auc_score
from gabor_face import WIN, face_filters, patch_features, PCAPNN, GaborFaceDetector

RNG = np.random.default_rng(0)
KERNELS = face_filters()
NORM = "logl2"


def to20(img):
    return cv2.resize(np.ascontiguousarray(img, np.float32), (WIN, WIN), interpolation=cv2.INTER_AREA)


def augment(face, n):
    """Flip / shift / scale / rotate / brightness variations of one face crop."""
    out, h = [], face.shape[0]
    for _ in range(n):
        M = cv2.getRotationMatrix2D((h / 2, h / 2), RNG.uniform(-10, 10), RNG.uniform(0.92, 1.08))
        M[:, 2] += RNG.uniform(-1.2, 1.2, 2)
        im = cv2.warpAffine(face.astype(np.float32), M, (h, h), borderMode=cv2.BORDER_REFLECT)
        if RNG.random() < 0.5:
            im = im[:, ::-1]
        out.append(to20(np.clip(im * RNG.uniform(0.8, 1.2) + RNG.uniform(-0.08, 0.08), 0, 1)))
    return out


def part_negatives(face, n):
    """Badly localised windows built from a known face: zoomed-in parts, half-shifted and zoomed-out views.
    They teach the classifier that only a TIGHT face window is a face."""
    h, out = face.shape[0], []
    for _ in range(n):
        mode = RNG.integers(3)
        if mode == 0:                                              # zoomed-in part (eye, nose, mouth ...)
            s = int(RNG.uniform(0.3, 0.65) * h)
            y, x = RNG.integers(0, h - s + 1, 2)
            crop = face[y:y + s, x:x + s]
        else:
            pad = int(h * (RNG.uniform(0.5, 1.2) if mode == 1 else RNG.uniform(0.1, 0.2)))
            big = cv2.copyMakeBorder(face, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            if mode == 1:                                          # zoomed-out: face shrunk inside context
                crop = big
            else:                                                  # shifted by 40-60 % of the face size
                d = int(h * RNG.uniform(0.4, 0.6)) * RNG.choice([-1, 1], 2)
                y0, x0 = max(0, min(big.shape[0] - h, pad + d[0])), max(0, min(big.shape[1] - h, pad + d[1]))
                crop = big[y0:y0 + h, x0:x0 + h]
        out.append(to20(crop))
    return out


def random_crops(gray, n):
    h, w = gray.shape
    out = []
    for _ in range(n):
        s = int(RNG.uniform(WIN, max(WIN + 1, min(h, w) * 0.6)))
        y, x = RNG.integers(0, h - s + 1), RNG.integers(0, w - s + 1)
        out.append(to20(gray[y:y + s, x:x + s]))
    return out


def synthetic_images(n=8):
    """Clouds, stripes and blobs - cheap extra non-face texture."""
    imgs = []
    for i in range(n):
        base = RNG.random((256, 256)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), RNG.uniform(1.5, 12))
        if i % 3 == 1:                                    # add oriented stripes
            yy, xx = np.mgrid[0:256, 0:256]
            t = RNG.uniform(0, np.pi); f = RNG.uniform(0.02, 0.2)
            base = base + 0.5 * np.sin(2 * np.pi * f * (xx * np.cos(t) + yy * np.sin(t)))
        if i % 3 == 2:                                    # add blobs / edges
            for _ in range(12):
                cv2.circle(base, tuple(int(v) for v in RNG.integers(0, 256, 2)), int(RNG.integers(5, 50)),
                           float(RNG.random()), -1 if RNG.random() < .5 else 2)
        imgs.append((base - base.min()) / (base.max() - base.min() + 1e-9))
    return imgs


def negative_images():
    """Bundled scikit-image pictures that contain no human face (astronaut and camera are excluded)."""
    names = ["coins", "text", "page", "clock", "rocket", "coffee", "brick", "grass", "gravel", "moon",
             "horse", "chelsea", "retina", "immunohistochemistry", "cell", "hubble_deep_field",
             "colorwheel", "checkerboard", "binary_blobs", "shepp_logan_phantom"]
    imgs = []
    for n in names:
        try:
            im = np.asarray(getattr(skd, n)(), np.float32)
            if im.ndim == 3:
                im = im[..., :3].mean(2)
            imgs.append((im - im.min()) / (im.max() - im.min() + 1e-9))
        except Exception:
            pass
    return imgs + synthetic_images()


def load_folder(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            out.append(im.astype(np.float32) / 255.0)
    return out


def feats(patches):
    return np.array([patch_features(p, KERNELS, NORM) for p in patches])


def evaluate(m, Vf, Vn):
    pf, pn = m.predict_proba(Vf), m.predict_proba(Vn)
    auc = roc_auc_score(np.r_[np.ones(len(pf)), np.zeros(len(pn))], np.r_[pf, pn])
    return auc, (pf > np.quantile(pn, 0.99)).mean(), (pf > 0.5).mean(), (pn <= 0.5).mean()


def mine(model, images, rounds_cap=3000, thr=0.5):
    """Hard negatives = windows of face-free images that the current model calls a face."""
    det, hard = GaborFaceDetector(model, KERNELS), []
    for im in images:
        for (x, y, w, h, p) in det.candidates(im, threshold=thr, max_dim=320, stride=3):
            crop = im[int(y):int(y + h), int(x):int(x + w)]
            if crop.size:
                hard.append(to20(crop))
    print(f"   hard negatives found: {len(hard)}")
    pick = RNG.permutation(len(hard))[:rounds_cap]
    return [hard[i] for i in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces"), ap.add_argument("--nonfaces")
    ap.add_argument("--out", default="model.npz")
    ap.add_argument("--aug", type=int, default=25, help="augmented copies per face")
    ap.add_argument("--components", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=3, help="hard-negative mining rounds")
    a = ap.parse_args()

    data = skd.lfw_subset()
    faces, nonf = list(data[:100]), list(data[100:])
    if a.faces:
        faces += [cv2.resize(f, (25, 25), interpolation=cv2.INTER_AREA) for f in load_folder(a.faces)]
    bg = negative_images() + (load_folder(a.nonfaces) if a.nonfaces else [])

    # ---- hold-outs (split BEFORE augmentation so there is no leakage) ------------------------------
    pf, pn = RNG.permutation(len(faces)), RNG.permutation(len(bg))
    val_faces = [faces[i] for i in pf[: len(faces) // 4]]
    tr_faces = [faces[i] for i in pf[len(faces) // 4:]]
    val_bg = [bg[i] for i in pn[:4]]
    tr_bg = [bg[i] for i in pn[4:]]
    Vf = feats([to20(f) for f in val_faces] + sum((augment(f, 4) for f in val_faces), []))
    Vn = feats(sum((random_crops(im, 120) for im in val_bg), []) + [to20(n) for n in nonf[:20]])

    def build(face_list, bg_list):
        P = [to20(f) for f in face_list] + sum((augment(f, a.aug) for f in face_list), [])
        N = [to20(n) for n in nonf] + sum((augment(n, 2) for n in nonf), [])
        N += sum((part_negatives(f, 12) for f in face_list), [])
        for im in bg_list:
            N += random_crops(im, 220)
        return feats(P), feats(N)

    print("extracting features ...")
    Xf, Xn = build(tr_faces, tr_bg)
    print(f"train: {len(Xf)} faces / {len(Xn)} non-faces | validation: {len(Vf)} faces / {len(Vn)} non-faces")
    X, y = np.vstack([Xf, Xn]), np.r_[np.ones(len(Xf)), np.zeros(len(Xn))]

    best = None
    for sigma in (0.08, 0.1, 0.13, 0.17, 0.22, 0.3):
        m = PCAPNN(a.components, sigma, NORM).fit(X, y)
        auc, tpr1, rec, rej = evaluate(m, Vf, Vn)
        print(f"  sigma={sigma:.2f}  AUC={auc:.3f}  recall@1%FPR={tpr1:.2f}  (p>.5: recall={rec:.2f}, reject={rej:.2f})")
        if best is None or auc + tpr1 > best[0]:
            best = (auc + tpr1, sigma)
    sigma = best[1]
    print("selected sigma:", sigma)

    model = PCAPNN(a.components, sigma, NORM).fit(X, y)
    for r in range(a.rounds):
        print(f"hard-negative mining round {r + 1}/{a.rounds}")
        hard = mine(model, tr_bg)
        if not hard:
            break
        X = np.vstack([X, feats(hard)]); y = np.r_[y, np.zeros(len(hard))]
        model = PCAPNN(a.components, sigma, NORM).fit(X, y)
        auc, tpr1, rec, rej = evaluate(model, Vf, Vn)
        print(f"   validation: AUC={auc:.3f}  recall@1%FPR={tpr1:.2f}  recall={rec:.2f}  reject={rej:.2f}")

    # ---- final model: everything (incl. held-out faces/backgrounds) ----------------------------------
    print("fitting final model on all data ...")
    Xf2, Xn2 = build(faces, bg)
    X = np.vstack([Xf2, Xn2, X[len(Xf) + len(Xn):]])
    y = np.r_[np.ones(len(Xf2)), np.zeros(len(X) - len(Xf2))]
    PCAPNN(a.components, sigma, NORM).fit(X, y).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
