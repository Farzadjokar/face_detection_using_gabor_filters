import os
import numpy as np, cv2
import skimage.data as skd
from PIL import Image
import viz
from gabor_face import PCAPNN, GaborFaceDetector, face_filters

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs")
os.makedirs(OUT, exist_ok=True)
det = GaborFaceDetector(PCAPNN.load(os.path.join(HERE, "model.npz")), face_filters())


def run(rgb, name, thr):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    boxes, _ = det.detect(gray, threshold=thr)
    viz.draw_detections(Image.fromarray(rgb), boxes).save(os.path.join(OUT, name))
    print(name, len(boxes), "face(s)")
    return gray


gray = run(skd.astronaut(), "example_astronaut.png", 0.8)

# scene with faces of different sizes pasted onto a face-free photo (faces come from the training set!)
bg, lf = skd.coffee().copy(), skd.lfw_subset()
for i, s, x, y in [(3, 90, 60, 40), (20, 64, 300, 150), (41, 120, 400, 230)]:
    face = (cv2.resize(lf[i].astype(np.float32), (s, s), interpolation=cv2.INTER_CUBIC) * 255).astype(np.uint8)
    bg[y:y + s, x:x + s] = face[..., None]
run(bg, "example_composite.png", 0.9)

viz.bank_image().save(os.path.join(OUT, "gabor_bank.png"))
viz.response_image(gray, face_filters()).save(os.path.join(OUT, "gabor_responses.png"))
