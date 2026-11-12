import cv2 as cv
import torch
import numpy as np

from models.yolo import YOLO
from utils.detection import decode_predictions


IMAGE_SIZE = 448
NUM_CLASSES = 3

CLASS_NAMES = ["cat", "dog", "person"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load model
model = YOLO(num_classes=NUM_CLASSES)

model.load_state_dict(
    torch.load(
        "yolo_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()


# Read image
image_path = "test.jpg"

image = cv.imread(image_path)

if image is None:
    raise ValueError("Cannot read image")


original = image.copy()

# Preprocess
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

resized = cv.resize(
    image_rgb,
    (IMAGE_SIZE, IMAGE_SIZE)
)

tensor = torch.from_numpy(
    resized.astype(np.float32) / 255.0
)

tensor = tensor.permute(2, 0, 1).unsqueeze(0)

tensor = tensor.to(device)


# Inference
with torch.no_grad():

    predictions = model(tensor)

    detections = decode_predictions(
        predictions,
        confidence_threshold=0.25
    )


# Draw detections
h, w = original.shape[:2]

for detection in detections:

    x1, y1, x2, y2, score, class_id = detection

    x1 = int(x1 * w / IMAGE_SIZE)
    x2 = int(x2 * w / IMAGE_SIZE)

    y1 = int(y1 * h / IMAGE_SIZE)
    y2 = int(y2 * h / IMAGE_SIZE)

    class_name = CLASS_NAMES[int(class_id)]

    cv.rectangle(
        original,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"{class_name}: {score:.2f}"

    cv.putText(
        original,
        label,
        (x1, max(20, y1 - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


cv.imshow("YOLO Object Detection", original)

cv.waitKey(0)
cv.destroyAllWindows()
