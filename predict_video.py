
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


# Load trained model
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


# Load video
cap = cv.VideoCapture(
    "comp_vis/opencv/videos/kitten.mp4"
)

if not cap.isOpened():
    raise ValueError("Cannot open video")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Video processing completed")
        break

    original = frame.copy()

    # Preprocessing
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    resized = cv.resize(
        rgb,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    tensor = tensor.to(device)


    # YOLO inference
    with torch.no_grad():

        predictions = model(tensor)

        detections = decode_predictions(
            predictions,
            confidence_threshold=0.25
        )


    # Draw boxes
    h, w = original.shape[:2]

    for detection in detections:

        x1, y1, x2, y2, score, class_id = detection

        x1 = int(x1 * w / IMAGE_SIZE)
        x2 = int(x2 * w / IMAGE_SIZE)

        y1 = int(y1 * h / IMAGE_SIZE)
        y2 = int(y2 * h / IMAGE_SIZE)

        label = (
            f"{CLASS_NAMES[int(class_id)]}: {score:.2f}"
        )

        cv.rectangle(
            original,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv.putText(
            original,
            label,
            (x1, max(20, y1 - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    cv.imshow("YOLO Video Detection", original)

    if cv.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv.destroyAllWindows()
