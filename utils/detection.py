import torch
from torchvision.ops import nms


def decode_predictions(
    predictions,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    image_size=448
):

    results = []

    grid_size = predictions.shape[1]

    predictions = predictions[0]

    for row in range(grid_size):

        for col in range(grid_size):

            cell = predictions[row, col]

            # Object confidence
            confidence = torch.sigmoid(cell[4])

            # Class probabilities
            class_probs = torch.softmax(cell[5:], dim=0)

            class_conf, class_id = torch.max(
                class_probs, dim=0
            )

            score = confidence * class_conf

            if score.item() < confidence_threshold:
                continue

            # Decode center coordinates
            x = (col + torch.sigmoid(cell[0])) / grid_size
            y = (row + torch.sigmoid(cell[1])) / grid_size

            # Decode width and height
            w = torch.sigmoid(cell[2])
            h = torch.sigmoid(cell[3])

            # Convert to pixel coordinates
            x1 = (x - w / 2) * image_size
            y1 = (y - h / 2) * image_size

            x2 = (x + w / 2) * image_size
            y2 = (y + h / 2) * image_size

            x1 = x1.clamp(0, image_size)
            y1 = y1.clamp(0, image_size)
            x2 = x2.clamp(0, image_size)
            y2 = y2.clamp(0, image_size)

            results.append([
                x1.item(),
                y1.item(),
                x2.item(),
                y2.item(),
                score.item(),
                class_id.item()
            ])

    if not results:
        return []

    detections = torch.tensor(results)

    # Apply NMS separately for each class
    final_detections = []

    for class_id in detections[:, 5].unique():

        class_mask = detections[:, 5] == class_id

        class_boxes = detections[class_mask, :4]
        class_scores = detections[class_mask, 4]

        keep = nms(
            class_boxes,
            class_scores,
            iou_threshold
        )

        final_detections.append(
            detections[class_mask][keep]
        )

    if not final_detections:
        return []

    return torch.cat(final_detections, dim=0).tolist()
