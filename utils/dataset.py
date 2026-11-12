
import os
import torch
import cv2
import numpy as np

from torch.utils.data import Dataset


class YOLODataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=448,
        grid_size=7,
        num_classes=3
    ):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.image_size = image_size
        self.grid_size = grid_size
        self.num_classes = num_classes

        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir, image_name
        )

        label_name = os.path.splitext(image_name)[0] + ".txt"

        label_path = os.path.join(
            self.label_dir, label_name
        )

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1)

        # Target:
        # [grid, grid, 5 + classes]

        target = torch.zeros(
            self.grid_size,
            self.grid_size,
            5 + self.num_classes
        )

        if os.path.exists(label_path):

            with open(label_path, "r") as file:

                for line in file:

                    values = list(map(float, line.split()))

                    class_id, x, y, w, h = values

                    class_id = int(class_id)

                    if not 0 <= class_id < self.num_classes:
                        raise ValueError("Invalid class ID")

                    if not all(0 <= v <= 1 for v in (x, y, w, h)):
                        raise ValueError("Invalid normalized box")

                    # Find grid cell
                    cell_x = min(
                        int(x * self.grid_size),
                        self.grid_size - 1
                    )

                    cell_y = min(
                        int(y * self.grid_size),
                        self.grid_size - 1
                    )

                    # One object per cell in this simplified model
                    if target[cell_y, cell_x, 4] == 1:
                        continue

                    # Coordinates relative to cell
                    x_cell = x * self.grid_size - cell_x
                    y_cell = y * self.grid_size - cell_y

                    target[cell_y, cell_x, 0] = x_cell
                    target[cell_y, cell_x, 1] = y_cell

                    target[cell_y, cell_x, 2] = w
                    target[cell_y, cell_x, 3] = h

                    target[cell_y, cell_x, 4] = 1

                    target[
                        cell_y, cell_x, 5 + class_id
                    ] = 1

        return image, target
