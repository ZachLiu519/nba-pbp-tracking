import numpy as np

from .constants import FIELD_HEIGHT, FIELD_WIDTH


def getFieldPoints(scaled_width: int, scaled_height: int) -> np.ndarray:
    """Generate 91 scaled field points for a basketball court.

    Args:
        scaled_width (int): Width of the basketball court to be scaled onto.
        scaled_height (int): Height of the basketball court to be scaled onto.

    Returns:
        np.ndarray: Scaled field points for a basketball court.
    """
    points = []
    u0 = 175
    r = 30
    u = u0
    s = 0
    for _ in range(0, 7):
        for i in range(0, 13):
            points.append([i * FIELD_WIDTH / 12, FIELD_HEIGHT - s])
        s += u
        u += r

    points = np.array(
        [
            [x * scaled_width / FIELD_WIDTH, y * scaled_height / FIELD_HEIGHT]
            for x, y in points
        ],
        dtype=np.float32,
    )

    return points


def find_midpoint_lower_side(xywh: np.ndarray) -> np.ndarray:
    """Find the midpoint of the lower side of the bounding box. Use this to determine player positions.

    Args:
        xywh (np.ndarray): Bounding box coordinates.

    Returns:
        np.ndarray: Midpoints of the lower side of the bounding box.
    """
    midpoints = []
    for box in xywh:
        x, y, w, h = box
        # Calculate the midpoint of the lower side
        midpoints.append((x, y + h / 2))
    return np.array(midpoints)
