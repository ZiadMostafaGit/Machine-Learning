# Make sure your code is robust for many corner cases

def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def calculate_iou(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    if ax2 < bx1 or bx2 < ax1:
        return 0  # no intersection

    if ay2 < by1 or by2 < ay1:
        return 0  # no intersection

    int_x1 = max(ax1, bx1)
    int_x2 = min(ax2, bx2)

    int_y1 = max(ay1, by1)
    int_y2 = min(ay2, by2)


    intersect_area = area(int_x1, int_y1, int_x2, int_y2)
    area1 = area(ax1, ay1, ax2, ay2)
    area2 = area(bx1, by1, bx2, by2)
    union = area1 + area2 - intersect_area

    if union == 0:
        return 0

    return intersect_area / union



if __name__ == '__main__':
    box1 = (10, 10, 50, 50)
    box2 = (30, 30, 70, 70)

    print(calculate_iou(box1, box2))
