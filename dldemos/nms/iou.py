from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


def box_intersection(
        b1: Tuple[int, int, int, int],
        b2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    xl = max(x11, x21)
    xr = min(x12, x22)
    yt = max(y11, y21)
    yb = min(y12, y22)
    return (xl, yt, xr, yb)


def area(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    return width * height


def iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    intersection = box_intersection(b1, b2)
    inter_area = area(intersection)
    union_area = area(b1) + area(b2) - inter_area
    return inter_area / union_area


def main():
    img0 = Image.new('RGB', (400, 200), 'white')
    imgs = []
    durations = [200]
    img = img0.copy()
    image_darw = ImageDraw.Draw(img)
    bbox1 = (70, 70, 160, 150)
    bbox2 = (40, 60, 140, 130)
    text_x = 170
    text_y = 30
    font = ImageFont.truetype(
        'times.ttf',
        16,
    )

    def draw_line_of_text(text: str):
        nonlocal text_y, image_darw
        tw, th = font.getsize(text)
        image_darw.text((text_x, text_y), text, 'black')
        text_y += th

    image_darw.rectangle(bbox1, outline='orange', width=2)
    imgs.append(img.copy())
    durations.append(500)
    image_darw.rectangle(bbox2, outline='purple', width=2)
    imgs.append(img.copy())
    durations.append(500)

    image_darw.rectangle(bbox1, outline='orange', fill='orange', width=2)
    draw_line_of_text(f'a1 = {area(bbox1)}')
    imgs.append(img.copy())
    durations.append(800)

    image_darw.rectangle(bbox2, outline='purple', fill='purple', width=2)
    draw_line_of_text(f'a2 = {area(bbox2)}')
    imgs.append(img.copy())
    durations.append(800)

    ibox = box_intersection(bbox1, bbox2)
    image_darw.rectangle(ibox, outline='red', fill='red', width=2)
    draw_line_of_text(f'i = {area(ibox)}')
    imgs.append(img.copy())
    durations.append(1000)

    image_darw.rectangle(bbox1, outline='green', fill='green', width=2)
    image_darw.rectangle(bbox2, outline='green', fill='green', width=2)
    draw_line_of_text(
        f'u = a1 + a2 - i = {area(bbox1) + area(bbox2) - area(ibox)}')
    imgs.append(img.copy())
    durations.append(1500)

    image_darw.rectangle(ibox, outline='red', fill='red', width=2)
    draw_line_of_text(f'iou = i / o = {iou(bbox1, bbox2)}')
    imgs.append(img.copy())
    durations.append(2000)

    img0.save('work_dirs/NMS/1.gif',
              save_all=True,
              append_images=imgs,
              duration=durations,
              loop=0)


if __name__ == '__main__':
    main()
