from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def draw_bbox(img: Image.Image,
              bbox: Tuple[float, float, float, float],
              prob: float,
              rect_color: Tuple[int, int, int] = (255, 0, 0),
              text: Optional[str] = None,
              better_font: Optional[str] = None):
    img_draw = ImageDraw.Draw(img, 'RGBA')
    x1, y1, x2, y2 = bbox
    if better_font is not None:
        font = ImageFont.truetype(
            better_font,
            12,
        )
    else:
        font = ImageFont.load_default()

    img_draw.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2),
                       outline=rect_color,
                       width=2)

    # Show class label on the top right corner
    if text is not None:
        tw, th = font.getsize(text)
        img_draw.rectangle((x2 - tw, y1, x2, y1 + th), fill='black')
        img_draw.text((x2 - tw, y1), text, font=font, anchor='rt')

    # Show probablity of top left corner
    tw, th = font.getsize(f'{prob:.2f}')
    img_draw.rectangle((x1, y1, x1 + tw, y1 + th), fill='black')
    img_draw.text((x1, y1), f'{prob:.2f}', font=font)


def main():
    img = Image.open('work_dirs/detection_demo.jpg')
    draw_bbox(img, (191, 105, 294, 157), 0.95)
    draw_bbox(img, (168, 111, 280, 150), 0.8)
    draw_bbox(img, (218, 113, 284, 159), 0.7)
    draw_bbox(img, (193, 140, 231, 153), 0.3)

    draw_bbox(img, (323, 112, 380, 145), 0.7)
    draw_bbox(img, (305, 107, 364, 134), 0.8)
    draw_bbox(img, (294, 114, 376, 151), 0.9)
    draw_bbox(img, (319, 138, 358, 155), 0.3)
    img.save('work_dirs/NMS/1.jpg')


if __name__ == '__main__':
    main()
