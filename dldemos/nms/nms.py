import numpy as np
from PIL import Image

from dldemos.nms.iou import iou
from dldemos.nms.show_bbox import draw_bbox


class BoxRenderer():
    def __init__(self, img: str, predicts: np.ndarray):
        self.ori_img = Image.open(img)
        self.imgs = []
        self.durations = [500]
        self.predicts = predicts

        # Mode number explanation:
        # 0: original box
        # 1: faint
        # 2: invisible
        # 3: highlight
        self.box_mode = [0] * self.predicts.shape[0]
        self.render(2000)

    def reset(self, index):
        self.box_mode[index] = 0

    def fade_out(self, index):
        self.box_mode[index] = 1

    def hide(self, index):
        self.box_mode[index] = 2

    def highlight(self, index):
        self.box_mode[index] = 3

    def render(self, duration: int):
        img = self.ori_img.copy()
        for p, mode in zip(self.predicts, self.box_mode):
            if mode == 0:
                draw_bbox(img, p[1:], p[0], (255, 100, 100, 200))
            elif mode == 1:
                draw_bbox(img, p[1:], p[0], (255, 200, 200, 150))
            elif mode == 3:
                draw_bbox(img, p[1:], p[0])
        self.imgs.append(img)
        self.durations.append(duration)

    def save(self, fn):
        self.ori_img.save(fn,
                          save_all=True,
                          append_images=self.imgs,
                          duration=self.durations,
                          loop=0)


def nms_render(predicts: np.ndarray,
               renderer: BoxRenderer,
               score_thresh=0.6,
               iou_thresh=0.3):
    """Non-Maximum Suppression

    Args:
        predicts (np.ndarray): Tensor of shape [n, 5]. The second demesion
            includes 1 probability and 4 numbers x, y, w, h denoting a bounding
            box.
    """
    # False for unvisited item
    n_remainder = len(predicts)
    vis = [False] * n_remainder

    # Filter predicts with low probability
    for i, predict in enumerate(predicts):
        if predict[0] < score_thresh:
            vis[i] = True
            n_remainder -= 1
            renderer.fade_out(i)
            renderer.render(800)

    for i, predict in enumerate(predicts):
        if predict[0] < score_thresh:
            renderer.hide(i)

    renderer.render(1000)

    # NMS
    output_predicts = []
    while n_remainder > 0:
        max_pro = -1
        max_index = 0
        # Find argmax
        for i, p in enumerate(predicts):
            if not vis[i]:
                if max_pro < p[0]:
                    max_index = i
                    max_pro = p[0]

        # Append output
        max_p = predicts[max_index]
        output_predicts.append(max_p)
        renderer.highlight(max_index)
        renderer.render(1000)

        # Suppress
        tmp_indices = []
        for i, p in enumerate(predicts):
            if not vis[i] and i != max_index:
                if iou(p[1:5], max_p[1:5]) > iou_thresh:
                    vis[i] = True
                    n_remainder -= 1
                    renderer.fade_out(i)
                    renderer.render(800)
                    tmp_indices.append(i)

        for i in tmp_indices:
            renderer.hide(i)
        renderer.render(1000)

        vis[max_index] = True
        n_remainder -= 1

    return output_predicts


def nms(predicts: np.ndarray, score_thresh=0.6, iou_thresh=0.3):
    """Non-Maximum Suppression

    Args:
        predicts (np.ndarray): Tensor of shape [n, 5]. The second demesion
            includes 1 probability and 4 numbers x, y, w, h denoting a bounding
            box.
    """
    # Filter predicts with low probability
    filtered_predicts = []
    for predict in predicts:
        if predict[0] >= score_thresh:
            filtered_predicts.append(predict)

    # NMS
    n_remainder = len(filtered_predicts)
    vis = [False] * n_remainder  # False for unvisited item
    output_predicts = []
    while n_remainder > 0:
        max_pro = -1
        max_index = 0
        # Find argmax
        for i, p in enumerate(filtered_predicts):
            if not vis[i]:
                if max_pro < p[0]:
                    max_index = i
                    max_pro = p[0]

        # Append output
        max_p = filtered_predicts[max_index]
        output_predicts.append(max_p)

        # Suppress
        for i, p in enumerate(filtered_predicts):
            if not vis[i] and i != max_index:
                if iou(p[1:5], max_p[1:5]) > iou_thresh:
                    vis[i] = True
                    n_remainder -= 1
        vis[max_index] = True
        n_remainder -= 1

    return output_predicts


def main():
    predicts = np.array([(0.95, 191, 105, 294, 157), (0.8, 168, 111, 280, 150),
                         (0.7, 218, 113, 284, 159), (0.3, 193, 140, 231, 153),
                         (0.7, 323, 112, 380, 145), (0.8, 305, 107, 364, 134),
                         (0.9, 294, 114, 376, 151), (0.3, 319, 138, 358, 155)])
    renderer = BoxRenderer('work_dirs/detection_demo.jpg', predicts)
    nms_render(predicts, renderer)
    renderer.save('work_dirs/NMS/2.gif')


if __name__ == '__main__':
    main()
