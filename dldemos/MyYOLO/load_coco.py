import json
import os


def print_json():
    with open('data/coco/annotations/instances_val2014.json') as fp:
        root = json.load(fp)
    print('info:')
    print(root['info'])
    print('categories:')
    print(root['categories'])
    print('Length of images:', len(root['images']))
    print(root['images'][0])
    print('Length of annotations:', len(root['annotations']))
    print(root['annotations'][0])


def load_img_ann():
    """return [{img_name, [{x, y, h, w, label}]}]"""
    with open('data/coco/annotations/instances_val2014.json') as fp:
        root = json.load(fp)
    img_dict = {}
    for img_info in root['images']:
        img_dict[img_info['id']] = {'name': img_info['file_name'], 'anns': []}
    for ann_info in root['annotations']:
        img_dict[ann_info['image_id']]['anns'].append(
            ann_info['bbox'] + [ann_info['category_id']])

    return img_dict


def show_img_ann(img_info):
    from PIL import Image

    from dldemos.nms.show_bbox import draw_bbox
    print(img_info)

    with open('data/coco/annotations/instances_val2014.json') as fp:
        root = json.load(fp)
    categories = root['categories']
    category_dict = {int(c['id']): c['name'] for c in categories}

    img_path = os.path.join('data/coco/val2014', img_info['name'])
    img = Image.open(img_path)
    for ann in img_info['anns']:
        x, y, w, h = ann[0:4]
        x1, y1, x2, y2 = x, y, x + w, y + h
        draw_bbox(img, (x1, y1, x2, y2), 1.0, text=category_dict[ann[4]])

    img.save('work_dirs/tmp.jpg')


def main():
    print_json()
    img_dict = load_img_ann()
    keys = list(img_dict.keys())
    show_img_ann(img_dict[keys[1]])


if __name__ == '__main__':
    main()
