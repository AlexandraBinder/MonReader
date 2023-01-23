import os
from PIL import Image
import argparse
from rembg import remove
import os


def main(args):
    # rename_and_reorganize_images(args.image_path, args.original_image_path, args.image_split_path)
    final_images_path = os.path.join(args.image_path, args.image_split_path)
    categories = os.listdir(final_images_path)
    for cat in categories:
        dir_path = os.path.join(final_images_path, cat)

        for img_name in os.listdir(dir_path):
            img = Image.open(os.path.join(final_images_path, cat, img_name))
            img = remove(img)           # remove background
            img = img.convert('L')      # convert to grayscale
            img.save(os.path.join(args.image_path, args.final_image_path, cat, img_name))
      


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default='images')
    parser.add_argument("--original_image_path", default='original_personal')
    parser.add_argument("--image_split_path", default='final')
    parser.add_argument("--final_image_path", default='final')
    return parser.parse_args()

def rename_and_reorganize_images(img_path, original_img_path, image_split_path):
    count_no_flip = 0
    count_flip = 0
    for root in os.walk(os.path.join(img_path, original_img_path)):

        if root == os.path.join(img_path, original_img_path):
            flip_dir = os.path.join(img_path, image_split_path, 'flip')
            no_flip_dir = os.path.join(img_path, image_split_path, 'no flip')
            os.makedirs(flip_dir, exist_ok=True)
            os.makedirs(no_flip_dir, exist_ok=True)

        if 'no' in root:
            img_list = os.listdir(root)
            for img in img_list:
                os.link(os.path.join(root, img), os.path.join(no_flip_dir, f'no_flip_img_{str(count_no_flip)}.png'))
                count_no_flip += 1
        
        elif 'flip' in root:
            img_list = os.listdir(root)
            for img in img_list:
                os.link(os.path.join(root, img), os.path.join(flip_dir, f'flip_img_{str(count_flip)}.png'))
                count_flip += 1


if __name__ == '__main__':
    args = get_args()
    main(args)