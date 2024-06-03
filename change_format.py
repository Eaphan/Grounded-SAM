import os
import json
import numpy as np
from PIL import Image
from multiprocessing import Pool

def process_image(file_path):
    if file_path.endswith(('.jpg', '.png', '.jpeg')):  # 确保处理的是图片文件
        new_path = file_path.replace('nuscenes_seg_grounded_sam_output', 'camseg')
        os.makedirs(os.path.dirname(new_path), exist_ok=True)  # 确保新目录存在

        # with Image.open(file_path) as img:
        #     img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转图片
        #     img_flipped.save(new_path)  # 保存翻转后的图片
        img = Image.open(file_path)
        img = np.array(img)

        im_mask_json_path = file_path.replace("png", "json")
        im_mask_json = json.load(open(im_mask_json_path, 'r'))

        l_map_dict = {item['value']:item['label'] for item in im_mask_json}
        im_mask = np.vectorize(l_map_dict.get)(img)
        im_mask = Image.fromarray(np.uint8(im_mask))
        im_mask.save(new_path)

def walk_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def main():
    base_dir = 'nuscenes_seg_grounded_sam_output'
    pool = Pool(processes=os.cpu_count())  # 创建与 CPU 数量等同的进程池
    pool.map(process_image, walk_directory(base_dir))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()