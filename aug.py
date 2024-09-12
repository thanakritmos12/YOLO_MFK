import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random

# กำหนดเส้นทางที่เก็บรูปภาพเดิมและที่เก็บรูปภาพใหม่
imgs_source_folder = r'D:\yolov5-master\data\images\train'  # โฟลเดอร์ที่เก็บไฟล์ imgs
imgs_destination_folder = r'D:\yolov5-master\data\images\augm_train'  # โฟลเดอร์ที่ต้องการเซฟไฟล์ imgs ใหม่

labels_source_folder = r'D:\yolov5-master\data\labels\train'  # โฟลเดอร์ที่เก็บไฟล์ labels
labels_destination_folder = r'D:\yolov5-master\data\labels\augm_train'  # โฟลเดอร์ที่ต้องการเซฟไฟล์ labels ใหม่

# ตรวจสอบให้แน่ใจว่าโฟลเดอร์ปลายทางมีอยู่แล้ว
if not os.path.exists(imgs_destination_folder):
    os.makedirs(imgs_destination_folder)
if not os.path.exists(labels_destination_folder):
    os.makedirs(labels_destination_folder)

# Helper function to apply transformations to bounding boxes
def flip_bboxes(bboxes, img_width):
    """Flip bounding boxes horizontally"""
    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 1] = 1 - bboxes[:, 1]  # Flip the x-center
    return flipped_bboxes

def rotate_bboxes_90(bboxes, img_width, img_height, clockwise=True):
    """Rotate bounding boxes 90 degrees"""
    rotated_bboxes = bboxes.copy()
    if clockwise:
        rotated_bboxes[:, [1, 2]] = np.column_stack((bboxes[:, 2], 1 - bboxes[:, 1]))
        rotated_bboxes[:, [3, 4]] = np.column_stack((bboxes[:, 4], bboxes[:, 3]))
    else:
        rotated_bboxes[:, [1, 2]] = np.column_stack((1 - bboxes[:, 2], bboxes[:, 1]))
        rotated_bboxes[:, [3, 4]] = np.column_stack((bboxes[:, 4], bboxes[:, 3]))
    return rotated_bboxes

# ฟังก์ชันเพิ่มจุดสีขาวและดำ
def add_white_and_black_spots(img):
    img = img.convert('RGB')  # Convert to RGB if not already
    np_img = np.array(img)
    num_spots = int(np_img.size * 0.01)  # จำนวนจุดที่ต้องการเพิ่ม (1% ของพิกเซลทั้งหมด)
    
    for _ in range(num_spots):
        x = random.randint(0, np_img.shape[1] - 1)
        y = random.randint(0, np_img.shape[0] - 1)
        if random.choice([True, False]):
            np_img[y, x] = [255, 255, 255]  # เพิ่มจุดสีขาว
        else:
            np_img[y, x] = [0, 0, 0]  # เพิ่มจุดสีดำ
    
    return Image.fromarray(np_img)

# ฟังก์ชันทำ augmentation สำหรับโฟลเดอร์ใดโฟลเดอร์หนึ่ง
def augment_images(source_folder, destination_folder, labels_folder, destination_labels_folder, do_noise_and_blur=True):
    files = [f for f in sorted(os.listdir(source_folder)) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # วนลูปไฟล์ทั้งหมดที่พบ
    for idx, file_name in enumerate(files, start=1):
        file_path = os.path.join(source_folder, file_name)
        img = Image.open(file_path).convert('RGB')  # Ensure image is in RGB mode
        img_width, img_height = img.size

        # Load corresponding label
        label_name = file_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(labels_folder, label_name)
        if not os.path.exists(label_path):
            continue  # Skip if the label does not exist

        # Load bounding boxes
        bboxes = np.loadtxt(label_path).reshape(-1, 5)  # [class_id, x_center, y_center, width, height]

        # แยกชื่อไฟล์และนามสกุล
        name, ext = os.path.splitext(file_name)

        # 1. กลับหัวรูปภาพ (flip horizontally)
        flipped_img = ImageOps.mirror(img)
        flipped_bboxes = flip_bboxes(bboxes, img_width)
        flipped_img.save(os.path.join(destination_folder, f"{str(100 + (idx-1) * 7 + 1).zfill(3)}{ext}"))
        np.savetxt(os.path.join(destination_labels_folder, f"{str(100 + (idx-1) * 7 + 1).zfill(3)}.txt"), flipped_bboxes, fmt='%f')

        # 2. เอียงขวา (rotate right)
        rotated_right_img = img.rotate(-90, expand=True)
        rotated_right_bboxes = rotate_bboxes_90(bboxes, img_width, img_height, clockwise=True)
        rotated_right_img.save(os.path.join(destination_folder, f"{str(100 + (idx-1) * 7 + 2).zfill(3)}{ext}"))
        np.savetxt(os.path.join(destination_labels_folder, f"{str(100 + (idx-1) * 7 + 2).zfill(3)}.txt"), rotated_right_bboxes, fmt='%f')

        # 3. เอียงซ้าย (rotate left)
        rotated_left_img = img.rotate(90, expand=True)
        rotated_left_bboxes = rotate_bboxes_90(bboxes, img_width, img_height, clockwise=False)
        rotated_left_img.save(os.path.join(destination_folder, f"{str(100 + (idx-1) * 7 + 3).zfill(3)}{ext}"))
        np.savetxt(os.path.join(destination_labels_folder, f"{str(100 + (idx-1) * 7 + 3).zfill(3)}.txt"), rotated_left_bboxes, fmt='%f')

        # ถ้าอนุญาตให้ทำ noise และ blur
        if do_noise_and_blur:
            # 4. เพิ่มทั้งจุดสีขาวและสีดำ (add white and black spots)
            white_black_spots_img = add_white_and_black_spots(img)
            white_black_spots_img.save(os.path.join(destination_folder, f"{str(100 + (idx-1) * 7 + 4).zfill(3)}{ext}"))
            np.savetxt(os.path.join(destination_labels_folder, f"{str(100 + (idx-1) * 7 + 4).zfill(3)}.txt"), bboxes, fmt='%f')

            # 5. ทำการเบลอ (blur)
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))
            blurred_img.save(os.path.join(destination_folder, f"{str(100 + (idx-1) * 7 + 5).zfill(3)}{ext}"))
            np.savetxt(os.path.join(destination_labels_folder, f"{str(100 + (idx-1) * 7 + 5).zfill(3)}.txt"), bboxes, fmt='%f')

        else:
            # 4. บันทึกภาพปกติแทนการเพิ่ม noise
            img.save(os.path.join(destination_folder, f"{str(100 + (idx-1) * 7 + 4).zfill(3)}{ext}"))
            np.savetxt(os.path.join(destination_labels_folder, f"{str(100 + (idx-1) * 7 + 4).zfill(3)}.txt"), bboxes, fmt='%f')

# เรียกใช้ฟังก์ชันสำหรับ imgs (ทำ noise และ blur)
augment_images(imgs_source_folder, imgs_destination_folder, labels_source_folder, labels_destination_folder, do_noise_and_blur=True)

print("Finish!")
