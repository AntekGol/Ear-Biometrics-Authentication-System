import os
import cv2
import numpy as np
import random
#skalowanie obrazu z zachowaniem proporcji
def resize_with_padding(image, target_size=(224, 224)):
    old_size = image.shape[:2]
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image
#wymazywanie losowych pikseli
def random_erasing(image, max_rects=3):
    h, w = image.shape[:2]
    erased = image.copy()
    for _ in range(random.randint(1, max_rects)):
        erase_w = random.randint(int(0.1*w), int(0.2*w))
        erase_h = random.randint(int(0.1*h), int(0.2*h))
        x1 = random.randint(0, w - erase_w)
        y1 = random.randint(0, h - erase_h)
        erased[y1:y1+erase_h, x1:x1+erase_w] = np.random.randint(0, 256, (erase_h, erase_w, 3), dtype=np.uint8)
    return erased

def augment_image(image):
    augmented_images = []

    # Oryginalne
    augmented_images.append(image)

    # Kolor mono
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    augmented_images.append(gray)

    # Obrót -20 do -5 stopni
    rows, cols = image.shape[:2]
    kat_lewo = random.uniform(-20, -5)
    M_left = cv2.getRotationMatrix2D((cols/2, rows/2), kat_lewo, 1)
    rotated_left = cv2.warpAffine(image, M_left, (cols, rows))
    augmented_images.append(rotated_left)

    # Obrót +5 do +20 stopni
    kat_prawo = random.uniform(5, 20)
    M_right = cv2.getRotationMatrix2D((cols/2, rows/2), kat_prawo, 1)
    rotated_right = cv2.warpAffine(image, M_right, (cols, rows))
    augmented_images.append(rotated_right)

    # Noise (gauss)
    noise = np.random.randint(-35, 36, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy_image)

    # Blur (gauss)
    blurred = cv2.blur(image, (15, 15))
    augmented_images.append(blurred)

    # Jasność większa
    bw = random.uniform(1.35, 1.45)
    bright = cv2.convertScaleAbs(image, bw, beta=30)
    augmented_images.append(bright)

    # Jasność mniejsza
    bm = random.uniform(0.2, 0.3)
    dark = cv2.convertScaleAbs(image, bm, beta=-30)
    augmented_images.append(dark)

    # Saturacja zmieniona (mocniej i słabiej)
    sm = random.uniform(0.3, 0.6)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sm, 0, 255)
    sat_low = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    augmented_images.append(sat_low)

    sw = random.uniform(1.3, 1.6)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sw, 0, 255)
    sat_high = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    augmented_images.append(sat_high)

    # Random erasing (3 wersje)
    for _ in range(3):
        erased = random_erasing(image)
        augmented_images.append(erased)


    # Zoom
    zoom_factor = random.uniform(0.7, 0.9)
    h, w = image.shape[:2]
    # Skalowanie obrazu
    zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
    zh, zw = zoomed.shape[:2]

    # Padding - dopasowanie do 224x224
    pad_top = (h - zh) // 2
    pad_bottom = h - zh - pad_top
    pad_left = (w - zw) // 2
    pad_right = w - zw - pad_left
    zoomed = cv2.copyMakeBorder(zoomed, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


    augmented_images.append(zoomed)

    # (odbicie lustrzane poziome)
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    return augmented_images

def process_images(input_folder, output_folder="augmented_ears", target_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Nie można wczytać obrazu: {image_file}")
            continue

        augmented_imgs = augment_image(image)

        folder_name = f"user{idx}"
        output_ear_folder = os.path.join(output_folder, folder_name)
        os.makedirs(output_ear_folder, exist_ok=True)

        for i, aug_img in enumerate(augmented_imgs):
            resized = resize_with_padding(aug_img, target_size)
            save_path = os.path.join(output_ear_folder, f"{folder_name}_aug{i}.jpg")
            cv2.imwrite(save_path, resized)

process_images("uszy2")

