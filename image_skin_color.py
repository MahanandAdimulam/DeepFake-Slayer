import os
import cv2
import numpy as np
import csv
from mediapipe import solutions
import glob
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_clahe(denormalized_image):
    scaled_image = (denormalized_image * 255).astype(np.uint8)
    lab_image = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    lab_clahe_image = cv2.merge((l_clahe, a_channel, b_channel))
    normalized_image = cv2.cvtColor(lab_clahe_image, cv2.COLOR_LAB2RGB)
    return normalized_image


def get_region_mask(image, landmarks, region_points):
    height, width, _ = image.shape
    points = [(int(landmarks[pt].x * width), int(landmarks[pt].y * height)) for pt in region_points]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
    return mask


def compute_color_features(image, mask):
    image_tensor = torch.from_numpy(image).to(device).float()
    mask_tensor = torch.from_numpy(mask).to(device).unsqueeze(-1).float()
    region = image_tensor * mask_tensor
    region = region.cpu().numpy().astype(np.uint8)
    rgb_mean = region[mask == 1].mean(axis=0)
    lab_image = cv2.cvtColor(region, cv2.COLOR_RGB2Lab)
    lab_mean = lab_image[mask == 1].mean(axis=0)
    ycbcr_image = cv2.cvtColor(region, cv2.COLOR_RGB2YCrCb)
    ycbcr_mean = ycbcr_image[mask == 1].mean(axis=0)
    return lab_mean, rgb_mean, ycbcr_mean


def process_images_from_total_results(total_results, output_csv, batch_size=16, start_index=0):
    mp_face_mesh = solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.2
    )
    face_detected = 0
    face_not_detected = 0
    total_images = len(total_results['img'])

    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = [
            "Image_Index", "Avg_Lab_L", "Avg_Lab_a", "Avg_Lab_b",
            "Avg_RGB_R", "Avg_RGB_G", "Avg_RGB_B",
            "Avg_YCbCr_Y", "Avg_YCbCr_Cb", "Avg_YCbCr_Cr"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if start_index == 0:
            writer.writeheader()

        for batch_start in range(start_index, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_images = total_results['img'][batch_start:batch_end]
            print(f"Processing batch from {batch_start} to {batch_end - 1}")
            try:
                for idx, tensor_image in enumerate(batch_images, start=batch_start):
                    image = np.transpose(tensor_image, (1, 2, 0))
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)

                    mean = np.array([0.5, 0.5, 0.5])
                    std = np.array([0.5, 0.5, 0.5])
                    denormalized_image = tensor_image * std[:, None, None] + mean[:, None, None]
                    denormalized_image = np.transpose(denormalized_image, (1, 2, 0))
                    clahe_image = apply_clahe(denormalized_image)
                    image = clahe_image

                    results = mp_face_mesh.process(image)
                    if results.multi_face_landmarks:
                        face_detected += 1
                        landmarks = results.multi_face_landmarks[0].landmark
                        left_cheek_mask = get_region_mask(image, landmarks, [50, 118, 101])
                        right_cheek_mask = get_region_mask(image, landmarks, [330, 280, 347])
                        nose_tip_mask = get_region_mask(image, landmarks, [5, 4, 51])
                        forehead_mask = get_region_mask(image, landmarks, [9, 107, 66, 105, 63, 68, 54, 103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336])

                        left_lab, left_rgb, left_ycbcr = compute_color_features(image, left_cheek_mask)
                        right_lab, right_rgb, right_ycbcr = compute_color_features(image, right_cheek_mask)
                        nose_lab, nose_rgb, nose_ycbcr = compute_color_features(image, nose_tip_mask)
                        forehead_lab, forehead_rgb, forehead_ycbcr = compute_color_features(image, forehead_mask)

                        avg_lab_l = (left_lab[0] + right_lab[0] + nose_lab[0] + forehead_lab[0]) / 4
                        avg_lab_a = (left_lab[1] + right_lab[1] + nose_lab[1] + forehead_lab[1]) / 4
                        avg_lab_b = (left_lab[2] + right_lab[2] + nose_lab[2] + forehead_lab[2]) / 4

                        avg_rgb_r = (left_rgb[0] + right_rgb[0] + nose_rgb[0] + forehead_rgb[0]) / 4
                        avg_rgb_g = (left_rgb[1] + right_rgb[1] + nose_rgb[1] + forehead_rgb[1]) / 4
                        avg_rgb_b = (left_rgb[2] + right_rgb[2] + nose_rgb[2] + forehead_rgb[2]) / 4

                        avg_ycbcr_y = (left_ycbcr[0] + right_ycbcr[0] + nose_ycbcr[0] + forehead_ycbcr[0]) / 4
                        avg_ycbcr_cb = (left_ycbcr[1] + right_ycbcr[1] + nose_ycbcr[1] + forehead_ycbcr[1]) / 4
                        avg_ycbcr_cr = (left_ycbcr[2] + right_ycbcr[2] + nose_ycbcr[2] + forehead_ycbcr[2]) / 4

                        writer.writerow({
                            "Image_Index": idx,
                            "Avg_Lab_L": avg_lab_l, "Avg_Lab_a": avg_lab_a, "Avg_Lab_b": avg_lab_b,
                            "Avg_RGB_R": avg_rgb_r, "Avg_RGB_G": avg_rgb_g, "Avg_RGB_B": avg_rgb_b,
                            "Avg_YCbCr_Y": avg_ycbcr_y, "Avg_YCbCr_Cb": avg_ycbcr_cb, "Avg_YCbCr_Cr": avg_ycbcr_cr,
                        })
                    else:
                        face_not_detected += 1
                        print(f"No face landmarks detected in image {idx}")
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"Out of memory error at index {batch_start}. Stopping.")
                    return batch_start
                else:
                    raise e
    print(f"Face detected: {face_detected}")
    print(f"Face not detected: {face_not_detected}")
    return None


def compute_result_file(rfn):
    rf = loadmat(rfn)
    res = {}
    for r in ['lab', 'msk', 'score', 'pred', 'mask', 'img']:
        res[r] = rf[r].squeeze()
    return res


def main():
    start_index = 0
    TOTAL_RESULTS = {}
    RESDIR = '/shared/rc/defake/Deepfake-Slayer/models_binary/test/xcp_reg/'
    RESFILENAMES = glob.glob(RESDIR + '*.mat')
    for rfn in RESFILENAMES:
        rf = compute_result_file(rfn)
        for r in rf:
            if r not in TOTAL_RESULTS:
                TOTAL_RESULTS[r] = rf[r]
            else:
                TOTAL_RESULTS[r] = np.concatenate([TOTAL_RESULTS[r], rf[r]], axis=0)

    print("Total Results dictionary made")
    print(TOTAL_RESULTS.keys())
    print(f"Resuming processing from index: {start_index}")
    process_images_from_total_results(
        TOTAL_RESULTS, 
        '/shared/rc/defake/Deepfake-Slayer/output/test/output_skin_tone.csv', 
        batch_size=128,
        start_index=start_index
    )

if __name__ == "__main__":
    main()