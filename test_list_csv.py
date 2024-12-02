from mediapipe import solutions
import cv2
import numpy as np
import pickle
import csv
from PIL import Image

def get_region_mask(image, landmarks, region_points):
    height, width, _ = image.shape
    points = [(int(landmarks[pt].x * width), int(landmarks[pt].y * height)) for pt in region_points]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
    return mask

def compute_color_features(image, mask):
# Apply the mask
    region = image * mask[..., np.newaxis]

    # Compute RGB mean
    rgb_mean = region[mask == 1].mean(axis=0)

    # Convert to LAB and compute mean
    lab_image = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
    lab_mean = lab_image[mask == 1].mean(axis=0)

    # Convert to YCbCr and compute mean
    ycbcr_image = cv2.cvtColor(region, cv2.COLOR_RGB2YCrCb)
    ycbcr_mean = ycbcr_image[mask == 1].mean(axis=0)

    return lab_mean, rgb_mean, ycbcr_mean

def apply_clahe(denormalized_image):
    # Scale back to [0, 255] for CLAHE (denormalized_image is assumed to be in [0, 1])
    # scaled_image = (denormalized_image * 255).astype(np.uint8)

    # Convert RGB to LAB color space
    lab_image = cv2.cvtColor(denormalized_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    # Merge the CLAHE L-channel back with A and B channels
    lab_clahe_image = cv2.merge((l_clahe, a_channel, b_channel))

    # Convert back to RGB color space
    normalized_image = cv2.cvtColor(lab_clahe_image, cv2.COLOR_LAB2RGB)

    return normalized_image

def process_images_from_test_list(test_list, output_csv, start_index=0):
  # Initialize the Face Mesh solution
  mp_face_mesh = solutions.face_mesh.FaceMesh(
      static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.2
  )

  face_detected = 0
  face_not_detected = 0

  # Iterate through test_list
  with open(output_csv, 'a', newline='') as csvfile:
    fieldnames = ["Image_Index", "Avg_Lab_L", "Avg_Lab_a", "Avg_Lab_b", "Avg_RGB_R", "Avg_RGB_G", "Avg_RGB_B", "Avg_YCbCr_Y", "Avg_YCbCr_Cb", "Avg_YCbCr_Cr"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if start_index == 0:
      writer.writeheader()  # Write the header once
    for idx, (image_path, mask_path) in enumerate(test_list, start=start_index):
      print(f"Processing image {idx}: {image_path}")
      try:
        # Read the image from the file path
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            print(f"Image at {image_path} could not be loaded.")
            continue

        # Convert to RGB as Mediapipe requires RGB images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(10, 10))
        # plt.subplot(1, 3, 1)
        # plt.title('Original Image')
        # plt.imshow(image)

        # Apply CLAHE or any other preprocessing
        image = apply_clahe(image)

        # plt.subplot(1, 3, 2)
        # plt.title('Clahe Image')
        # plt.imshow(image)
        # plt.show()

        #Process with Mediapipe Face Mesh
        results = mp_face_mesh.process(image)

        if results.multi_face_landmarks:
          face_detected += 1
          landmarks = results.multi_face_landmarks[0].landmark

          # Define masks for different facial regions
          left_cheek_mask = get_region_mask(image, landmarks, [50, 118, 101])
          right_cheek_mask = get_region_mask(image, landmarks, [330, 280, 347])
          nose_tip_mask = get_region_mask(image, landmarks, [5, 4, 51])
          forehead_mask = get_region_mask(image, landmarks, [9, 107, 66, 105, 63, 68, 54, 103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336])

          # Compute color features for each region
          left_lab, left_rgb, left_ycbcr = compute_color_features(image, left_cheek_mask)
          right_lab, right_rgb, right_ycbcr = compute_color_features(image, right_cheek_mask)
          nose_lab, nose_rgb, nose_ycbcr = compute_color_features(image, nose_tip_mask)
          forehead_lab, forehead_rgb, forehead_ycbcr = compute_color_features(image, forehead_mask)
          
          # plt.subplot(1, 3, 3)
          # plt.imshow(image)
          # plt.title("Highlighted regions in the image")
          # plt.axis("off")

          # # Overlay the left cheek mask
          # plt.imshow(left_cheek_mask, cmap='Reds', alpha=0.3)  # Red color mask with 50% transparency

          # # Overlay the right cheek mask
          # plt.imshow(right_cheek_mask, cmap='Blues', alpha=0.3)  # Blue color mask with 50% transparency

          # # Overlay the nose mask
          # plt.imshow(nose_tip_mask, cmap='Greens', alpha=0.3)

          # plt.imshow(forehead_mask, cmap='Purples', alpha=0.3)

          # plt.tight_layout()
          # plt.show()

          # Calculate average values
          avg_lab_l = (left_lab[0] + right_lab[0] + nose_lab[0] + forehead_lab[0]) / 4
          avg_lab_a = (left_lab[1] + right_lab[1] + nose_lab[1] + forehead_lab[1]) / 4
          avg_lab_b = (left_lab[2] + right_lab[2] + nose_lab[2] + forehead_lab[2]) / 4

          avg_rgb_r = (left_rgb[0] + right_rgb[0] + nose_rgb[0] + forehead_rgb[0]) / 4
          avg_rgb_g = (left_rgb[1] + right_rgb[1] + nose_rgb[1] + forehead_rgb[1]) / 4
          avg_rgb_b = (left_rgb[2] + right_rgb[2] + nose_rgb[2] + forehead_rgb[2]) / 4

          avg_ycbcr_y = (left_ycbcr[0] + right_ycbcr[0] + nose_ycbcr[0] + forehead_ycbcr[0]) / 4
          avg_ycbcr_cb = (left_ycbcr[1] + right_ycbcr[1] + nose_ycbcr[1] + forehead_ycbcr[1]) / 4
          avg_ycbcr_cr = (left_ycbcr[2] + right_ycbcr[2] + nose_ycbcr[2] + forehead_ycbcr[2]) / 4

          # Write data to CSV
          writer.writerow({
              "Image_Index": idx,
              "Avg_Lab_L": avg_lab_l, "Avg_Lab_a": avg_lab_a, "Avg_Lab_b": avg_lab_b,
              "Avg_RGB_R": avg_rgb_r, "Avg_RGB_G": avg_rgb_g, "Avg_RGB_B": avg_rgb_b,
              "Avg_YCbCr_Y": avg_ycbcr_y, "Avg_YCbCr_Cb": avg_ycbcr_cb, "Avg_YCbCr_Cr": avg_ycbcr_cr,
          })
        else:
          face_not_detected += 1
          print(f"No face landmarks detected in image {idx}")

      except RuntimeError as e:
          if 'oom_kill' in str(e).lower():
              print(f"Out of memory error at index {idx}. Stopping.")
              break
          else:
              print(f"Runtime error: {e}")
              continue
      # break
  print(f"Face detected: {face_detected}")
  print(f"Face not detected: {face_not_detected}")

def main():
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceSwap.pkl', 'rb') as file:#'/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceSwap.pkl'
    FaceSwap_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/Face2Face.pkl', 'rb') as file:#'/shared/rc/defake/Deepfake-Slayer/pickel_file/Face2Face.pkl'
    Face2Face_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceShifter.pkl', 'rb') as file:#'/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceShifter.pkl'
    FaceShifter_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/fake_NeuralTextures.pkl', 'rb') as file:#'/shared/rc/defake/Deepfake-Slayer/pickel_file/fake_NeuralTextures.pkl'
    fake_NeuralTextures_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_yt_test.pkl', 'rb') as file:#/shared/rc/defake/Deepfake-Slayer/pickel_file/real_yt_test.pkl
    real_yt_test = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_actors_test.pkl', 'rb') as file:#/shared/rc/defake/Deepfake-Slayer/pickel_file/real_actors_test.pkl
    real_actors_test = pickle.load(file)

  test_list = FaceSwap_mask['test'] + Face2Face_mask['test'] + FaceShifter_mask['test'] + fake_NeuralTextures_mask['test'] + real_yt_test + real_actors_test
  output_csv = '/shared/rc/defake/Deepfake-Slayer/output/test/test_list_color_values.csv'
  print("Processing from test list start")
  process_images_from_test_list(test_list, output_csv, start_index=0)
  print("Processing from test list ended")

if __name__ == "__main__":
    main()