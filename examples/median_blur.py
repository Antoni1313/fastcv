import cv2
import torch
import fastcv

kernel_size = 3
filename = "artifacts/binary.jpg"

img = cv2.imread(filename)
img_tensor = torch.from_numpy(img).cuda()
blurred_tensor = fastcv.median_blur(img_tensor, kernel_size)

blurred_image_fastcv = blurred_tensor.cpu().numpy()
blurred_image_opencv = cv2.medianBlur(img, kernel_size)
cv2.imwrite("output_blur_fastcv.jpg", blurred_image_fastcv)
cv2.imwrite("output_blur_opencv.jpg", blurred_image_opencv)

print("saved blurred image.")

row_id = 0
difference_detected = False
for row_fastcv, row_opencv in zip(blurred_image_fastcv, blurred_image_opencv):
    pixel_id = 0
    for pixel_fastcv, pixel_opencv in zip(row_fastcv, row_opencv):
        difference = abs(int(pixel_fastcv[0]) - int(pixel_opencv[0])) + abs(int(pixel_fastcv[1]) - int(pixel_opencv[1])) + abs(int(pixel_fastcv[2]) - int(pixel_opencv[2]))
        if difference != 0:
            print("Difference at: x - ", pixel_id, ", y - ", row_id)
            difference_detected = True
        pixel_id += 1
    row_id += 1

if difference_detected:
    print("Detected differences")
else:
    print("No differences, HURRAY")