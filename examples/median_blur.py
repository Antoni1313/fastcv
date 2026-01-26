import cv2
import torch
import fastcv

img = cv2.imread("artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
blurred_tensor = fastcv.median_blur(img_tensor, 3)

blurred_image = blurred_tensor.cpu().numpy()
cv2.imwrite("output_blur.jpg", blurred_image)

print("saved blurred image.")