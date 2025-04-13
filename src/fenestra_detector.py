import cv2
import numpy as np

# 1. Read & convert to grayscale
img = cv2.imread("resources/images/diapsids/cropped_BYU_Utahraptor_skeletal_mount.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Optional: Histogram equalization (enhance contrast)
# gray = cv2.equalizeHist(gray)

# 3. Blur to remove small details (try both!)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# # OR
blur = cv2.medianBlur(gray, 23)
# Show the result
cv2.imshow("blur", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Edge detection
edges = cv2.Canny(blur, 21, 215)
# Show the result
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(edges, kernel, iterations=2)
cv2.imshow("structured edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

max_finestra_area = 20000
min_finestra_area = 250

for c in contours:
    area = cv2.contourArea(c)
    if area < min_finestra_area or area > max_finestra_area:
        continue

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    extent = area / (w * h)
    hull = cv2.convexHull(c)
    solidity = area / cv2.contourArea(hull)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0

    if 0.4 < aspect_ratio < 2.1 and 0.1 < circularity < 5.5 and solidity > 0.1:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

# Show the result
cv2.imshow("Finestra", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
