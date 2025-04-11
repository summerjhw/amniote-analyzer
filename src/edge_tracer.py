import cv2

# Read an image
synapsid_filepath = "resources/images/synapsids/2_cropped_Therapsida_3.png"
diapsid_filepath = "resources/images/diapsids/512px-Skull_diapsida_1.svg.png"
img = cv2.imread(synapsid_filepath, cv2.IMREAD_GRAYSCALE)

# Apply edge detection
edges = cv2.Canny(img, threshold1=100, threshold2=200)

# Show the result
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()