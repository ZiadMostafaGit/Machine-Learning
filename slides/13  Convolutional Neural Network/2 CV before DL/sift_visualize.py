import cv2
import matplotlib.pyplot as plt

# Read the image
image_path = '/home/moustafa/Downloads/cat.jpg'  # replace with your image path
image = cv2.imread(image_path)

# Check if image is loaded properly
if image is None:
    print("Error: Could not read image.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features, also known as keypoints
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Draw keypoints on the image
    keypoint_image = cv2.drawKeypoints(image, keypoints,
                                       None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Convert image color from BGR to RGB
    keypoint_image = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)

    # Show the image with keypoints
    plt.imshow(keypoint_image)
    plt.title('SIFT Keypoints')
    plt.axis('off')  # Hide axis
    plt.show()
