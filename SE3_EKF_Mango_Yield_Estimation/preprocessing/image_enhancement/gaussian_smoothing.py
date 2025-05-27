import cv2
import numpy as np

class GaussianSmoothing:
    def __init__(self, kernel_size=(5, 5), sigma_x=1.0, sigma_y=None):
        """
        Initializes the GaussianSmoothing processor.

        Args:
            kernel_size (tuple): Gaussian kernel size (width, height). Must be odd and positive.
            sigma_x (float): Gaussian kernel standard deviation in X direction.
            sigma_y (float, optional): Gaussian kernel standard deviation in Y direction.
                                       If None, sigma_y is taken as equal to sigma_x.
        """
        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2 and
                kernel_size[0] > 0 and kernel_size[0] % 2 == 1 and
                kernel_size[1] > 0 and kernel_size[1] % 2 == 1):
            raise ValueError("Kernel size must be a tuple of two positive odd integers.")
        if sigma_x <= 0:
            raise ValueError("sigma_x must be positive.")
        if sigma_y is not None and sigma_y <= 0:
            raise ValueError("sigma_y must be positive if specified.")

        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y if sigma_y is not None else sigma_x

    def apply(self, image):
        """
        Applies Gaussian smoothing to an image.

        Args:
            image (np.array): Input image (BGR or grayscale).

        Returns:
            np.array: Smoothed image.
        """
        if image is None:
            print("Warning: GaussianSmoothing received a None image.")
            return None
        if image.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color).")

        return cv2.GaussianBlur(image, self.kernel_size, self.sigma_x, sigmaY=self.sigma_y)

if __name__ == '__main__':
    # Example Usage
    # Create a dummy noisy image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.randu(dummy_image, 0, 255) # Add random noise

    # Parameters from paper's config (preprocessing.yaml)
    kernel_w, kernel_h = 5, 5
    sigma_val = 1.0

    smoother = GaussianSmoothing(kernel_size=(kernel_w, kernel_h), sigma_x=sigma_val)
    smoothed_image = smoother.apply(dummy_image.copy())

    print(f"Applied Gaussian smoothing with kernel=({kernel_w},{kernel_h}) and sigma={sigma_val}")

    # Display images (optional, requires GUI)
    # cv2.imshow("Original Noisy", dummy_image)
    # cv2.imshow("Smoothed Image", smoothed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Test with a real image if available
    # test_image_path = "path_to_your_test_image.jpg"
    # try:
    #     test_image = cv2.imread(test_image_path)
    #     if test_image is not None:
    #         smoothed_test_image = smoother.apply(test_image)
    #         cv2.imshow("Original Test", test_image)
    #         cv2.imshow("Smoothed Test", smoothed_test_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     else:
    #         print(f"Could not read test image from {test_image_path}")
    # except Exception as e:
    #     print(f"Error processing test image: {e}")