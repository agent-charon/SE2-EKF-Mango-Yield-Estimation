import cv2
import numpy as np

class GammaCorrection:
    def __init__(self, gamma=1.0):
        """
        Initializes the GammaCorrection processor.

        Args:
            gamma (float): The gamma value for correction.
                           gamma < 1 will make the image darker.
                           gamma > 1 will make the image brighter.
                           gamma = 1 will have no effect.
        """
        if gamma <= 0:
            raise ValueError("Gamma value must be positive.")
        self.gamma = gamma
        # Build a lookup table mapping pixel values [0, 255] to
        # their adjusted gamma values.
        self.inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** self.inv_gamma) * 255
                               for i in np.arange(0, 256)]).astype("uint8")

    def apply(self, image):
        """
        Applies gamma correction to an image.

        Args:
            image (np.array): Input image (BGR format).

        Returns:
            np.array: Gamma corrected image.
        """
        if image is None:
            print("Warning: GammaCorrection received a None image.")
            return None
        if image.ndim not in [2, 3]: # Grayscale or Color
            raise ValueError("Image must be 2D (grayscale) or 3D (color).")

        return cv2.LUT(image, self.table)

if __name__ == '__main__':
    # Example Usage
    # Create a dummy image
    dummy_image_dark = np.full((100, 100, 3), 50, dtype=np.uint8)
    dummy_image_bright = np.full((100, 100, 3), 200, dtype=np.uint8)

    # Gamma value from paper's config (preprocessing.yaml)
    gamma_value = 1.5 # Makes image brighter

    gamma_corrector = GammaCorrection(gamma=gamma_value)

    corrected_dark_image = gamma_corrector.apply(dummy_image_dark.copy())
    corrected_bright_image = gamma_corrector.apply(dummy_image_bright.copy())

    print(f"Original dark image sample pixel: {dummy_image_dark[0,0]}")
    print(f"Corrected dark image sample pixel (gamma={gamma_value}): {corrected_dark_image[0,0]}")

    print(f"Original bright image sample pixel: {dummy_image_bright[0,0]}")
    print(f"Corrected bright image sample pixel (gamma={gamma_value}): {corrected_bright_image[0,0]}")

    # Display images (optional, requires GUI)
    # cv2.imshow("Original Dark", dummy_image_dark)
    # cv2.imshow("Corrected Dark", corrected_dark_image)
    # cv2.imshow("Original Bright", dummy_image_bright)
    # cv2.imshow("Corrected Bright", corrected_bright_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Test with a real image if available
    # test_image_path = "path_to_your_test_image.jpg"
    # try:
    #     test_image = cv2.imread(test_image_path)
    #     if test_image is not None:
    #         corrected_test_image = gamma_corrector.apply(test_image)
    #         cv2.imshow("Original Test", test_image)
    #         cv2.imshow("Corrected Test", corrected_test_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     else:
    #         print(f"Could not read test image from {test_image_path}")
    # except Exception as e:
    #     print(f"Error processing test image: {e}")