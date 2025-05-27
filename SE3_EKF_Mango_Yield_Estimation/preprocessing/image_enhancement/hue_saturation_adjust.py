import cv2
import numpy as np

class HueSaturationAdjust:
    def __init__(self, hue_shift_range=(-10, 10), saturation_scale_range=(-5, 5)):
        """
        Initializes the HueSaturationAdjust processor.
        The paper's ranges for hue (-10 to +10) and saturation (-5 to +5) need careful interpretation.
        Hue in OpenCV (uint8) is 0-179. A shift of -10 to +10 is direct.
        Saturation in OpenCV (uint8) is 0-255. A "scale" of -5 to +5 is unusual.
        It might mean adding/subtracting this value, or scaling by (1 + value/100).
        Let's assume an additive shift for saturation for now, clipped to 0-255.
        Or, it might be a percentage change if the config values are treated as percentages.
        The paper says "saturation was modified between -5 and +5 to control color intensity".
        This sounds like a small absolute change to the S channel value.

        Args:
            hue_shift_range (tuple): Min and max hue shift (degrees, will be scaled for OpenCV).
                                     OpenCV Hue is 0-179 for U8.
            saturation_scale_range (tuple): Min and max adjustment for saturation.
                                         Assuming an absolute shift.
        """
        self.hue_min_shift_deg = hue_shift_range[0]
        self.hue_max_shift_deg = hue_shift_range[1]
        
        # For saturation, if values are -5 to +5, it's likely an absolute adjustment
        # on the 0-255 scale after converting from float.
        # If it was a scale factor like 0.95 to 1.05, the range would be different.
        self.sat_min_adjust = saturation_scale_range[0]
        self.sat_max_adjust = saturation_scale_range[1]

    def apply(self, image, hue_shift=None, saturation_adjust=None):
        """
        Applies hue and saturation adjustment to an image.
        Adjustments are applied randomly within the defined ranges if not specified.

        Args:
            image (np.array): Input image (BGR format).
            hue_shift (int, optional): Specific hue shift to apply (scaled for OpenCV 0-179).
                                       If None, a random shift from the range is chosen.
            saturation_adjust (int, optional): Specific saturation adjustment.
                                            If None, a random adjustment from the range is chosen.

        Returns:
            np.array: Adjusted image.
        """
        if image is None:
            print("Warning: HueSaturationAdjust received a None image.")
            return None
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image.")

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # Apply Hue Shift
        if hue_shift is None:
            # Random hue shift within range
            # Convert degrees to OpenCV's 0-179 range for U8
            actual_hue_shift_deg = np.random.uniform(self.hue_min_shift_deg, self.hue_max_shift_deg)
            # OpenCV hue for U8 is 0-179. Shift needs to be scaled. 1 degree ~ 0.5 in OpenCV U8.
            hue_shift_opencv = int(actual_hue_shift_deg * 0.5)
        else:
            hue_shift_opencv = int(hue_shift)


        h_shifted = cv2.add(h, hue_shift_opencv)
        # Handle wraparound for hue (0-179 for CV_8U)
        h_shifted[h_shifted < 0] += 180
        h_shifted[h_shifted > 179] -= 180
        h_new = h_shifted.astype(h.dtype)


        # Apply Saturation Adjustment
        if saturation_adjust is None:
            # Random saturation adjustment within range
            actual_sat_adjust = np.random.uniform(self.sat_min_adjust, self.sat_max_adjust)
        else:
            actual_sat_adjust = saturation_adjust
        
        s_adjusted = cv2.add(s, int(actual_sat_adjust))
        s_new = np.clip(s_adjusted, 0, 255).astype(s.dtype)


        final_hsv = cv2.merge([h_new, s_new, v])
        adjusted_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return adjusted_image

if __name__ == '__main__':
    # Example Usage
    dummy_image_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image_bgr[:, :, 0] = 255  # Blue channel
    dummy_image_bgr[:, :, 1] = 100  # Green channel
    dummy_image_bgr[:, :, 2] = 50   # Red channel

    # Parameters from paper's config (preprocessing.yaml)
    # hue_shift_range: [-10, 10] # degrees
    # saturation_scale_range: [-5, 5] # absolute adjustment
    
    # Create an instance with default ranges from config
    hs_adjuster = HueSaturationAdjust(hue_shift_range=(-10, 10), saturation_scale_range=(-5, 5))

    # Apply with random adjustment within ranges
    adjusted_image_random = hs_adjuster.apply(dummy_image_bgr.copy())
    print("Applied random hue/saturation adjustment.")

    # Apply with specific adjustments
    # E.g., shift hue by +5 degrees (OpenCV scale: 5 * 0.5 = 2 or 3)
    # E.g., increase saturation by +3
    specific_hue_shift_opencv = int(5 * 0.5) # 5 degrees
    specific_sat_adjust = 3
    adjusted_image_specific = hs_adjuster.apply(dummy_image_bgr.copy(),
                                                hue_shift=specific_hue_shift_opencv,
                                                saturation_adjust=specific_sat_adjust)
    print(f"Applied specific hue shift (OpenCV val: {specific_hue_shift_opencv}) and saturation adjust ({specific_sat_adjust}).")


    # Display images (optional, requires GUI)
    # cv2.imshow("Original BGR", dummy_image_bgr)
    # cv2.imshow("Adjusted Random HS", adjusted_image_random)
    # cv2.imshow("Adjusted Specific HS", adjusted_image_specific)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Test with a real image if available
    # test_image_path = "path_to_your_test_image.jpg"
    # try:
    #     test_image = cv2.imread(test_image_path)
    #     if test_image is not None:
    #         adjusted_test_image = hs_adjuster.apply(test_image)
    #         cv2.imshow("Original Test", test_image)
    #         cv2.imshow("Adjusted Test HS", adjusted_test_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     else:
    #         print(f"Could not read test image from {test_image_path}")
    # except Exception as e:
    #     print(f"Error processing test image: {e}")