import unittest
import cv2
import numpy as np
from joblib import load
from recognize_n import preprocess_image, segment_digits, recognize_digits

class TestDigitRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cargar el clasificador SVM y el escalador desde archivos preentrenados
        cls.clf = load('svm_digit_classifier.joblib')
        cls.scaler = load('scaler.joblib')

    def test_preprocess_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        preprocessed_image = preprocess_image(image)
        self.assertEqual(preprocessed_image.shape, (100, 100), "Preprocessed image shape is incorrect")
        # El valor debe ser 0 porque la imagen es completamente negra
        self.assertTrue(np.all(preprocessed_image == 0), "Preprocessed image is not binary")

    def test_segment_digits(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(image, '4', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        preprocessed_image = preprocess_image(image)
        digits, positions = segment_digits(preprocessed_image)
        self.assertGreater(len(digits), 0, "Digit segmentation failed")
        for digit in digits:
            self.assertEqual(digit.shape, (1, 64), "Segmented digit is not 8x8 pixels flattened")
        self.assertEqual(len(digits), len(positions), "Number of digits and positions do not match")

    def test_recognize_digits(self):
        digit = np.zeros((8, 8), dtype=np.uint8)
        cv2.putText(digit, '4', (1, 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        digit = digit.flatten().reshape(1, -1).astype('float32') / 255
        digits = [digit]
        predictions = recognize_digits(digits, self.scaler)
        self.assertEqual(len(predictions), len(digits), "Digit recognition failed")
        self.assertIsInstance(predictions[0], np.integer, "Prediction is not an integer")

if __name__ == '__main__':
    unittest.main()
