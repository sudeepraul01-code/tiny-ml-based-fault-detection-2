import unittest

import numpy as np
import tensorflow as tf

from models.network import build_classifier
from quantize.convert import convert_to_int8
from quantize.export_header import render_c_header


class ModelQuantizationTests(unittest.TestCase):
    def test_train_and_convert_small_classifier_to_full_int8(self):
        tf.keras.utils.set_random_seed(42)
        model = build_classifier()
        samples = np.random.default_rng(42).normal(size=(20, 18, 1)).astype(np.float32)
        labels = np.tile(np.arange(5), 4)
        model.fit(samples, labels, epochs=1, batch_size=5, verbose=0)
        flatbuffer = convert_to_int8(model, samples)
        interpreter = tf.lite.Interpreter(model_content=flatbuffer)
        interpreter.allocate_tensors()
        self.assertEqual(interpreter.get_input_details()[0]["dtype"], np.int8)
        self.assertEqual(interpreter.get_output_details()[0]["dtype"], np.int8)
        header = render_c_header(flatbuffer)
        self.assertIn("fault_model_int8_len", header)
        self.assertIn("alignas(16)", header)


if __name__ == "__main__":
    unittest.main()
