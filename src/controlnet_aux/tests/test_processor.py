"""Test the Processor class."""
import unittest
from PIL import Image

from controlnet_aux.processor import Processor


class TestProcessor(unittest.TestCase):
    def test_hed(self):
        processor = Processor('hed')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_midas(self):
        processor = Processor('midas')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_mlsd(self):
        processor = Processor('mlsd')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_openpose(self):
        processor = Processor('openpose')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_pidinet(self):
        processor = Processor('pidinet')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_normalbae(self):
        processor = Processor('normalbae')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_lineart(self):
        processor = Processor('lineart')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_lineart_coarse(self):
        processor = Processor('lineart_coarse')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_lineart_anime(self):
        processor = Processor('lineart_anime')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_canny(self):
        processor = Processor('canny')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_content_shuffle(self):
        processor = Processor('content_shuffle')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_zoe(self):
        processor = Processor('zoe')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_mediapipe_face(self):
        processor = Processor('mediapipe_face')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)

    def test_tile(self):
        processor = Processor('tile')
        image = Image.open('test_image.png')
        processed_image = processor(image)
        self.assertIsInstance(processed_image, bytes)


if __name__ == '__main__':
    unittest.main()