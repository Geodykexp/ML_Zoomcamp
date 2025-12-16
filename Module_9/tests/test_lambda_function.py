import sys
import types
from io import BytesIO
from unittest.mock import patch
import importlib
import numpy as np
import pytest

# Provide a fake onnxruntime to allow importing the module even if the real package isn't installed
fake_ort = types.ModuleType("onnxruntime")

class FakeSession:
    def __init__(self, model_path):
        self.model_path = model_path
        self.last_inputs = None
    def run(self, outputs, inputs):
        self.last_inputs = inputs
        # default fake output
        return [[[0.5]]]

fake_ort.InferenceSession = FakeSession
sys.modules.setdefault("onnxruntime", fake_ort)

# Now import the module under test
import Lambda_function as lf
from PIL import Image


def test_prepare_image_converts_to_rgb_and_normalizes():
    # Create a non-RGB (grayscale) image to trigger conversion
    img = Image.new("L", (10, 20), color=128)
    arr = lf.prepare_image(img, target_size=(8, 6))

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 3, 6, 8)
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()
    # Values should be in [-1, 1] after normalization
    assert arr.min() >= -1.0 - 1e-6
    assert arr.max() <= 1.0 + 1e-6


def test_prepare_image_resizes_with_nearest_and_shapes_correctly():
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    arr = lf.prepare_image(img, target_size=(2, 3))
    assert arr.shape == (1, 3, 3, 2)  # (1, C, H, W)


def test_download_image_reads_via_urlopen_and_returns_image():
    # Create a small PNG in-memory
    buf = BytesIO()
    Image.new("RGB", (3, 4), color=(10, 20, 30)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    class FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return png_bytes

    with patch.object(lf.request, "urlopen", return_value=FakeResp()):
        img = lf.download_image("http://example.com/image.png")

    assert isinstance(img, Image.Image)
    assert img.size == (3, 4)


def test_predict_returns_float_and_uses_provided_model_path():
    # Patch download_image to avoid network
    test_img = Image.new("RGB", (5, 5), color=(0, 128, 255))

    class CustomSession(FakeSession):
        def run(self, outputs, inputs):
            self.last_inputs = inputs
            return [[[0.8]]]

    seen_paths = []
    def make_session(path):
        seen_paths.append(path)
        return CustomSession(path)

    with patch.object(lf, "download_image", return_value=test_img), \
         patch.object(lf.ort, "InferenceSession", side_effect=make_session):
        pred = lf.predict("http://any", model_path="custom.onnx")

    assert isinstance(pred, float)
    assert pred == pytest.approx(0.8)
    assert seen_paths == ["custom.onnx"]


def test_predict_passes_correct_input_tensor_to_session():
    # Use a deterministic image so the shape after prepare_image is known
    test_img = Image.new("RGB", (4, 6), color=(1, 2, 3))

    class CapturingSession(FakeSession):
        def run(self, outputs, inputs):
            self.last_inputs = inputs
            # return any valid numeric output
            return [[[0.3]]]

    captured = {}
    def make_session(path):
        s = CapturingSession(path)
        captured["session"] = s
        return s

    with patch.object(lf, "download_image", return_value=test_img), \
         patch.object(lf.ort, "InferenceSession", side_effect=make_session):
        _ = lf.predict("http://any")

    sess = captured["session"]
    assert isinstance(sess.last_inputs, dict)
    assert "input" in sess.last_inputs
    input_arr = sess.last_inputs["input"]
    # prepare_image resizes to (200, 200) then to (1, 3, H, W)
    assert isinstance(input_arr, np.ndarray)
    assert input_arr.shape == (1, 3, 200, 200)


def test_lambda_handler_returns_error_when_url_missing():
    result = lf.lambda_handler({}, None)
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"].lower().startswith("no image url")


def test_lambda_handler_success_path_returns_prediction():
    with patch.object(lf, "predict", return_value=0.42):
        result = lf.lambda_handler({"url": "http://example.com"}, None)
    assert result == {"prediction": 0.42}


def test_lambda_handler_catches_exceptions_and_returns_error():
    with patch.object(lf, "predict", side_effect=Exception("boom")):
        result = lf.lambda_handler({"url": "http://example.com"}, None)
    assert isinstance(result, dict)
    assert result.get("error") == "boom"
