
# %%
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img: Image.Image, target_size=(200, 200)) -> np.ndarray:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    img_array = np.array(img).astype('float32')
    img_array = img_array / 255.0          # scale to [0,1]
    img_array = (img_array - 0.5) / 0.5    # rescale to [-1,1]
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# %%
# --- Run inference ---
def predict(url: str, model_path="hair_classifier_empty.onnx") -> float:
    img = download_image(url)
    img_array = prepare_image(img)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    return float(np.squeeze(outputs[0]))

# %%
# --- Lambda handler ---
def lambda_handler(event, context):
    url = event.get("url")
    if not url:
        return {"error": "No image URL provided"}
    try:
        prediction = predict(url)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


