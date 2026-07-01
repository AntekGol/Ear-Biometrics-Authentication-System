import cv2
import numpy as np
import tensorflow as tf
from keras.applications.resnet import preprocess_input

MODEL_PATH = 'ear_authentication_model2.h5'
TARGET_SIZE = (224, 224)


# Layer normalizing the embedding vector to length 1 (L2 norm)
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)


def resize_with_padding(image, target_size=TARGET_SIZE):

    #Scale image to model size maintaining aspect ratio with black padding

    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded


def preprocess_for_embedding(image_bgr):

    #Prepare BGR image for model input: color conversion, resize + padding, scaling and normalization

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = resize_with_padding(image_rgb)
    image_float = image_resized.astype(np.float32)
    image_pre = preprocess_input(image_float)
    image_batch = np.expand_dims(image_pre, axis=0)
    return image_batch


def generate_embedding(image_bgr):

    #Generate embedding using the trained model. Returns: 1D vector

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'L2Normalization': L2Normalization},
        compile=False
    )
    preprocessed = preprocess_for_embedding(image_bgr)
    embedding = model.predict(preprocessed, verbose=0)
    return embedding[0]


def compare_embeddings(emb1, emb2, threshold=0.70):

    #Compare two embeddings using Euclidean distance
    distance = np.linalg.norm(emb1 - emb2)
    return (distance <= threshold), distance
