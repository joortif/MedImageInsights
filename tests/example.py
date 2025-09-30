import base64

import numpy as np

# Define basic methods
def read_image_bytes(path: str) -> bytes:
  with open(path, 'rb') as f:
    return f.read()

def image_to_base64(path: str) -> str:
  return base64.encodebytes(read_image_bytes(path)).decode('utf-8')

# Initialize classifier and load the model
from MedImageInsights.Models.medimageinsightmodel import MedImageInsight

classifier = MedImageInsight()
classifier.load_model()

# Define image paths
phneumonia = "images/phneumonia.png"
head_ct = "images/head_ct.jpg"
breast_ultrasound = "images/breast_ultrasound.png"
melanoma = "images/melanoma.jpg"

# Example zero-shot classification
labels = ['pneumonia','head CT', 'dermatology', 'breast', 'ultrasound']
image_b64 = image_to_base64(phneumonia)
images = [image_b64] 

print('Running predict (zero-shot) with labels:', labels)
zs_results = classifier.predict(images, labels)
print('Predict result:')
print(zs_results)

# Example image embeddings
img_emb = classifier.encode(images=images)['image_embeddings']
print('Image embedding obtained. Type/shape:', type(img_emb), np.array(img_emb).shape)
print('Image embedding:', img_emb)

# Example text embeddings
txt_emb = classifier.encode(texts=labels)['text_embeddings']
print('Text embeddings obtained. Type/shape:', type(txt_emb), np.array(txt_emb).shape)
print('Text embedding:', txt_emb)