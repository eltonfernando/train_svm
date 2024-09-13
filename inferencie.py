import cv2
import numpy as np

# Função para carregar e pré-processar uma nova imagem para inferência
def preprocess_image(image_path: str, resize: tuple = (64, 64)) -> np.array:
    img = cv2.imread(image_path)
    img = cv2.resize(img, resize)
    img = img.flatten()  # Achatar a imagem em um vetor 1D
    img = np.array(img, np.float32).reshape(1, -1)  # Converter para float32 e ajustar para forma (1, n_características)
    return img

# Carregar o modelo SVM salvo
svm = cv2.ml.SVM_load("svm_model.xml")

# Caminho da nova imagem para inferência
new_image_path = 'yes.jpg'

# Pré-processar a imagem
new_image = preprocess_image(new_image_path)

# Fazer a predição
result = svm.predict(new_image)[1]

# Mostrar o resultado da inferência
if result[0][0] == 1:
    print("Classe: YES")
else:
    print("Classe: NOT")
