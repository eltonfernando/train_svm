from glob import glob
import cv2
import numpy as np

def load_images_from_folder(folder:str, resize:tuple=(64,64)) -> list:
    images = []
    for filename in glob(folder):
        img = cv2.imread(filename)
        img = cv2.resize(img, resize)
        if img is not None:
            images.append(img.flatten())
    return images

load_yes = load_images_from_folder('img/yes/*.jpg')
label_yes = [1 for i in range(len(load_yes))]
print(f"class 1: yes {len(load_yes)}=={label_yes.count(1)}")

load_not = load_images_from_folder('img/not/*.jpg')
label_not = [0 for i in range(len(load_not))]
print(f"class 0: not {len(load_not)}=={label_not.count(0)}")

label = label_yes + label_not
train = load_yes + load_not

print(len(train), len(label))
train = np.array(train,np.float32)
label = np.array(label,np.int32)

print(f"Tamanho do dataset de treino: {train.shape}")
print(f"Tamanho dos rótulos: {label.shape}")

svm = cv2.ml.SVM()
svm =svm.create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setGamma(1.000083)
svm.setC(1.00)
svm.train(train,cv2.ml.ROW_SAMPLE, label)

svm.save("svm_model.xml")
print("Modelo SVM treinado e salvo com sucesso!")