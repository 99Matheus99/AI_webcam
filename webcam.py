# o modelo usado é o Random Forest Classifier (não aparenta ser bom)
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
webcam = cv2.VideoCapture(0) # abre a webcam

if webcam.isOpened():
    val, frame = webcam.read() # caso a webcam esteja aberta, ele guarda o frame e uma validação
    while val: # aqui crio um loop que irá sempre mostrar a imagem a cada 5ms
        val, frame = webcam.read()
        cv2.imshow('video da webcam', frame)
        key = cv2.waitKey(5)
        if key == 27: # botão ESC
            break
    cv2.imwrite('imagem_mao.png', frame) # "tira um print" da webcam


# redimensionando a imagem
print('Redimensionando a imagem...')
imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
imagem = cv2.resize(imagem, (28,28))

cv2.imwrite('imagem_mao_2.png', imagem) # "tira um print" da webcam

imagem = np.array(imagem).flatten()

# fecha a webcam e a tela que mostra a imagem
webcam.release()
cv2.destroyAllWindows()

# abrindo o banco de dados
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target.astype(int)
X = X.astype(float) / 255.0  # Normalizar os valores de pixel, faz comparação grotesca

# dividir em treino e teste
X_teste, X_treino, y_teste, y_treino = train_test_split(X, y, test_size=0.2, random_state=42)

# iniciando e treinando o modelo
print('Treinando o modelo...')
# model = LogisticRegression(solver='lbfgs', max_iter=500)
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_treino, y_treino)
print('INICIANDO PREVISAO...')

# adiciono nome às colunas
colunas = [f'pixel{i}' for i in range(1, 785)] 
df = pd.DataFrame(columns=colunas)
df.loc[0] = imagem

# fazendo a previsão
previ = model.predict(df)
print(f'Dígito previsto: {previ[0]}')
print('ACABOU')
