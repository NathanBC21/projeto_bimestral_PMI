import cv2
import numpy as np
import matplotlib.pyplot as plt

#1-Leitura e Exibição Inicial

#Exibe a imagem original
imagem = cv2.imread("imagem_exame.jpg")
cv2.imshow("imagem", imagem)

#2-Pré processamento

#Converte para a escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("escala_cinza", imagem_cinza)
cv2.imwrite("escala_cinza.png", imagem_cinza)

#Aplicação de equalização de histograma
imagem_equalizada = cv2.equalizeHist(imagem_cinza)
cv2.imshow("imagem equalizada", imagem_equalizada)
cv2.imwrite("imagem_equalizada.png", imagem_equalizada)

#3-Modificação de Cores

#Conversão da imagem original para o espaço de cores HSV
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
cv2.imshow("imagme hsv", imagem_hsv)
cv2.imwrite("imagme_hsv.png", imagem_hsv)

#Aumento do canal S em 30%
H, S, V = cv2.split(imagem_hsv)
S = S * 1.3
S = np.clip(S, 0, 255)
S = S.astype(np.uint8)
canal_s_aumentado = cv2.merge([H, S, V])
cv2.imshow("canal_s_aumentado", canal_s_aumentado)
cv2.imwrite("canal_s_aumentado.png", canal_s_aumentado)

#4-Ajuste de Contraste e Brilho

#Aumento do brilho em 50 unidades
H, S, V = cv2.split(imagem_hsv)
V = V + 50
V = np.clip(V, 0, 255)
V = V.astype(np.uint8)
brilho_aumentado = cv2.merge([H, S, V])
cv2.imshow("brilho_aumentado", brilho_aumentado)
cv2.imwrite("brilho_aumentado.png", brilho_aumentado)

#Ajuste do contraste aplicando uma transformação linear
H, S, V = cv2.split(brilho_aumentado)
alpha = 1.5
V = alpha * V
V = np.clip(V, 0, 255)
V = V.astype(np.uint8)
ajuste_contraste = cv2.merge([H, S, V])
cv2.imshow("ajuste_contraste", ajuste_contraste)
cv2.imwrite("ajuste_contraste.png", ajuste_contraste)

#5-Redimensionamento e Interpolação

#Imagem redimensionada em 50% usando interpolação bicúbica.
altura_original, largura_original = imagem.shape[:2]
nova_largura = int(largura_original * 0.5)
nova_altura = int(altura_original * 0.5)
imagem_menor50 = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)
cv2.imshow("imagem_menor50", imagem_menor50)
cv2.imwrite("imagem_menor50.png", imagem_menor50)

#Imagem redimensionada em 200% usando interpolação linear.
altura_original, largura_original = imagem.shape[:2]
nova_largura = int(largura_original * 2)
nova_altura = int(altura_original * 2)
imagem_maior200 = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_LINEAR)
cv2.imshow("imagem_maior200", imagem_maior200)
cv2.imwrite("imagem_maior200.png", imagem_maior200)

#6-Transformações Geométricas

#Imagem rotacionada em 45º mantendo o tamanho original
altura, largura = imagem.shape[:2]
centro = (largura//2, altura//2)
matriz = cv2.getRotationMatrix2D(centro, 45, 1.0)
imagem_rotacionada = cv2.warpAffine(imagem, matriz, (largura, altura))
cv2.imshow("imagem_rotacionada", imagem_rotacionada)
cv2.imwrite("imagem_rotacionada.png", imagem_rotacionada)

#Imagem espelhada horizontalmente
imagem_espelhada = cv2.flip(imagem, 1)
cv2.imshow("imagem_espelha", imagem_espelhada)
cv2.imwrite("imagem_espelhada.png", imagem_espelhada)

#Retângulo central de 300x300 pixels.
altura, largura =imagem.shape[:2]
meio_x = largura// 2
meio_y = altura// 2
largura_recorte = 300
altura_recorte = 300
x1 = meio_x - largura_recorte // 2
x2 = meio_x + largura_recorte // 2
y1 = meio_y - altura_recorte // 2
y2 = meio_y + altura_recorte // 2
recorte = imagem[y1:y2, x1:x2]
cv2.imshow("recorte_central", recorte)
cv2.imwrite("recorte_central.png", recorte)

#7-Binarização e Salvamento

#Binarização Otsu na imagem em escala de cinza.
_, imagem_otsu = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("otsu", imagem_otsu)
cv2.imwrite("otsu.png", imagem_otsu)


cv2.waitKey(0)
cv2.destroyAllWindows()


