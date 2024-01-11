import cv2

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
# fecha a webcam e a tela que mostra a imagem
webcam.release()
cv2.destroyAllWindows()