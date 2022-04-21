# Importamos las librerias
import cv2
import mediapipe as mp

# Declaramos la deteccion de rostros
detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence= 0.5, model_selection=0)
# Dibujo
dibujorostro = mp.solutions.drawing_utils

# Declaramos la deteccion de manos
detman = mp.solutions.hands
manos = detman.Hands(static_image_mode = False, max_num_hands= 2, min_detection_confidence= 0.5, min_tracking_confidence= 0.5)
# Dibujo
dibujomanos = mp.solutions.drawing_utils

# Declaramos la deteccion de malla facial
detmal =mp.solutions.face_mesh
malla =detmal.FaceMesh(max_num_faces= 1, min_detection_confidence= 0.5)
# Dibujo
dibmalla = mp.solutions.drawing_utils
dibujomalla =dibmalla.DrawingSpec(thickness= 1, circle_radius=1)

# Realizamos VideoCaptura
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3,1280)
cap.set(4,720)
marca = 0

# Funcion de eventos de mouse
def mouse(evento, xm, ym, bandera, param):
    global xmo, ymo, marca

    # Evaluamos los eventos
    # Click izquierdo
    if evento == cv2.EVENT_LBUTTONDOWN:
        # Coordenadas del click
        xmo = xm
        ymo = ym
        print(xmo, ymo)
        marca = 1

    # Click derecho
    if evento == cv2.EVENT_RBUTTONDOWN:
        # Coordenadas del click
        xmo = xm
        ymo = ym
        print(xmo, ymo)
        marca = 2

    # Click central
    if evento == cv2.EVENT_MBUTTONDOWN:
        # Coordenadas
        xmo = xm
        ymo = ym
        print(xmo, ymo)
        marca = 3

    # cv2.EVENT_LBUTTONUP -> Se suelta click izquierdo
    # cv2.EVENT_RBUTTONUP -> Se suelta click derecho
    # cv2.EVENT_MBUTTONUP -> Se suelta click ruedita
    # cv2.EVENT_LBUTTONDBLCLK -> Doble click izquierdo
    # cv2.EVENT_RBUTTONDBLCLK -> Doble click derecho
    # cv2.EVENT_MBUTTONDBLCLK -> Doble click ruedita

# Procesamiento en tiempo real
while True:
    # Leemos frames
    ret, frame = cap.read()

    # Conversion de color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Leemos el click
    cv2.namedWindow('Camara IA')
    cv2.setMouseCallback('Camara IA', mouse)

    # Si es click izquierdo
    if marca == 1:

        # Procesamos la deteccion de rostros
        resrostros = rostros.process(rgb)

        # Si hay rostros
        if resrostros.detections is not None:
            # Registramos
            for rostro in resrostros.detections:
                # Dibujamos rostro
                dibujorostro.draw_detection(frame, rostro, dibujorostro.DrawingSpec(color=(0, 255, 0)))

                # Dibujamos click
                cxr = xmo
                cyr = ymo
                cv2.circle(frame, (cxr, cyr), 10, (0, 255, 0), 2)

    # Si es click derecho
    elif marca == 2:

        # Procesamos los frames para malla
        resmalla = malla.process(rgb)

        # Preguntamos si hay resultados
        if resmalla.multi_face_landmarks:
            for mesh in resmalla.multi_face_landmarks:
                # Dibujamos malla
                dibmalla.draw_landmarks(frame, mesh, detmal.FACEMESH_TESSELATION, dibujomalla, dibujomalla)
                # Dibujamos click
                cxm = xmo
                cym = ymo
                cv2.circle(frame, (cxm, cym), 10, (255, 0, 0), 2)

    # Si es click central
    elif marca == 3:

        # Procesamos los frames para
        resmanos = manos.process(rgb)

        # Preguntamos si hay resultados
        if resmanos.multi_hand_landmarks:
            for mano in resmanos.multi_hand_landmarks:
                # Dibujamos manos
                dibujomanos.draw_landmarks(frame, mano, detman.HAND_CONNECTIONS)

                # Dibujamos click
                cxh = xmo
                cyh = ymo
                cv2.circle(frame, (cxh, cyh), 10, (0, 0, 255), 2)

    # Mostramos los fotogramas
    cv2.imshow("Camara IA", frame)

    # Condicion para romper el while
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
