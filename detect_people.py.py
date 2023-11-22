import cv2
from ultralytics import YOLO
import supervision as sv

# Leer el modelo yolo (modelo estandar)
model = YOLO('yolov8n.pt')

#valor del color del texto del contado de personas 
region_text_color = (37, 255, 225)
#valor del color del area del contado de personas 
region_color = (0, 0, 0)

#Establecer las dimenciones del video 
frame_width, frame_height = [640,360]

# Ruta del video 
video_path = "Walking_las_condes.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

#inicializamos la anotacion de la caja que se mostrara en ecena 
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


size = (frame_width, frame_height) 
   

    
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, size)

def set_text(frame_width,frame_height, count):
    """
    Establece la posicion del texto en pantalla
    parametros:
        - frame_width: ancho de la escena
        - frame_height: alto de la escena 
        - count: conteo de los objectos

    """
    text_size, _ = cv2.getTextSize(str(count),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.7,thickness=2)
    text_x = frame_width // 2
    text_y = frame_height  // 2
    #dibuja del retangulo
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
                          region_color, -1)
    #dibuja el texto
    cv2.putText(frame, str(count), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
                        2)
    
# Loop mientras el video este ejecutandose 
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        
        # Corre YOLOv8 Detecta los frame
        results = model(frame,classes=0)[0]
        COUNT=0
        labels = []
        detections = sv.Detections.from_ultralytics(results)
        #se obtiene el ID de las clases ( es decir donde aparece un objecto. se obtiene los ids)
        for class_id in detections.class_id:
            #se cuenta la cantidad de objeto en la escena 
            COUNT = COUNT + 1
            #labels.append(f'{COUNT} - {results.names[class_id]}')
            labels.append(f'{COUNT}')

        #annotated_frame = results[0].plot()
   
        annotated_image = bounding_box_annotator.annotate(
        scene=frame,detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )
        set_text(frame_width,frame_height, COUNT)
        # muestra el video con la anotación 
        out.write(annotated_image)
        cv2.imshow("YOLOv8", annotated_image)

        if (cv2.waitKey(30)==27):
                break
    
out.release()