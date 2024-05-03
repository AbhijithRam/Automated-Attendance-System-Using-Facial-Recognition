import face_recognition
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
from keras.models import load_model
import pandas as pd
from datetime import datetime
from datetime import date
model_face_recognition = load_model('model_rgb_frozen.h5')
model_yolo = YOLO("yolov8n-face.pt")
from time import sleep

def face():

    AVB_image = face_recognition.load_image_file("data/AVB/200.jpg")
    AVB_face_encoding = face_recognition.face_encodings(AVB_image)[0]

    Abhijith_image = face_recognition.load_image_file("data/Abhijith/175.jpg")
    Abhijith_face_encoding = face_recognition.face_encodings(Abhijith_image)[0]

    Abhinava_image = face_recognition.load_image_file("data/Abhinava/198.jpg")
    Abhinava_face_encoding = face_recognition.face_encodings(Abhinava_image)[0]

    Abhinavrijit_image = face_recognition.load_image_file("/Users/abhinavrijit/siamese4/Datasets/Test/Abhinavrijit/108.jpg")
    Abhinavrijit_face_encoding = face_recognition.face_encodings(Abhinavrijit_image)[0]

    Adithyaraj_image = face_recognition.load_image_file("data/Adithyaraj/186.jpg")
    Adithyaraj_face_encoding = face_recognition.face_encodings(Adithyaraj_image)[0]

    Adwaith_image = face_recognition.load_image_file("data/Adwaith/184.jpg")
    Adwaith_face_encoding = face_recognition.face_encodings(Adwaith_image)[0]

    Afras_image = face_recognition.load_image_file("data/Afras/183.jpg")
    Afras_face_encoding = face_recognition.face_encodings(Afras_image)[0]

    Aneena_image = face_recognition.load_image_file("data/Aneena/185.jpg")
    Aneena_face_encoding = face_recognition.face_encodings(Aneena_image)[0]

    Anirudh_image = face_recognition.load_image_file("data/Anirudh/191.jpg")
    Anirudh_face_encoding = face_recognition.face_encodings(Anirudh_image)[0]

    Ansal_image = face_recognition.load_image_file("data/Ansal/185.jpg")
    Ansal_face_encoding = face_recognition.face_encodings(Ansal_image)[0]

    Aquin_image = face_recognition.load_image_file("data/Aquin/173.jpg")
    Aquin_face_encoding = face_recognition.face_encodings(Aquin_image)[0]

    Arjunanikuttan_image = face_recognition.load_image_file("data/Arjunanikuttan/173.jpg")
    Arjunanikuttan_face_encoding = face_recognition.face_encodings(Arjunanikuttan_image)[0]

    Arjunv_image = face_recognition.load_image_file("data/Arjunv/173.jpg")
    Arjunv_face_encoding = face_recognition.face_encodings(Arjunv_image)[0]

    Ashfin_image = face_recognition.load_image_file("data/Ashfin/172.jpg")
    Ashfin_face_encoding = face_recognition.face_encodings(Ashfin_image)[0]

    Athira_image = face_recognition.load_image_file("data/Athira/171.jpg")
    Athira_face_encoding = face_recognition.face_encodings(Athira_image)[0]

    Chandana_image = face_recognition.load_image_file("data/Chandana/171.jpg")
    Chandana_face_encoding = face_recognition.face_encodings(Chandana_image)[0]

    Devalakshmy_image = face_recognition.load_image_file("data/Devalakshmy/200.jpg")
    Devalakshmy_face_encoding = face_recognition.face_encodings(Devalakshmy_image)[0]

    Devananda_image = face_recognition.load_image_file("data/Devananda/172.jpg")
    Devananda_face_encoding = face_recognition.face_encodings(Devananda_image)[0]

    Dinasree_image = face_recognition.load_image_file("data/Dinasree/200.jpg")
    Dinasree_face_encoding = face_recognition.face_encodings(Dinasree_image)[0]

    Firoza_image = face_recognition.load_image_file("data/Firoza/173.jpg")
    Firoza_face_encoding = face_recognition.face_encodings(Firoza_image)[0]

    Hariprasadh_image = face_recognition.load_image_file("data/Hariprasadh/200.jpg")
    Hariprasadh_face_encoding = face_recognition.face_encodings(Hariprasadh_image)[0]

    Krishnendhu_image = face_recognition.load_image_file("data/Krishnendhu/200.jpg")
    Krishnendhu_face_encoding = face_recognition.face_encodings(Krishnendhu_image)[0]
    '''
    Malavika_image = face_recognition.load_image_file("data/Malavika/.jpg")
    Malavika_face_encoding = face_recognition.face_encodings(Malavika_image)[0]'''

    Manhal_image = face_recognition.load_image_file("data/Manhal/200.jpg")
    Manhal_face_encoding = face_recognition.face_encodings(Manhal_image)[0]

    Maria_image = face_recognition.load_image_file("data/Maria/200.jpg")
    Maria_face_encoding = face_recognition.face_encodings(Maria_image)[0]

    Muralee_image = face_recognition.load_image_file("data/Muralee/172.jpg")
    Muralee_face_encoding = face_recognition.face_encodings(Muralee_image)[0]

    Nandhitha_image = face_recognition.load_image_file("data/Nandhitha/173.jpg")
    Nandhitha_face_encoding = face_recognition.face_encodings(Nandhitha_image)[0]

    Parvati_image = face_recognition.load_image_file("data/Parvati/200.jpg")
    Parvati_face_encoding = face_recognition.face_encodings(Parvati_image)[0]

    Rebin_image = face_recognition.load_image_file("data/Rebin/173.jpg")
    Rebin_face_encoding = face_recognition.face_encodings(Rebin_image)[0]

    Riya_image = face_recognition.load_image_file("data/Riya/200.jpg")
    Riya_face_encoding = face_recognition.face_encodings(Riya_image)[0]

    Rizin_image = face_recognition.load_image_file("data/Rizin/173.jpg")
    Rizin_face_encoding = face_recognition.face_encodings(Rizin_image)[0]

    Sada_image = face_recognition.load_image_file("data/Sada/200.jpg")
    Sada_face_encoding = face_recognition.face_encodings(Sada_image)[0]

    Sanjo_image = face_recognition.load_image_file("data/Sanjo/200.jpg")
    Sanjo_face_encoding = face_recognition.face_encodings(Sanjo_image)[0]

    Shibla_image = face_recognition.load_image_file("data/Shibla/200.jpg")
    Shibla_face_encoding = face_recognition.face_encodings(Shibla_image)[0]

    Sreelakshmi_image = face_recognition.load_image_file("data/Sreelakshmi/172.jpg")
    Sreelakshmi_face_encoding = face_recognition.face_encodings(Sreelakshmi_image)[0]

    Stephy_image = face_recognition.load_image_file("data/Stephy/200.jpg")
    Stephy_face_encoding = face_recognition.face_encodings(Stephy_image)[0]    

    Aswin_image = face_recognition.load_image_file("data/Aswin/42.jpg")
    Aswin_face_encoding = face_recognition.face_encodings(Aswin_image)[0]

    Vaishak_image = face_recognition.load_image_file("data/Vaishak/11.jpg")
    Vaishak_face_encoding = face_recognition.face_encodings(Vaishak_image)[0]



    known_face_encodings = [
    AVB_face_encoding,  
    Abhijith_face_encoding,  
    Abhinava_face_encoding,  
    Abhinavrijit_face_encoding,
    Adithyaraj_face_encoding,  
    Adwaith_face_encoding,  
    Afras_face_encoding,  
    Aneena_face_encoding,  
    Anirudh_face_encoding,  
    Ansal_face_encoding,  
    Aquin_face_encoding,  
    Arjunanikuttan_face_encoding,  
    Arjunv_face_encoding,  
    Ashfin_face_encoding,  
    Athira_face_encoding,  
    Chandana_face_encoding,  
    Devalakshmy_face_encoding,  
    Devananda_face_encoding,  
    Dinasree_face_encoding,  
    Firoza_face_encoding,  
    Hariprasadh_face_encoding,  
    Krishnendhu_face_encoding,  
    Manhal_face_encoding,  
    Maria_face_encoding,  
    Muralee_face_encoding,  
    Nandhitha_face_encoding,  
    Parvati_face_encoding,  
    Rebin_face_encoding,  
    Riya_face_encoding,  
    Rizin_face_encoding,  
    Sada_face_encoding,  
    Sanjo_face_encoding,  
    Shibla_face_encoding,  
    Sreelakshmi_face_encoding,  
    Stephy_face_encoding, 
    Aswin_face_encoding,
    Vaishak_face_encoding
    ]
    known_face_names = [
        "AVB",
        "Abhijith",
        "Abhinava",
        "Abhinavrijit",
        "Adithyaraj",
        "Adwaith",
        "Afras",
        "Aneena",
        "Anirudh",
        "Ansal",
        "Aquin",
        "Arjunanikuttan",
        "Arjunv",
        "Ashfin",
        "Athira",
        "Chandana",
        "Devalakshmy",
        "Devananda",
        "Dinasree",
        "Firoza",
        "Hariprasadh", 
        "Krishnendhu",
        "Manhal",
        "Maria",
        "Muralee",
        "Nandhitha",
        "Parvati",
        "Rebin",
        "Riya",
        "Rizin",
        "Sada",
        "Sanjo",
        "Shibla",
        "Sreelakshmi",
        "Stephy",
        "Aswin",
        "Vaishak"
    ]
    face_locations = []
    face_encodings = []
    face_names = []
    face_array = []
    face_array_1 = []
    datetoday = date.today().strftime("%m_%d_%y")
    # datetoday2 = date.today().strftime("%d-%B-%Y")
    current_time_1= datetime.now().strftime("%H:%M")
    if f'Attendance-{datetoday}-{current_time_1}.csv' not in os.listdir('/Users/abhinavrijit/Desktop/Attendance_output'):
        with open(f'/Users/abhinavrijit/Desktop/Attendance_output/Attendance-{datetoday}-{current_time_1}.csv', 'w') as f:
            f.write('Name,Roll,Timestamp')

    def add_attendance(name):
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")
        
        df = pd.read_csv(f'/Users/abhinavrijit/Desktop/Attendance_output/Attendance-{datetoday}-{current_time_1}.csv')
        if int(userid) not in list(df['Roll']):
            with open(f'/Users/abhinavrijit/Desktop/Attendance_output/Attendance-{datetoday}-{current_time_1}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
    process_this_frame = True
    cap = cv2.VideoCapture(0)
    tolerance = 0.54
    for i in range(20):
        sleep(1)
        if i == 10:
            sleep(5)
        ret, frame = cap.read()
        results = model_yolo.predict(frame, show=False)
        img = frame 
        process_this_frame = True
        if process_this_frame:
            #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                name = "Unknown"
        
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
        
                face_names.append(name)
                if i <10:
                    face_array += face_names
                else :
                    face_array_1 += face_names
            process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            '''top *= 4
            right *= 4
            bottom *= 4
            left *= 4'''

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
            # cv2.imshow('Video', frame)   #==>  Frame for face
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                xywh = box.xyxy[0]
                x, y, w, h = map(int, xywh)
                cv2.rectangle(img, (x, y), (w, h), (0, 255, 255), 2)
                cropped_face = img[y:y+h, x:x+w]
                cropped_face_resized = cv2.resize(cropped_face, (224, 224))
                cropped_face_rgb = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)
                cropped_face_preprocessed = cropped_face_rgb.astype('float32') / 255.0
                cropped_face_preprocessed = np.expand_dims(cropped_face_preprocessed, axis=0)
                pred = model_face_recognition.predict(cropped_face_preprocessed)
                max_index=np.argmax(pred)
                if pred[0, max_index] > 0.99:
                    if i <10:
                        face_array += face_names
                    else :
                        face_array_1 += face_names
                    
    common_faces = [face for face in face_array_1 if face in face_array]
    print(common_faces)
    unique_faces = set(common_faces)
    for face in unique_faces:
        if face == 'AVB':
            add_attendance('AVB_02')
        if face == 'Abhijith':
            add_attendance('Abhijith_01')
        if face == 'Abhinava':
            add_attendance('Abhinava_03')
        if face == 'Abhinavrijit':
            add_attendance('Abhinavrijit_04')
        if face == 'Adithyaraj':
            add_attendance('Adithyaraj_05')
        if face == 'Adwaith':
            add_attendance('Adwaith_06')
        if face == 'Afras':
            add_attendance('Afras_07')
        if face == 'Aneena':
            add_attendance('Aneena_08')
        if face == 'Anirudh':
            add_attendance('Anirudh_09')
        if face == 'Ansal':
            add_attendance('Ansal_10')
        if face == 'Aquin':
            add_attendance('Aquin_11')
        if face == 'Arjunanikuttan':
            add_attendance('Arjunanikuttan_12')
        if face == 'Arjunv':
            add_attendance('Arjunv_13')
        if face == 'Ashfin':
            add_attendance('Ashfin_14')
        if face == 'Athira':
            add_attendance('Athira_15')
        if face == 'Chandana':
            add_attendance('Chandana_16')
        if face == 'Devalakshmy':
            add_attendance('Devalakshmy_17')
        if face == 'Devananda':
            add_attendance('Devananda_18')
        if face == 'Dinasree':
            add_attendance('Dinasree_19')
        if face == 'Firoza':
            add_attendance('Firoza_20')
        if face == 'Hariprasadh':
            add_attendance('Hariprasadh_21')
        if face == 'Krishnendhu':
            add_attendance('Krishnendhu_22')
        if face == 'Manhal':
            add_attendance('Manhal_23')
        if face == 'Maria':
            add_attendance('Maria_24')
        if face == 'Muralee':
            add_attendance('Muralee_25')
        if face == 'Nandhitha':
            add_attendance('Nandhitha_26')
        if face == 'Parvati':
            add_attendance('Parvati_27')
        if face == 'Rebin':
            add_attendance('Rebin_28')
        if face == 'Riya':
            add_attendance('Riya_29')
        if face == 'Rizin':
            add_attendance('Rizin_30')
        if face == 'Sada':
            add_attendance('Sada_31')
        if face == 'Sanjo':
            add_attendance('Sanjo_32')
        if face == 'Shibla':
            add_attendance('Shibla_33')
        if face == 'Sreelakshmi':
            add_attendance('Sreelakshmi_34')
        if face == 'Stephy':
            add_attendance('Stephy_35')
        if face == 'Aswin':
            add_attendance('Aswin_61')
        if face == 'Vaishak':
            add_attendance('Vaishak_38')
    cv2.destroyAllWindows()

if _name_ == "_main_":
    face()
