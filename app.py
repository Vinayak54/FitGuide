from fileinput import filename
import cv2
import time
import streamlit as st
import numpy as np
import json
import requests
import pyttsx3
import threading
from streamlit_lottie import st_lottie
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils  # for mp models to draw on images
mp_pose = mp.solutions.pose  # for importing pose estimation models e.g. face detection, face mesh, hand detections etc


# ----Config Section-----

# Set up title
st.set_page_config(page_title="Fitguide",
                   layout="wide", 
                   page_icon="🧊", 
                   menu_items= 
         {'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!" }    
)

st.title("FITGUIDE")





# Load animations via Lottie website
def load_animation_via_link(web_link):
    r = requests.get(web_link)
    if r.status_code != 200:
        return None
    return r.json()


# Load animations via local environment
def load_animation_via_desktop(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Style your contact form
def style_contact_doc(file_name):
    with open(file=file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)




# -------Header Section------
with st.container():
    columnL, columnR = st.columns(2)
    with columnL:
        st.subheader("A Computer Vision based Fitness solution :movie_camera: ")
        st.markdown("Where artificial intelligence tracks your workout techniques in real-time to get the most out of your workout :camera: ")
        # lottie_animation = load_animation_via_link("https://assets5.lottiefiles.com/packages/lf20_v4isjbj5.json")
        lottie_animation = load_animation_via_link("https://lottie.host/a1af0e5e-d564-4905-89d6-c1337bdbcbad/syo4qnlkiP.json")
        st_lottie(lottie_animation, height=400, width=700, key="ai_robot1")
    with columnR:
        
        lottie_animation = load_animation_via_link("https://lottie.host/d1d47dc5-131f-4a73-a584-723848bda680/zJzFOs4Vul.json")
        st_lottie(lottie_animation, height=400, width=700, key="github")
        st.subheader(" How to use this app ")
        st.markdown("""     
        
        FITGUIDE is designed to monitor your workout movement and technique during each exercise rep. **Good workout technique** is rewarded with one rep count and **bad workout technique** is not recorded at all. 

        1. Select on a workout on the left 
        2. Click **Start Workout** to begin workout session
        3. Press 'q' on your keyboard to exit session once completed


        """)



# --------------Workout Selection -----------------------
with st.container():
    columnL, columnR = st.columns(2)
    with columnL:
        st.write('---------------')
        workout = st.selectbox('Select your exercise:', ('None', 'Bicep Curls', 'Shoulder Press', 'Chest Press' , 'Dead-lift'))
        st.write('Your selection:', workout)
        st.text("")
        # run = st.checkbox('Start Workout', value=False, key=1)
        run = st.button('Start Workout', key=1)
        if run:
            run = True
        st.text("")
        st.text("")
    with columnR:
        pass

# Prompt user if  "Run Workout" is selected without an exercise picked
if workout == 'None' and run is True:
    st.markdown("You need to select a workout before you can begin session")




# Bicep Curls
if workout == "Bicep Curls":
    with st.container():
        columnL, columnR = st.columns(2)
        with columnL:
            chest_press_url = "https://cdn.shopify.com/s/files/1/1501/0558/files/KillerArm2_BasicCurls.gif?v=1514662455"
            st.image(chest_press_url)
        with columnR:
            st.header("Bicep Curls")
            st.subheader("Preparation")
            st.markdown("""
                * Warm up your arms by performing light stretches before beginning workout 
                

            """)
            # st.markdown("""""")

            st.subheader("Tips")
            st.markdown("""
            
            - Curl dumbbell towards the region between your chest and bicep
            - Curl back down to starting position and repeat motion 
            - Keep shoulder and elbow locked to your side through out the exercise
            - Squeeze biceps tight through out the exercise
              

            """)
            st.subheader("Notes")
            st.markdown("""
                * Make sure your streaming device (camera/webcam) is on 
                * Face your camera during your workout for the algorithm to successfully detect your body joints during your workout

            """)
            if run:
                st.markdown(""" 
                **Remember to prioritize good technique when performing your exercise to get the best 
                out of your sessions - enjoy your workout!**  
                """)

# Chaitanya code bicep curl


                engine = pyttsx3.init()
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose


                def speak_count(count):
                    count_text = str(count)
                    engine.say(count_text)
                    engine.runAndWait()


                # def speak_text(text_to_speak):
                #   engine.say(text_to_speak)
                #  engine.runAndWait()
                def calculate_angle(a, b, c):
                    a = np.array(a)  # First
                    b = np.array(b)  # Mid
                    c = np.array(c)  # End
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    if angle > 180.0:
                        angle = 360 - angle
                    return angle


                cap = cv2.VideoCapture(0)
                counter = 0
                stage = None
                counted = False
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        try:
                            landmarks = results.pose_landmarks.landmark
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            angle = calculate_angle(shoulder, elbow, wrist)
                            cv2.putText(image, str(angle),
                                        tuple(np.multiply(wrist, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                            if 70 < angle < 140:
                                cv2.putText(image, "Lift Upwards", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 225), 2,
                                            cv2.LINE_AA)
                            if angle > 160:
                                stage = "Perfect!"
                            if angle < 30 and stage == 'Perfect!':
                                stage = "Complete"
                                counter += 1
                                print(counter)
                                tts_thread = threading.Thread(target=speak_count, args=(counter,))
                                tts_thread.start()
                                count_text = str(counter)
                                engine.say(count_text)
                                engine.runAndWait()
                        except:
                            cv2.putText(image, "Camera is not aligned", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2,
                                        cv2.LINE_AA)
                        cv2.putText(image, 'REPS', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter),
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, 'STAGE', (100, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage,
                                    (100, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        mp.solutions.drawing_utils.draw_landmarks(
                            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )
                        # cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
                        # cv2.imread('FITGUIDE',cv2.IMREAD_COLOR)
                        cv2.imshow('FITGUIDE', image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                cap.release()
                cv2.destroyAllWindows()

# Shoulder Press
if workout == "Shoulder Press":
    with st.container():
        columnL, columnR = st.columns(2)
        with columnL:
            chest_press_url = "https://cdn.shopify.com/s/files/1/1501/0558/files/BoulderShoulders5_ArniePress.gif?v=1514663967"

            st.image(chest_press_url)
        with columnR:
            st.header("Shoulder Press")
            st.subheader("Preparation")
            st.markdown("""
                * Warm up your arms by performing light stretches before beginning workout 


            """)
            # st.markdown("""""")

            st.subheader("Tips")
            st.markdown("""

            - Lift dumbbell towards the region between your shoulder and lift upwards
            - Put it back down to starting position and repeat motion 
            - Keep shoulder and elbow locked to your side through out the exercise
            - Focus on shoulders through out the exercise


            """)
            st.subheader("Notes")
            st.markdown("""
                * Make sure your streaming device (camera/webcam) is on 
                * Face your camera during your workout for the algorithm to successfully detect your body joints during your workout

            """)
            if run:
                st.markdown(""" 
                **Remember to prioritize good technique when performing your exercise to get the best 
                out of your sessions - enjoy your workout!**  
                """)

# Prasheel code shoulder press



        engine = pyttsx3.init()
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose


        def speak_count(count):
            count_text = str(count)
            engine.say(count_text)
            engine.runAndWait()


        def calculate_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            if angle > 180.0:
                angle = 360 - angle
            return angle


        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        counter = 0
        stage = None
        counted = False

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to capture frame")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)

                    cv2.putText(image, str(angle),
                                tuple(np.multiply(wrist, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    if 50 < angle < 170:
                        cv2.putText(image, "Straighten Your Elbows", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 225), 2,
                                    cv2.LINE_AA)
                    if angle > 160:
                        stage = "Perfect!"
                    if angle < 50 and stage == 'Perfect!':
                        stage = "Complete"
                        counter += 1
                        print(counter)
                        tts_thread = threading.Thread(target=speak_count, args=(counter,))
                        tts_thread.start()
                        count_text = str(counter)
                        engine.say(count_text)
                        engine.runAndWait()

                except Exception as e:
                    print(f"Error: {e}")
                    cv2.putText(image, "Camera is not aligned", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

                cv2.putText(image, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'STAGE', (100, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                # cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
                cv2.imshow('FITGUIDE', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

# Chest press
if workout == "Chest Press":
    with st.container():
        columnL, columnR = st.columns(2)
        with columnL:
            chest_press_url = "https://cdn.shopify.com/s/files/1/1501/0558/files/FlatBenchPress.gif?v=1507564367"

            st.image(chest_press_url)
            # lottie_animation = load_animation_via_link("https://assets1.lottiefiles.com/packages/lf20_fyj5ox9g.json")
            # st_lottie(lottie_animation, height=400, width=600, key="push_ups")
        with columnR:
            st.header("Chest press ")
            st.subheader("Preparation")
            st.markdown("""
                * Ensure you have performed light cardio and stretches before performing push ups to get your blood flow running 

            """)
            # st.markdown("""""")
            st.subheader("Tips")
            st.markdown("""

                        - Maintain a flat back throughout the lift.
                        - Use a strong, hip-width grip on the barbell.
                        - Lift with your legs and hips, not your back.
                        - Gradually increase the weight to challenge yourself safely.


                        """)
            st.subheader("Notes")
            st.markdown("""
                * Make sure your streaming device (camera/webcam) is on 
                * Face your camera during your workout for the algorithm to successfully detect your body joints during your workout

            """)
            if run:
                st.markdown("""
                **Remember to prioritize good technique when performing your exercise to get the best 
                out of your sessions - enjoy your workout!**

                """)

                import cv2
                import mediapipe as mp
                import numpy as np
                import pyttsx3
                import threading

                engine = pyttsx3.init()
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose


                def speak_count(count):
                    count_text = str(count)
                    engine.say(count_text)
                    engine.runAndWait()


                def calculate_angle(a, b, c):
                    a = np.array(a)  # First
                    b = np.array(b)  # Mid
                    c = np.array(c)  # End
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    if angle > 180.0:
                        angle = 360 - angle
                    return angle


                cap = cv2.VideoCapture(0)
                counter = 0
                stage = None
                counted = False
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        try:
                            landmarks = results.pose_landmarks.landmark
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            angle = calculate_angle(shoulder, elbow, wrist)
                            cv2.putText(image, str(angle),
                                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                            if angle > 130:
                                stage = "Perfect!"
                            if angle < 90 and stage == 'Perfect!':
                                stage = "Complete"
                                counter += 1
                                print(counter)
                                tts_thread = threading.Thread(target=speak_count, args=(counter,))
                                tts_thread.start()
                                count_text = str(counter)
                                engine.say(count_text)
                                engine.runAndWait()
                        except:
                            cv2.putText(image, "Camera is not aligned", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2,
                                        cv2.LINE_AA)
                        cv2.putText(image, 'REPS', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter),
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, 'STAGE', (100, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage,
                                    (100, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        mp.solutions.drawing_utils.draw_landmarks(
                            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )
                        # cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
                        cv2.imshow('FITGUIDE', image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                cap.release()
                cv2.destroyAllWindows()


# Dead Lift
if workout == "Dead-lift":
    with st.container():
        columnL, columnR = st.columns(2)
        with columnL:
            chest_press_url = "https://i.pinimg.com/originals/81/f1/23/81f1230ab56427e0bb86e2b3c2c6cb6f.gif"
            st.image(chest_press_url,width=500)
        with columnR:
            st.header("Dead-lift")
            st.subheader("Preparation")
            st.markdown("""
                * Perform warm up exercises like jogging/skipping on the spot to prepare your legs for squats 

            """)
            # st.markdown("""""")

            st.subheader("Tips")
            st.markdown("""
            
            - Maintain a flat back throughout the lift.
            - Use a strong, hip-width grip on the barbell.
            - Lift with your legs and hips, not your back.
            - Gradually increase the weight to challenge yourself safely.
              
            """)
            st.subheader("Notes")
            st.markdown("""
                * Make sure your streaming device (camera/webcam) is on 
                * Face your camera during your workout for the algorithm to successfully detect your body joints during your workout

            """)

            engine = pyttsx3.init()
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose


            def speak_count(count):
                count_text = str(count)
                engine.say(count_text)
                engine.runAndWait()


            def calculate_angle(a, b, c):
                a = np.array(a)  # First
                b = np.array(b)  # Mid
                c = np.array(c)  # End
                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)
                if angle > 180.0:
                    angle = 360 - angle
                return angle


            cap = cv2.VideoCapture(0)
            counter = 0
            stage = None
            counted = False
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    try:
                        landmarks = results.pose_landmarks.landmark
                        foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        angle = calculate_angle(foot, hip, wrist)
                        cv2.putText(image, str(angle),
                                    tuple(np.multiply(wrist, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        if 20 < angle < 50:
                            cv2.putText(image, "Straighten Your Knees", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 225), 2,
                                        cv2.LINE_AA)
                        if angle < 20:
                            stage = "Perfect!"
                        if angle > 90 and stage == 'Perfect!':
                            stage = "Complete"
                            counter += 1
                            print(counter)
                            tts_thread = threading.Thread(target=speak_count, args=(counter,))
                            tts_thread.start()
                            count_text = str(counter)
                            engine.say(count_text)
                            engine.runAndWait()
                    except:
                        cv2.putText(image, "Camera is not aligned", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    2,
                                    cv2.LINE_AA)
                    cv2.putText(image, 'REPS', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter),
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'STAGE', (100, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage,
                                (100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    # cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
                    cv2.imshow('FITGUIDE', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()


st.write('---------------')



with st.container():
        st.markdown(
        """ **Disclaimer: The information provided on this app should not be considered as professional, medical or health advice (these are strictly the opinions of the developer) - please seek your doctor's recommendations for professional advice on your health situation or read at your own discretion.** """)

    # Sample diet and workout data with healthier Indian dishes
        veg_meals = {
            "Sunday": ["Oats Upma", "Moong Dal Chilla", "Sprouts Salad", "Green Tea"],
            "Monday": ["Quinoa Pulao", "Cucumber Raita", "Mixed Vegetable Soup"],
            "Tuesday": ["Paneer Tikka", "Brown Rice", "Dal Palak", "Mint Chutney"],
            "Wednesday": ["Masoor Dal", "Tofu Bhurji", "Spinach Salad", "Buttermilk"],
            "Thursday": ["Lentil Soup", "Stir-fried Tofu", "Cucumber Raita", "Green Tea"],
            "Friday": ["Mushroom Curry", "Quinoa", "Mixed Greens Salad", "Herbal Tea"],
            "Saturday": ["Grilled Vegetable Sandwich", "Sprouts Salad", "Fresh Fruit Juice"],
        }

        non_veg_meals = {
            "Sunday": ["Grilled Chicken Breast", "Brown Rice", "Stir-fried Broccoli", "Herbal Tea"],
            "Monday": ["Salmon Salad", "Quinoa", "Mixed Greens", "Lemon Water"],
            "Tuesday": ["Tandoori Chicken", "Roti", "Mixed Vegetable Salad", "Mint Raita"],
            "Wednesday": ["Chicken and Vegetable Stir-Fry", "Steamed Rice", "Cucumber Salad", "Green Tea"],
            "Thursday": ["Lamb Curry", "Brown Rice", "Cauliflower Sabzi", "Buttermilk"],
            "Friday": ["Fish Tacos", "Quinoa Salad", "Cucumber Raita", "Lemon Water"],
            "Saturday": ["Turkey Sandwich", "Mixed Greens Salad", "Fresh Fruit Juice"],
        }

        # Sample workout plan with exercise details
        sample_workouts = {

            "Monday": ["Strength Training", "Upper Body Workout", "Chest Press", "Bicep Curl", "Shoulder Press"],
            "Tuesday": ["Yoga", "Stretching"],
            "Wednesday": ["HIIT", "Core Workout", "Planks", "Russian Twists", "Mountain Climbers"],
            "Thursday": ["Cardio", "Legs Workout", "Deadlifts", "Leg Curls", "Calf Raises"],
            "Friday": ["Strength Training", "Upper Body Workout", "Bench Press", "Tricep Dips", "Lat Pulldowns"],
            "Saturday": ["Cardio", "Legs Workout", "Squats", "Leg Press", "Lunges"],
            "Sunday": ["Rest"],
        }
with st.container():

        # Streamlit UI
        st.title("Get Diet and Workout Recommendations")
with st.container():
    columnL, columnR = st.columns([2, 1])
    with columnL:
        user_age = st.slider("Select your age:",18, 60, 30)
        user_gender = st.selectbox("Select your gender:", ["Male", "Female"])
        user_goal = st.selectbox("Select your fitness goal:", ["Build Muscle", "Lose Weight", "Maintain"])
        user_weight = st.number_input("Enter your weight (in kilograms):")
        user_height = st.number_input("Enter your height (in centimeters):")
        veg_or_non_veg = st.radio("Choose Vegetarian or Non-Vegetarian", ["Vegetarian", "Non-Vegetarian"])


        def calculate_caloric_needs(weight, height, age, gender, goal):
            if gender == "Male":
                caloric_needs = 10 * weight + 6.25 * height - 5 * age + 5
            else:
                caloric_needs = 10 * weight + 6.25 * height - 5 * age - 161

            if goal == "Build Muscle":
                caloric_needs += 300
            elif goal == "Lose Weight":
                caloric_needs -= 300

            return caloric_needs


        if st.button("Calculate Daily Caloric Needs"):
            caloric_needs = calculate_caloric_needs(user_weight, user_height, user_age, user_gender, user_goal)
            st.write(f"Your estimated daily caloric needs: {caloric_needs:.2f} calories")

        if st.button("Generate Diet and Workout Plan"):
            # Determine the recommended meals based on the user's choice (Vegetarian or Non-Vegetarian)
            if user_goal == "Build Muscle":
                if veg_or_non_veg == "Non-Vegetarian":
                    recommended_meals = non_veg_meals
                else:
                    recommended_meals = veg_meals
            else:
                recommended_meals = veg_meals
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    # Display the recommended diet plan
                    st.write("Recommended Indian Diet Plan for the Week:")
                    for day, meals in recommended_meals.items():
                        st.write(f"{day}:")
                        for meal in meals:
                            st.write(f"- {meal}")
                with col2:
                    # Display the recommended workout plan
                    st.write("\nRecommended Workout Plan for the Week:")
                    for day, exercises in sample_workouts.items():
                        st.write(f"{day}:")
                        for exercise in exercises:
                            st.write(f"- {exercise}")

        with columnR:
            lottie_animation = load_animation_via_link(
                "https://lottie.host/ee8c8c31-d7df-4810-b63f-6f8a6e75c327/F2vX2Q2gOf.json")
            st_lottie(lottie_animation, height=450, width=500, key="diet")


st.write('---------------')
style_contact_doc("style.css")
