# Smart Glasses

The Smart Glasses is basically a object detection machine. It is able to detect and speaks out the object it thinks it is detecting. Through this project, I was able to gain insights on different parts of engineering like machine learning. 



| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Alex L | Leigh High School | Electrical Engineering | Incoming Sophomore



![Headstone Image](AlexanderL_(2).jpg)
  
# Fourth Milstone: Gemini 

For my next modification, I want my smart glass to havd gemini in it. My idea was that when I ask a question though the microphone, it would send it to gemini and speak out the response. My first step was to get gemini into my raspberry pi and allow me to text it. I first had to get my Gemini API Key, which I got from the link at resource 3. The model of Gemini that I was using was Gemini 1.5 flash. I then install google.generative, which is a Python library provided by Google to interact with Gemini models. Now, when I run this progrom, my name shows up like this Alex: . This means that I can now type an interact with it and when I'm done, all I have to type it "quit" or "exit".

My next step was to allow it to listen to my quesiton and response. To do this, I had to install speech recgnition. This allows it to reconigize everything that I ask. The rest of the code was pretty simliar to the first code expect I have to include the listening and recongizing parts to the code.   
# Code

```c++
#Gemini
import google.generativeai as genai

# Replace with your real API key from https://aistudio.google.com/app/apikey
API_KEY = "AIzaSyALhjz0MSktymeCYsnOdFZlFKJy5jeuvXI"

# Set up Gemini API
genai.configure(api_key=API_KEY)

# Load Gemini Pro model
model = genai.GenerativeModel("gemini-1.5-flash")

# Simple chat loop
print("Mr. AI – type 'exit' to quit\n")
while True:
    user_input = input("Alex: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    try:
        response = model.generate_content(user_input)
        print("Gemini:", response.text)
    except Exception as e:
        print("Error:", e)


#Gemini Voice
import speech_recognition as sr
import google.generativeai as genai
import os

#  Set up Gemini API
genai.configure(api_key="AIzaSyALhjz0MSktymeCYsnOdFZlFKJy5jeuvXI")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()

#  Set up microphone + recognizer
r = sr.Recognizer()
mic = sr.Microphone()

#  Speak function using espeak + aplay
def speak(text):
    print(" Speaking...")
    os.system(f'espeak "{text}" --stdout | aplay')

print("Voice Assistant Ready. Say 'exit' to stop.\n")

while True:
    try:
        with mic as source:
            print("🎙 Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

        print(" Recognizing...")
        question = r.recognize_google(audio)
        print(f"You: {question}")

        if question.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break

        print(" Asking Gemini...")
        response = chat.send_message(question)
        answer = response.text.strip()
        print("Gemini:", answer)

        speak(answer)

    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        speak("Sorry, I didn't catch that.")

    except Exception as e:
        print("Error:", e)
        speak("There was an error.")


```


# Third Milestone: Putting Everything Together and Modifications

<iframe width="560" height="315" src="https://www.youtube.com/embed/tK_v8TLOG5g?si=rV_tHapDcVUGc7e-" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


To put everything together, I had to attach the raspberry pi and camera to my glass. To do this, I first 3D printed out a case that I built for the camera. I design the case and built it out on Onshape. My case allows the camera to slide into the case, making it easy to take the camera on and off. With the case, it is easier hotglued the case onto the glasses. The case is placed in the middle of the glasses, so it can directly point to where you are looking at while not blocking you vision. 

For my modifications, I plan to make my smart glasses more about sports. I started with different types of balls for it to identify. To do this, I searched up on google "types of sport balls dataset". I was able to find two different data set link in other resources. The two dataset combined had about 350+ picture for each types of balls. I put all the picture into teachable machines, teaching my model to identify the different types. I then use the code below to use the model on my raspberry pi. Some challenges I had was that there weren't a lot of pictures and had to find more of it. The picture were also not accurate as some of it had other types of balls it in. I had to spend time delete all the incorrect images from the model. 

![Glasses](glassescam.jpg)
![Onshape](OnshapeCam.jpg)

# Code

```c++
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import time
from picamera2 import Picamera2, Preview
picam2 = Picamera2()
picam2.start_preview(Preview.NULL)



# --- Load labels from file ---
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]




# --- Set the input tensor for the interpreter ---
def set_input_tensor(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details['index'], image)




# --- Run inference and return top result ---
def classify_image(interpreter, image):
    set_input_tensor(interpreter, image)
    interpreter.invoke()




    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details['index'])[0]




    top_result = np.argmax(output)
    return top_result, output[top_result]


# --- Setup paths â
#Adjust Paths as needed
MODEL_PATH = "/home/alex/model_unquant.tflite"
LABEL_PATH = "/home/alex/labels.txt"


# --- Load model and allocate tensors ---
interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
_, height, width, _ = input_details[0]['shape']




# --- Load labels ---
labels = load_labels(LABEL_PATH)




# --- Initialize Picamera2 ---

picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()




# --- Main loop ---
print("Starting camera inference. Press 'q' to quit.")
while True:
    frame = picam2.capture_array()




    # Preprocess frame for model
    image = cv2.resize(frame, (width, height))




    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)




    #label_id is the index of the predicted label, prob is the confidence score
    label_id, prob = classify_image(interpreter, image)
    label_text = f"{labels[label_id]} ({prob:.2f})"




    # Display result on image
    cv2.putText(frame, f"{label_text}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)




    cv2.imshow("Picamera2 - TFLite eClassification", frame)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cv2.destroyAllWindows()
picam2.stop()

```

<!--For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE-->

# Second Milestone: Dectection and Text to Speech


<iframe width="560" height="315" src="https://www.youtube.com/embed/hrnL3eaF_RA?si=hmuBEDchvEbqhB54" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my second milestone, I was able to add the object detection and text to speech to work. I downloaded TensorFlow including the the detection model. It model also came with a text to speech feature where it will say the object it is confident in detecting. I was able to plug in a headphone into my raspberry pi and allowed it to work. Some challenges was that my raspberry pi was very bugging. This made me restart the pi mulitple times. For my final milestone, I would be making modification. I would have to attach the device to my glasses and come up with some ideas to enhance my project.


# First Milestone: Setting up my Pi and taking picutres

<iframe width="560" height="315" src="https://www.youtube.com/embed/qkuLk5L94jU?si=lQ4hLNNZCEYkl45Z" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my first milestone, I began by setting up my Raspberry Pi. The components I used included the Pi itself, a camera module, capture card, mouse, keyboard, and a micro HDMI to HDMI cable. First, I downloaded the operating system software onto the SD card. Then, I connected all the necessary devices incldung mouse, keyboard, power cord, and capture card. Next, I attached the camera module to the Pi. This step presented one of the main challenges I faced: the camera wasn’t working for several hours. After troubleshooting, I discovered that the issue was a loose connection—the camera wasn’t inserted deep enough into the port. I had to remove the Pi's case in order to secure the connection properly. My goal for the upcoming week is to begin coding and use the provided training model to enable the camera to detect objects.


# Starter Project: JitterBug

<iframe width="560" height="315" src="https://www.youtube.com/embed/jvyMFoV6fmQ?si=jYPeiz33Uxprydh4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my starter project, I did the JitterBug. There are many componenets to this project, including  switch, led lights, and the vibrator motor. The way it work is by clikcing on the switch. This allow current to be pass through to led lights, making them light up. This represents the eyes of the bug. Currents are also going to the vibrator. The vibrator are connected to the the legs of the bug. The constant shaking allows the bug to move forward. The main challenge of this project was soldering, which was new to me. I had never solder before so it took some time to learn. But after I got used to soldering, the project was pretty straight forward.

![JitterBug Image](image0.jpg)

<!--# Schematics 
Here's where you'll put images of your schematics. [Tinkercad](https://www.tinkercad.com/blog/official-guide-to-tinkercad-circuits) and [Fritzing](https://fritzing.org/learning/) are both great resoruces to create professional schematic diagrams, though BSE recommends Tinkercad becuase it can be done easily and for free in the browser. -->



# Bill of Materials

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Raspberry Pi | A single-board computer that runs my object detection models | $58.99 | <a href="https://www.amazon.com/dp/B0CMZST24Y?ref=cm_sw_r_cso_cp_apin_dp_Y63HAE5CHH0DGMEAV253_1&ref_=cm_sw_r_cso_cp_apin_dp_Y63HAE5CHH0DGMEAV253_1&social_share=cm_sw_r_cso_cp_apin_dp_Y63HAE5CHH0DGMEAV253_1&starsLeft=1"> Link </a> |
| Raspberry Pi Camera Module | Camera for object detection | $13.99 | <a href="https://www.amazon.com/dp/B01ER2SKFS?ref=cm_sw_r_cso_cp_apin_dp_1CTZAM6W55YVQMMQNVET&ref_=cm_sw_r_cso_cp_apin_dp_1CTZAM6W55YVQMMQNVET&social_share=cm_sw_r_cso_cp_apin_dp_1CTZAM6W55YVQMMQNVET&starsLeft=1"> Link </a> |
| Headphones | allows the user to hear what the camera is detecting | $6.97 | <a href="https://www.amazon.com/LUDOS-Headphones-Warranty-Earphones-Microphone/dp/B0DYJVN2P5/ref=sr_1_1_sspa?crid=2HTI16L5JEL3H&dib=eyJ2IjoiMSJ9.flmg1d_4rO4VWCZWRPBm-N9WuGa3AZEloEcI312URGPFSB7ZhkjxjbBf7eHtt_jAGkWznnjJqiF6wuDtPyNOMLHOLsExI3jfU5nI2zKfizAnrthE5K6td42UJaqbcTnQGeO33afql8MaKxWfA5Q1ha9CUtpLSHIJyQ5JoGrnjbjQBNboCWnLK0G4uA9bLUar-b9fBWw1XaYEFkNUPfh5AnApF9n4G8UAs9t1X6UF-L6-MGYQ3FFXMQSvY3Bqec3BpJkgv2OpFh77avGHFDnIwm80lkvXIqNmOcHkQW-AKMs.v8kYtETrIDKSvOLChZGgiOnZWOOKFodq6vpWt8rRuAs&dib_tag=se&keywords=ear%2Bbud%2Bwhite&qid=1750719519&s=electronics&sprefix=ear%2Bbud%2Bwhit%2Celectronics%2C166&sr=1-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"> Link </a> |
| Glasses | allows the user to wear the object dection device | $6.99 | <a href="https://www.amazon.com/dp/B0BSF4PL2Q?ref=cm_sw_r_cso_cp_apin_dp_9G1KS2KBDABJZ2GR6485&ref_=cm_sw_r_cso_cp_apin_dp_9G1KS2KBDABJZ2GR6485&social_share=cm_sw_r_cso_cp_apin_dp_9G1KS2KBDABJZ2GR6485&starsLeft=1"> Link </a> |
| CanaKit raspberry Pi 4 starter kit | assemble and set up raspberry pi | $109.99 | <a href="https://www.amazon.com/CanaKit-Raspberry-Pi-Starter-Kit/dp/B07V2B4W63/ref=sr_1_3?crid=2ESN7MMEDJ6J9&dib=eyJ2IjoiMSJ9.yMZqFlZ5jzlaNE8OMzrM0b662J8iglskUJLh2UCEXv9iiTf-HxPv5EK8gYOUiJ6GO518wQmOc8OjXk0Y666Oo9OzxD1zRF-f9I2UGLhRJAn--McjNKdaPY0QX_Sz1dibGFc5ChyoFD9EAlXC9T486Tgg1RWIcUu_Xcl341mKekBnUi9KZqjNOmsjxiVsiG7Q1RjFYbzcS1s8VWijfllJ7yNrjTCkFJkbPBQGwk5XzHdaddbOorRLH6znvQSigmt9DHd6lE9d3KS2toTJDXYiei9nGzVtmLELW6YcefcoSS0.eJLApTyUgxoCePcskCE1oNkM4MKL0WZEUjzwYirq3Uo&dib_tag=se&keywords=canakit%2Braspberry%2Bpi%2B4%2Bstarter%2Bkit&qid=1750719168&s=electronics&sprefix=canakit%2Brapsbeery%2Bpi%2B4%2Bstarter%2Bkit%2Celectronics%2C125&sr=1-3&th=1"> Link </a> |
| Canakit USB-C Power Supply | Power up Raspbery Pi | $9.99 | <a href="https://www.amazon.com/CanaKit-Raspberry-Power-Supply-USB-C/dp/B07TYQRXTK"> Link </a> |
| CanaKit Raspberry Pi 4 Case | portect the raspberry pi itself from physical damages | $9.99 | <a href="https://www.amazon.com/iUniker-Raspberry-Aluminium-Heatsink-Supply/dp/B07D3S4KBK/ref=pd_bxgy_thbs_d_sccl_2/143-2944956-3619348?pd_rd_w=Zbahh&content-id=amzn1.sym.de9a1315-b9df-4c24-863c-7afcb2e4cc0a&pf_rd_p=de9a1315-b9df-4c24-863c-7afcb2e4cc0a&pf_rd_r=8W9MDS8C1A6AG6HF0GWE&pd_rd_wg=rzCox&pd_rd_r=d3418ecd-a978-4529-becc-d2f7df4319f7&pd_rd_i=B07D3S4KBK&th=1"> Link </a> |

# Other Resources/Examples
- [Sports Ball Dataset 1]((https://www.kaggle.com/datasets/mdkabinhasan/sports-ball-dataset))
- [Sports Ball Dataset 2]((https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification))
- [Gemini Api Key]([(https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification)](https://aistudio.google.com/app/apikey))


