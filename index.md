# Smart Glasses

The Smart Glasses is basically a object detection machine. It is able to detect and speaks out the object it thinks it is detecting. Through this project, I was able to gain insights on different parts of engineering like machine learning. 



| **Engineer** | **School** | **Area of Interest** | **Grade* |
|:--:|:--:|:--:|:--:|
| Alex L | Leigh High School | Electrical Engineering | Incoming Sophomore

**Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.**

![Headstone Image](AlexanderL_(2).jpg)
  
# Final Milestone: Modification



I started my modification by searching for types of sports ball data set. I found one that had ten differet kinds of sport balls, with 250 plus images samples. This allows me to train the model and made it extremely accurate. I h


For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE

# Second Milestone: Dectection and Text to Speech


<iframe width="560" height="315" src="https://www.youtube.com/embed/hrnL3eaF_RA?si=hmuBEDchvEbqhB54" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my second milestone, I was able to add the object detection and text to speech to work. I downloaded TensorFlow including the the detection model. It model also came with a text to speech feature where it will say the object it is confident in detecting. I was able to plug in a headphone into my raspberry pi and allowed it to work. Some challenges was that my raspberry pi was very bugging. This made me restart the pi mulitple times. For my final milestone, I would be making modification. I would have to attach the device to my glasses and come up with some ideas to enhance my project.



# First Milestone: Setting up my Pi and taking picutres

<iframe width="560" height="315" src="https://www.youtube.com/embed/qkuLk5L94jU?si=lQ4hLNNZCEYkl45Z" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my first milestone, I began by setting up my Raspberry Pi. The components I used included the Pi itself, a camera module, capture card, mouse, keyboard, and a micro HDMI to HDMI cable. First, I downloaded the operating system software onto the SD card. Then, I connected all the necessary devices incldung mouse, keyboard, power cord, and capture card. Next, I attached the camera module to the Pi. This step presented one of the main challenges I faced: the camera wasn’t working for several hours. After troubleshooting, I discovered that the issue was a loose connection—the camera wasn’t inserted deep enough into the port. I had to remove the Pi's case in order to secure the connection properly. My goal for the upcoming week is to begin coding and use the provided training model to enable the camera to detect objects.


# Starter Project: JitterBug

<iframe width="560" height="315" src="https://www.youtube.com/embed/jvyMFoV6fmQ?si=jYPeiz33Uxprydh4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my starter project, it is a JitterBug. The componenets of this project are the on/off switch, led lights, and the vibrator motor. WHen you click the on switch, the current allows the led lights to turn on. It also makes the vibrator starts vibrating, making the legs move, creating a forword motion. I ran into some challenges along the way espcially in soldering and allowing it to connect.


# Schematics 
Here's where you'll put images of your schematics. [Tinkercad](https://www.tinkercad.com/blog/official-guide-to-tinkercad-circuits) and [Fritzing](https://fritzing.org/learning/) are both great resoruces to create professional schematic diagrams, though BSE recommends Tinkercad becuase it can be done easily and for free in the browser. 

# Code
Here's where you'll put your code. The syntax below places it into a block of code. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize it to your project needs. 

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
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Sports Ball Dataset]([https://trashytuber.github.io/YimingJiaBlueStamp/](https://www.kaggle.com/datasets/mdkabinhasan/sports-ball-dataset))
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)


