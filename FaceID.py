#Importing Kivy and other dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import os
import numpy as np
import cv2
import tensorflow as tf
from layers import L2Dist

#Building app an layout
class CamApp(App):
    def build(self):
        self.webcam = Image(size_hint=(1,.8))
        self.button = Button(text='Verify', on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text='Verification Uninitiated', size_hint=(1,.1))

        #Adding items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)

        #Loading .h5 SiameseNet model
        self.model = tf.keras.models.load_model('siamese_inference_model.h5', custom_objects={'L2Dist':L2Dist})

        #Setting up video capture
        self.capture = cv2.VideoCapture(0)
        #Running update() function for webcam feed
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    #Defeining continously-running function to get webcam feed
    def update(self, *args):
        #Read frame using OpenCV
        ret, frame = self.capture.read()
        frame = frame[100:100+250, 200:200+250, :]

        #Updating Image object
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture
    
    def preprocess_for_inference(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img, channels=3)
        img = tf.image.resize(img, (105, 105))
        img = img / 255.0
        #img = np.expand_dims(img, axis=0)
        return img
       
    #Defining verification function
    def verify(self, instance, detection_threshold=0.995, verification_threshold=0.1):
        #Capture input image from webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[100:100+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        #Build results array
        results = []
        #Loop through verification_images folder
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess_for_inference(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess_for_inference(os.path.join('application_data', 'verification_images', image))

            #Make predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0)
            print("RESULTS::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n", result)
            results.append(result)#[0][0])


        #Verification criteria
        verified = result < verification_threshold

        #Update verification text
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        #Logging inference data
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.6))
        Logger.info(np.sum(np.array(results) > 0.8))

        return results, verified

if __name__ == '__main__':
    CamApp().run()