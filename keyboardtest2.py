import pygame
import time
import wiringpi
import picamera
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--folder_name", default="/home/pi/images", help="where to save the images")
args = parser.parse_args()
if not os.path.exists(args.folder_name):
    os.makedirs(args.folder_name)
    
camera = picamera.PiCamera()
camera.resolution = (224, 224)
camera.framerate = 60
direction = ''
imagenumber = 0
wiringpi.wiringPiSetupGpio()
pygame.init()
pygame.display.set_mode((100,100))
wiringpi.pinMode(18, wiringpi.GPIO.PWM_OUTPUT)
wiringpi.pinMode(19, wiringpi.GPIO.PWM_OUTPUT)
wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

wiringpi.pwmSetClock(192)
wiringpi.pwmSetRange(2000)
start_time = time.time()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if( pygame.key.get_pressed()[pygame.K_w] != 0):
            wiringpi.pwmWrite(18, 160)
            wiringpi.pwmWrite(19, 140)
            print('Forward')
            direction = 'f'
        elif(pygame.key.get_pressed()[pygame.K_a] != 0):
            wiringpi.pwmWrite(18, 160)
            print('Left')
            direction = 'l'
        elif(pygame.key.get_pressed()[pygame.K_s] != 0):
            wiringpi.pwmWrite(18, 140)
            wiringpi.pwmWrite(19, 160)
            print('Backward')
            direction = 'b'
        elif(pygame.key.get_pressed()[pygame.K_d] != 0):
            wiringpi.pwmWrite(19, 140)
            print('Right')
            direction = 'r'
        else:
    #(len(pygame.key.get_pressed()) == 0):
            wiringpi.pwmWrite(18, 150)
            wiringpi.pwmWrite(19, 150)
            print('No movement')
            direction = ''
    if not direction == "":
        file_name =str.format('image%002d.%s.jpg' % (imagenumber, direction)) 
        full_file_name = os.path.join(args.folder_name, file_name)
        camera.capture_sequence([full_file_name])
        imagenumber +=1
        elapsedtime = time.time() - start_time
        print(elapsedtime)
        start_time = time.time()
