
import pyautogui
import argparse

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':1, 'slow':10, 'medium':5}

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def move(self, position):
        #print(position)
        x,y=position
        #print(x*2,y*2)
        #pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)
        pyautogui.moveTo(x,y,self.speed)
        #pyautogui.doubleClick()
        
    
