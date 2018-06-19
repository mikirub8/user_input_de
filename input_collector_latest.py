from pymouse import PyMouseEvent
from pymouse import PyMouse
from operator import sub
import time
import thread
import threading
from math import sqrt
import sys
from signal import signal,SIGINT
import daemon
import pickle
from os import getcwd

###lastest version- inclueds the stop func
def stop(signum,frame):
    print "Inside Stop"
    MouseE_inst.dump_to_disk()
    MouseE_inst.stop()
    sys.exit()
signal(SIGINT,stop)

class MouseE(PyMouseEvent,PyMouse):
    def __init__(self):
        PyMouseEvent.__init__(self)
        self.in_movment = 0
        (self.x,self.y) = (0,0)
        self.move_analysis = []
        self.move_element_db =[]
        self.start_time = 0
        self.end_time = 0
        self.click_time = 0
        self.move_time =0
        self.event = threading.Event()
    def click(self,x,y,button,press): 
        if(time.time() - self.click_time) > 1: #this line filters 2 entry in one click
            if len(self.move_analysis) > 10 :#only movment with len>10
                            self.click_time = time.time()
                            me_inst = MovmentElement(self.start_time,self.click_time,self.move_analysis)
                            self.move_element_db.append(me_inst)
                            print "length is:" , len(self.move_element_db)
            #print"len:",  len(self.move_analysis), "time:" ,  self.end_time - self.click_time
            self.event.set()
    def move(self,x,y):
        #print time.time() - self.move_time
        #self.move_time = time.time()
        if self.in_movment:
            ### new screen resultion is 1920x1080
            x_mod = x if x<=1920 and x>=0 else 1920 if x>1920 else 0
            y_mod = y if y<=1080 and y>=0 else 1080 if y>1080 else 0
            (self.x,self.y) = (x,y)
            self.move_analysis.append((x_mod,y_mod))
            #print x_mod,y_mod
            #print x,y
    def dump_to_disk(self):
        #no longer scan_data is used we generate db.pickle list of coords holding the path
        path = []
        i =0
        for item in self.move_element_db:
            path.append(item.relative_path)
            i+=1
        with open(getcwd() + "\db.pickle",'wb') as handle:
            #pickle.dump(path, handle)
             pickle.dump(self.move_element_db,handle)
    def detect_eom(self):
        (x,y) = (0,0)
        while 1:
            (x,y) =  self.position()
            time.sleep(0.05)
            if (x,y) == (self.x,self.y):
                if self.in_movment:
                    self.event.clear()
                    ##wait for click (3 sed timeout)
                    self.event.wait(2)
                    ## end moving
                    self.end_time = time.time()
                self.in_movment = 0
                self.move_analysis = list()
            else:
                if not self.in_movment:
                    ## start moving 
                    self.start_time = time.time()
                    #print "start_time is:",self.start_time,self.in_movment
                self.in_movment = 1
                (self.x,self.y) = self.position()


class MovmentElement:
    def __init__(self,start_time,end_time,move_analysis):
        self.duration = self.calc_duration(start_time,end_time)
        self.passed_distance = len(move_analysis)
        self.relative_path =  []
        self.calc_relative_path(move_analysis)
        self.distance =0
        self.direction_coords =0
        self.calc_distance(move_analysis)
        self.speed = self.calc_speed()
    def calc_distance(self,move_analysis):
        lidx = len(move_analysis) - 1
        first_coord = move_analysis[0]
        last_coord = move_analysis[lidx]
        self.distance = sqrt(((first_coord[0] - last_coord[0]) ** 2) + ((first_coord[1] - last_coord[1]) ** 2))
        self.direction_coords = [first_coord[0] - last_coord[0],first_coord[1] - last_coord[1]]
    def calc_duration(self,stime,etime):
        return (etime - stime)
    def calc_speed(self):
        return (self.passed_distance / self.duration)
    def calc_relative_path(self,move_analysis):
        #for item in move_analysis:
        #self.relative_path.append(tuple(map(sub,move_analysis[0],item)))
        self.relative_path = move_analysis

if __name__ == "__main__":
    user_input = raw_input("Confirm that you want to overrun old db (yes\\no) !!!\n")
    if(user_input!="yes"):
        quit()
    MouseE_inst = MouseE()
    MouseE_inst.daemon = False
    print "Inside Prog"
    ##thread.start_new_thread(MouseE_inst.start,())
    MouseE_inst.start() 
    thread.start_new_thread(MouseE_inst.detect_eom,())
    try:
        while True:
            MouseE_inst.join(0.1)
    except KeyboardInterrupt:
        "print keyboard interrupt"