#! /usr/bin/python

import sys
import struct
import serial
import time

locations=['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyUSB2','/dev/ttyUSB3']    
  
for device in locations:  
    try:  
        print "Trying...",device  
        TRNG = serial.Serial(
            port=device,
            baudrate=19200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            dsrdtr=False,
            rtscts=False,
            timeout=1)  
        break  
    except:  
        print "Failed to connect on",device     
  
try:
    count = 0
    while True:
        count = count + 1
        if (count > 8):
            TRNG.write('r')
        recv = TRNG.readline()
        if ("Ready" in recv):
            break
    print "Device is ready!"
except:
    print "Device not ready!"

try:  
    outb = open(sys.argv[1],"wb")
    start = time.time()
    TRNG.write('b')  
    for i in range(1000):
        recv = ord(TRNG.read()) 
        outb.write(struct.pack('B',recv))
        # print ord(recv)
    outb.close()
    finish = time.time()
    print (finish-start)
    TRNG.write('c')
except:  
    print "Failed to recieve data!"  
