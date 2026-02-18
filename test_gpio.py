#!/usr/bin/env python3

import Jetson.GPIO as GPIO
import time

# Pin Definitions
output_pin = 7  # Physical BOARD pin 7

def main():
    GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

    try:
        print("Relay ON")
        GPIO.output(output_pin, GPIO.LOW)   # LOW = ON (for most relay modules)
        time.sleep(50)

        print("Relay OFF")
        GPIO.output(output_pin, GPIO.HIGH)  # HIGH = OFF

    finally:
        GPIO.cleanup()
        print("GPIO Cleaned up")

if __name__ == '__main__':
    main()
