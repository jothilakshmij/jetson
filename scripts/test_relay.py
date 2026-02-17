#!/usr/bin/env python3
"""
Standalone Relay GPIO Test — Run directly on Jetson (no Docker needed)
=====================================================================
Tests Pin 7 (BOARD) by toggling HIGH/LOW so you can verify the relay clicks.

Wiring:
  Pin 2  (5V)    → Relay VCC
  Pin 6  (GND)   → Relay GND
  Pin 7  (GPIO09)→ Relay IN

Usage:
  sudo python3 test_relay.py           # 3 pulses, 1s each
  sudo python3 test_relay.py --pin 7 --pulses 5 --duration 2
"""

import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="Test Jetson GPIO relay output")
    parser.add_argument("--pin", type=int, default=7, help="BOARD pin number (default: 7)")
    args = parser.parse_args()

    try:
        import Jetson.GPIO as GPIO
    except ImportError:
        print("ERROR: Jetson.GPIO not installed.")
        print("  Install with:  sudo pip3 install Jetson.GPIO")
        sys.exit(1)

    print("=" * 50)
    print(f"  RELAY BLINK TEST — 2 pulses")
    print(f"  Pin: {args.pin} (BOARD mode)")
    print(f"  Watch the relay module LED!")
    print("=" * 50)

    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(args.pin, GPIO.OUT, initial=GPIO.LOW)
    print(f"\n[OK] GPIO Pin {args.pin} initialized (currently LOW)\n")

    try:
        # Pulse 1: ON
        print("  Pulse 1/2: ON  ← LED should light up")
        GPIO.output(args.pin, GPIO.HIGH)
        time.sleep(1)

        # Pulse 1: OFF
        print("  Pulse 1/2: OFF ← LED should turn off")
        GPIO.output(args.pin, GPIO.LOW)
        time.sleep(1)

        # Pulse 2: ON
        print("  Pulse 2/2: ON  ← LED should light up")
        GPIO.output(args.pin, GPIO.HIGH)
        time.sleep(1)

        # Pulse 2: OFF
        print("  Pulse 2/2: OFF ← LED should turn off")
        GPIO.output(args.pin, GPIO.LOW)

        print(f"\n[DONE] 2 blink pulses completed.")
        print("  YES relay LED blinked? → Wiring is correct!")
        print("  NO LED at all?")
        print("    1. Check VCC → Pin 2 (5V)")
        print("    2. Check GND → Pin 6")
        print("    3. Check IN  → Pin 7")

    except KeyboardInterrupt:
        print("\n\nInterrupted — setting pin LOW")
    finally:
        GPIO.output(args.pin, GPIO.LOW)
        GPIO.cleanup()
        print("[OK] GPIO cleaned up")

if __name__ == "__main__":
    main()
