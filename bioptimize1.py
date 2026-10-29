#!/usr/bin/env python3
"""
Bioptimize - biogas digester monitoring and pH optimization
Raspberry Pi 3 + ADS1115 + DS18B20 x2 + pH meter + MQ gas sensor + pressure sensor + 3 relays

Run:        python3 bioptimize1.py
Test on PC: BIOPTIMIZE_SIMULATE=1 python3 bioptimize1.py
Settings are read from a .env file (see .env.example)
"""
import glob
import logging
import os
import random
import sqlite3
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- settings

def load_env(path=os.path.join(BASE_DIR, ".env")):
    """small .env reader so we don't need python-dotenv on the pi"""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


load_env()


def env(name, default, cast=str):
    value = os.environ.get(name, "")
    return cast(value) if value != "" else default


SIMULATE = env("BIOPTIMIZE_SIMULATE", 0, int) == 1

# relay pins (BCM numbering)
RELAY_ADD_PH = env("RELAY_ADD_PH", 23, int)
RELAY_SUBTRACT_PH = env("RELAY_SUBTRACT_PH", 24, int)
RELAY_MOTOR = env("RELAY_MOTOR", 8, int)
RELAY_ACTIVE_LOW = env("RELAY_ACTIVE_LOW", 0, int) == 1  # most cheap relay boards are active low

ADD_PH_SECONDS = env("ADD_PH_SECONDS", 10, float)
SUBTRACT_PH_SECONDS = env("SUBTRACT_PH_SECONDS", 150, float)
MOTOR_SECONDS = env("MOTOR_SECONDS", 15, float)

CYCLE_SECONDS = env("CYCLE_SECONDS", 500, float)
CYCLES_PER_ROUND = env("CYCLES_PER_ROUND", 14, int)
PH_CHECK_EVERY = env("PH_CHECK_EVERY", 3, int)
PH_TOLERANCE = env("PH_TOLERANCE", 0.2, float)    # don't dose if pH is already this close
# the target pH is never set outside this range, whatever the model says.
# anaerobic digesters usually work best around pH 6.8 - 7.2
PH_SAFE_MIN = env("PH_SAFE_MIN", 6.5, float)
PH_SAFE_MAX = env("PH_SAFE_MAX", 7.5, float)

# DS18B20 ids (folder names in /sys/bus/w1/devices, e.g. 28-0316a2794bff)
SLUDGE_SENSOR_ID = env("SLUDGE_SENSOR_ID", "")
GAS_SENSOR_ID = env("GAS_SENSOR_ID", "")

ADS1115_ADDRESS = env("ADS1115_ADDRESS", 0x49, lambda v: int(v, 0))
ADS1115_LIB_DIR = env("ADS1115_LIB_DIR", os.path.join(BASE_DIR, ".."))

# pH calibration (mV), from the DFRobot calibration
PH_ACID_VOLTAGE = env("PH_ACID_VOLTAGE", 3154.0, float)    # reading in pH 4.0 buffer
PH_NEUTRAL_VOLTAGE = env("PH_NEUTRAL_VOLTAGE", 2540.0, float)  # reading in pH 7.0 buffer

THINGSPEAK_WRITE_KEY = env("THINGSPEAK_WRITE_KEY", "")
TWILIO_ACCOUNT_SID = env("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = env("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_SENDER = env("TWILIO_PHONE_SENDER", "")
TWILIO_PHONE_RECIPIENT = env("TWILIO_PHONE_RECIPIENT", "")

SLUDGE_TEMP_ALERT = env("SLUDGE_TEMP_ALERT", 50.0, float)
GAS_ALERT = env("GAS_ALERT", 90000.0, float)
ALERT_COOLDOWN_SECONDS = env("ALERT_COOLDOWN_SECONDS", 1800, float)

DB_FILE = env("DB_FILE", os.path.join(BASE_DIR, "database_optimize.db"))
TRAINING_CSV = env("TRAINING_CSV", os.path.join(BASE_DIR, "data_0423.csv"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("bioptimize")


# ---------------------------------------------------------------- GPIO

if SIMULATE:
    class _FakeGPIO:
        BCM, OUT, HIGH, LOW = "BCM", "OUT", 1, 0

        def setmode(self, mode): pass
        def setwarnings(self, flag): pass
        def setup(self, pin, mode, initial=0): pass
        def output(self, pin, value): log.info("  [sim] GPIO%d -> %s", pin, "HIGH" if value else "LOW")
        def cleanup(self): pass
    GPIO = _FakeGPIO()
else:
    try:
        import RPi.GPIO as GPIO
    except ImportError:
        GPIO = None  # fine for "train" / "export" on a normal PC

ON = 0 if RELAY_ACTIVE_LOW else 1
OFF = 1 if RELAY_ACTIVE_LOW else 0


def setup_gpio():
    if GPIO is None:
        sys.exit("RPi.GPIO not found. Run this on the Raspberry Pi, or set BIOPTIMIZE_SIMULATE=1 to test")
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    for pin in (RELAY_ADD_PH, RELAY_SUBTRACT_PH, RELAY_MOTOR):
        GPIO.setup(pin, GPIO.OUT, initial=OFF)


def run_relay(pin, seconds):
    """switch a relay on for some seconds. always switches off again, even on error/ctrl+c"""
    try:
        GPIO.output(pin, ON)
        time.sleep(seconds if not SIMULATE else 0)
    finally:
        GPIO.output(pin, OFF)


# ---------------------------------------------------------------- sensors

def _read_ds18b20(device_id, index):
    """reads a DS18B20. needs dtoverlay=w1-gpio in /boot/config.txt"""
    if SIMULATE:
        return round(random.uniform(26, 30) if index == 0 else random.uniform(28, 34), 1)

    base_dir = "/sys/bus/w1/devices/"
    if device_id:
        device_folder = base_dir + device_id
    else:
        # glob order is not guaranteed, so sort it. better to set the ids in .env
        folders = sorted(glob.glob(base_dir + "28*"))
        if len(folders) <= index:
            log.warning("temperature sensor %d not found", index)
            return None
        device_folder = folders[index]

    device_file = device_folder + "/w1_slave"
    for _ in range(10):
        with open(device_file, "r") as f:
            lines = f.readlines()
        if lines and lines[0].strip()[-3:] == "YES":
            equals_pos = lines[1].find("t=")
            if equals_pos != -1:
                return round(float(lines[1][equals_pos + 2:]) / 1000, 1)
        time.sleep(0.2)
    log.warning("temperature sensor %s did not give a valid reading", device_folder)
    return None


def Sludge_Temperature():
    return _read_ds18b20(SLUDGE_SENSOR_ID, 0)


def Gas_Temperature():
    return _read_ds18b20(GAS_SENSOR_ID, 1)


_ads1115 = None


def AnalogConverter():
    """returns the 4 ADS1115 channels in mV"""
    global _ads1115
    if SIMULATE:
        return (0.0, random.uniform(2500, 2700), random.uniform(1200, 2400), random.uniform(530, 600))
    if _ads1115 is None:
        # DFRobot_ADS1115.py can be next to this file or in ADS1115_LIB_DIR
        sys.path.extend([BASE_DIR, ADS1115_LIB_DIR])
        from DFRobot_ADS1115 import ADS1115
        ADS1115_REG_CONFIG_PGA_6_144V = 0x00
        _ads1115 = ADS1115()
        _ads1115.set_addr_ADS1115(ADS1115_ADDRESS)
        _ads1115.set_gain(ADS1115_REG_CONFIG_PGA_6_144V)
    values = []
    for channel in range(4):
        values.append(float(_ads1115.read_voltage(channel)["r"]))
        time.sleep(0.2)
    return tuple(values)


def voltage_to_pH(millivolts):
    slope = (7.0 - 4.0) / ((PH_NEUTRAL_VOLTAGE - 1500.0) / 3.0 - (PH_ACID_VOLTAGE - 1500.0) / 3.0)
    intercept = 7.0 - slope * (PH_NEUTRAL_VOLTAGE - 1500.0) / 3.0
    return round(slope * (millivolts - 1500.0) / 3.0 + intercept, 2)


def voltage_to_gas_ppm(millivolts):
    volts = millivolts / 1000
    if volts <= 0 or volts >= 3.3:
        return None  # sensor disconnected / out of range, would divide by zero
    ratio = ((3.3 / volts) - 1) * (996 / 850)
    return round(703 * (ratio ** -2.24), 1)


def voltage_to_pressure(millivolts):
    # sensor gives 0.5 - 4.5 V
    return round(125000 * (millivolts / 1000) - 62500, 1)


def pH_level(adc=None):
    adc = adc or AnalogConverter()
    return voltage_to_pH(adc[1])


def Gas(adc=None):
    adc = adc or AnalogConverter()
    return voltage_to_gas_ppm(adc[2])


def Pressure(adc=None):
    adc = adc or AnalogConverter()
    return voltage_to_pressure(adc[3])


def read_all():
    """read every sensor once per cycle (before, each function re-read all sensors)"""
    adc = AnalogConverter()
    return {
        "sludge_temp": Sludge_Temperature(),
        "gas_temp": Gas_Temperature(),
        "pH": pH_level(adc),
        "gas": Gas(adc),
        "pressure": Pressure(adc),
    }


# ---------------------------------------------------------------- IoT

def ThingSpeak_DB(r):
    if not THINGSPEAK_WRITE_KEY:
        log.info("no THINGSPEAK_WRITE_KEY set, skipping upload")
        return
    fields = {"api_key": THINGSPEAK_WRITE_KEY}
    for i, key in enumerate(["sludge_temp", "gas_temp", "pH", "pressure", "gas"], start=1):
        if r[key] is not None:
            fields["field%d" % i] = "%.2f" % r[key]
    try:
        with urlopen("https://api.thingspeak.com/update?" + urlencode(fields), timeout=15) as conn:
            log.info("ThingSpeak response: %s", conn.read().decode())  # "0" means it was rejected
    except Exception as e:
        log.warning("ThingSpeak upload failed: %s", e)


_last_alert = {}


def send_sms(kind, body):
    now = time.time()
    if now - _last_alert.get(kind, 0) < ALERT_COOLDOWN_SECONDS:
        return  # already warned recently, don't spam
    _last_alert[kind] = now
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_SENDER and TWILIO_PHONE_RECIPIENT):
        log.warning("ALERT (sms not configured): %s", body)
        return
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(to=TWILIO_PHONE_RECIPIENT, from_=TWILIO_PHONE_SENDER, body=body)
        log.info("SMS sent: %s", message.sid)
    except Exception as e:
        log.warning("SMS failed: %s", e)


def Notification(r):
    # high readings are re-checked after 5 s so one bad reading doesn't send an sms
    if r["sludge_temp"] is not None and r["sludge_temp"] >= SLUDGE_TEMP_ALERT:
        time.sleep(5 if not SIMULATE else 0)
        temp = Sludge_Temperature()
        if temp is not None and temp >= SLUDGE_TEMP_ALERT:
            send_sms("temp", "The Sludge Temperature is too High (%.1f C). "
                             "Please store the gas for the safety and wait to cooldown" % temp)
    if r["gas"] is not None and r["gas"] >= GAS_ALERT:
        time.sleep(5 if not SIMULATE else 0)
        gas = Gas()
        if gas is not None and gas >= GAS_ALERT:
            send_sms("gas", "The Biogas Tank is almost Full. Please store the gas for the safety")


def local_database(r):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS readings (
            time TEXT, sludge_temp REAL, gas_temp REAL, gas REAL, pressure REAL, pH REAL)""")
        # parameters instead of %d - %d was cutting pH 6.85 down to 6
        conn.execute("INSERT INTO readings VALUES (datetime('now','localtime'),?,?,?,?,?)",
                     (r["sludge_temp"], r["gas_temp"], r["gas"], r["pressure"], r["pH"]))
        conn.commit()
    finally:
        conn.close()
    log.info("Successfully recorded in the local database")


def export_database_csv(path):
    """same columns as data_0423.csv so it can be used for training"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT sludge_temp, gas_temp, gas, pressure, pH FROM readings", conn)
    conn.close()
    df.columns = ["Sludge Temperature", "Gas Temperature", "Gas", "Pressure", "pH"]
    df.dropna().to_csv(path, index=False)
    log.info("exported %d rows to %s", len(df), path)


# ---------------------------------------------------------------- algorithm

def LinearRegression(csv_path=TRAINING_CSV, sludge_temp=None):
    """
    Learns gas output from pH + sludge temperature (polynomial regression, degree 2,
    so the curve can have a peak) and returns the pH with the highest predicted gas.

    Only searches inside the pH range that is actually in the data - outside of it
    the model is just guessing. The result is then kept inside PH_SAFE_MIN..PH_SAFE_MAX.
    """
    df = pd.read_csv(csv_path)
    df = df[df["pH"] >= 2]  # pH 0 / 1 rows are sensor errors
    X = df[["pH", "Sludge Temperature"]].values
    y = df["Gas"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), SkLinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    if sludge_temp is None:
        sludge_temp = float(np.median(df["Sludge Temperature"]))
    lo, hi = max(3.0, df["pH"].min()), min(8.0, df["pH"].max())
    pH_range = np.round(np.arange(lo, hi + 0.001, 0.1), 1)
    X_new = np.column_stack([pH_range, np.full_like(pH_range, sludge_temp)])
    gas_output = model.predict(X_new)
    best_pH = float(pH_range[np.argmax(gas_output)])

    print("pH range in the data: {:.1f} - {:.1f}".format(lo, hi))
    print("Best pH for maximum gas output: {:.1f}".format(best_pH))
    print("Mean squared error: {:.2f}".format(mse))
    print("Root mean squared error: {:.2f}".format(rmse))
    print("R-squared: {:.2f}".format(r2))

    safe = round(min(max(best_pH, PH_SAFE_MIN), PH_SAFE_MAX), 1)
    if abs(safe - best_pH) > 1e-9:
        log.warning("model suggested pH %.1f, using %.1f (outside PH_SAFE_MIN/PH_SAFE_MAX)", best_pH, safe)
    return safe


# ---------------------------------------------------------------- control

def add_pH():
    log.info("Optimizing pH level(+)")
    run_relay(RELAY_ADD_PH, ADD_PH_SECONDS)


def subtract_pH():
    log.info("Optimizing pH level(-)")
    run_relay(RELAY_SUBTRACT_PH, SUBTRACT_PH_SECONDS)


def Motor():
    log.info("Mixing")
    run_relay(RELAY_MOTOR, MOTOR_SECONDS)


def adjust_pH(pH, target):
    """returns what it did: '+', '-' or None"""
    if pH is None:
        log.warning("no pH reading, skipping pH adjustment")
        return None
    if pH < target - PH_TOLERANCE:
        add_pH()
        return "+"
    if pH > target + PH_TOLERANCE:
        subtract_pH()
        return "-"
    log.info("pH %.2f is close to target %.1f", pH, target)
    return None


# ---------------------------------------------------------------- main loop

def main():
    setup_gpio()
    pH_optimized = LinearRegression()
    log.info("target pH: %.1f", pH_optimized)
    cycle = 0
    try:
        while True:
            r = read_all()
            log.info("reading: %s", r)
            ThingSpeak_DB(r)
            Notification(r)
            local_database(r)

            if cycle % PH_CHECK_EVERY == 0:
                adjust_pH(r["pH"], pH_optimized)

            cycle += 1
            if cycle >= CYCLES_PER_ROUND:
                Motor()
                cycle = 0
            time.sleep(CYCLE_SECONDS if not SIMULATE else 1)
    finally:
        for pin in (RELAY_ADD_PH, RELAY_SUBTRACT_PH, RELAY_MOTOR):
            GPIO.output(pin, OFF)
        GPIO.cleanup()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "export":
        export_database_csv(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "train":
        LinearRegression()
    else:
        main()
