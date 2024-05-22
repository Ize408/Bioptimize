#Imports
import RPi.GPIO as GPIO
import time
import os
import sys
import glob
import sqlite3
import math
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tflite_runtime.interpreter import Interpreter

from urllib.request import urlopen
from twilio.rest import Client

#Sensors

def Sludge_Temperature():
    os.system('modprobe w1-gpio')
    os.system('modproble w1-therm')
    base_dir = '/sys/bus/w1/devices/'
    device_folder = glob.glob(base_dir +'28*')[0]
    device_file = device_folder + '/w1_slave'
    f = open(device_file,'r')
    lines = f.readlines()
    f.close
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        f = open(device_file,'r')
        lines = f.readlines()
        f.close
    equals_pos = lines[1].find('t=')
    if equals_pos != 1:
        temp_string = lines[1][equals_pos+2:]
        Celcius = float(temp_string)/1000
        Celcius = round(Celcius,1)
        return Celcius

def Gas_Temperature():
    os.system('modprobe w1-gpio')
    os.system('modproble w1-therm')
    base_dir = '/sys/bus/w1/devices/'
    device_folder = glob.glob(base_dir +'28*')[1]
    device_file = device_folder + '/w1_slave'
    f = open(device_file,'r')
    lines = f.readlines()
    f.close
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        f = open(device_file,'r')
        lines = f.readlines()
        f.close
    equals_pos = lines[1].find('t=')
    if equals_pos != 1:
        temp_string = lines[1][equals_pos+2:]
        Celcius = float(temp_string)/1000
        Celcius = round(Celcius,1)
        return Celcius

def AnalogConverter():
    sys.path.append('../')
    from DFRobot_ADS1115 import ADS1115
    ADS1115_REG_CONFIG_PGA_6_144V        = 0x00
    ads1115 = ADS1115()

    ads1115.set_addr_ADS1115(0x49)
    ads1115.set_gain(ADS1115_REG_CONFIG_PGA_6_144V)
    adc0 = ads1115.read_voltage(0)
    time.sleep(0.2)
    adc1 = ads1115.read_voltage(1)
    time.sleep(0.2)
    adc2 = ads1115.read_voltage(2)
    time.sleep(0.2)
    adc3 = ads1115.read_voltage(3)
    time.sleep(0.2)
    sys.path.remove('../')
    return adc0, adc1, adc2, adc3

def pH_level():
    adc = AnalogConverter()
    analog1 = adc[1]['r']
    _temperature      = 25.0
    _acidVoltage      = 3154.00
    _neutralVoltage   = 2540.0
    slope     = (7.0-4.0)/((_neutralVoltage-1500.0)/3.0 - (_acidVoltage-1500.0)/3.0)
    intercept = 7.0 - slope*(_neutralVoltage-1500.0)/3.0
    _phValue  = slope*(int(analog1)-1500.0)/3.0+intercept
    _phValue = round(_phValue,2)
    return _phValue

def Gas():
    adc = AnalogConverter()
    analog2 = adc[2]['r']
    analog2 = int(analog2)/1000
    ratio = ((3.3/analog2)-1)*(996/850)
    ppm = 703*(ratio**-2.24)
    per = ppm
    return per

def Pressure():
    adc = AnalogConverter()
    analog3 = adc[3]['r']
    analog3 = float(analog3/1000)
    analog3 = (125000*analog3)-62500
    return analog3

#Application (IoT)

def ThingSpeak_DB():
    sludge_temp = Sludge_Temperature()
    pH = pH_level()
    gas = Gas()
    gas_temp = Gas_Temperature()
    pressure = Pressure()
    WRITE_API = "HCT3P98R9ZKIZZBJ"
    BASE_URL = "https://api.thingspeak.com/update?api_key={}".format(WRITE_API)
    thingspeakHttp = BASE_URL + "&field1={:.2f}&field2={:.2f}&field3={:.2f}&field4={:.2f}&field5={:.2f}".format(sludge_tedge_temp,gas_temp,pH,pressure,gas)
    print(thingspeakHttp)
    conn = urlopen(thingspeakHttp)
    print("Response: {}".format(conn.read()))
    conn.close()

def Notification():
    TWILIO_ACCOUNT_SID = "Insert SID"
    TWILIO_AUTH_TOKEN = "Insert Auth"
    TWILIO_PHONE_SENDER = "Insert"
    TWILIO_PHONE_RECIPIENT = "Insert"
    if pH >= 50:
        time.sleep(5)
        Temp_S = TS.read_temp()
        if Temp_S >= 50:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
            to=TWILIO_PHONE_RECIPIENT,
            from_=TWILIO_PHONE_SENDER,
            body="The Sludge Temperature is too High. Please store the gas for the safety and wait to cooldown"
            print(message.sid)
        else:
            print("Normal")
    elif BiogasA >= 90000:
        time.sleep(5)
        BiogasA = MS.Analog2a()
        if BiogasA >= 90000:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
            to=TWILIO_PHONE_RECIPIENT,
            from_=TWILIO_PHONE_SENDER,
            body="The Biogas Tank is almost Full. Please store the gas for the safety and wait to cooldown" 
            print(message.sid)
        else:
            print("Normal")

def local_database():
    conn = sqlite3.connect('database_optimize.db')
    sludge_temp = Sludge_Temperature()
    pH = pH_level()
    gas = Gas()
    gas_temp = Gas_Temperature()
    pressure = Pressure()
    c = conn.cursor()
    c.execute("INSERT INTO database VALUES (datetime('now','localtime'),%d,%d,%d,%d,%d)"%(sludge_temp,gas_temp,pH,gas,pre>
    conn.commit()
    conn.close()
    print("Succesfully recorded in the local database")

#Algorithms

def LinearRegression():
    # load the TFLite model
    interpreter = Interpreter('model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load the dataset
    df = pd.read_csv('database_0430.1.csv')
    
    # split the dataset into training and testing sets
    X = df.iloc[:, :-1].values # select all columns except the last one
    y = df.iloc[:, -1].values # select the last column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

    # define a function to predict using the TFLite model
    def predict_tflite(input_data):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    # make predictions on the testing set
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_pred_scaled = np.array([predict_tflite(x.reshape(1, -1)).flatten() for x in     X_test_scaled])
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    # evaluate the performance of the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    sludge_temp = 50 # set the sludge temperature
    pH_range = np.arange(3, 8, 0.1) # set the pH range to test
    gas_output = []
    for pH in pH_range:
        X1_new = np.array([[pH]])
        X1_new_2d = X1_new.reshape(1, -1)
        X2_new = np.array([sludge_temp])
        X2_new_2d = X2_new.reshape(1, -1)
        X1_new_scaled = scaler.transform(X1_new_2d)
        X2_new_scaled = scaler.transform(X2_new_2d)
        X_new_scaled = np.concatenate((X1_new_scaled, X2_new_scaled), axis=1).astype(np.float32)
        y_new_pred_scaled = predict_tflite(X_new_scaled)[0][0]
        y_new_pred = scaler.inverse_transform(y_new_pred_scaled.reshape(-1, 1)).flatten()
        gas_output.append(y_new_pred[0])
    best_pH = pH_range[np.argmax(gas_output)]
    print('Best pH for maximum gas output: {:.1f}'.format(best_pH))
    print('Mean squared error: {:.2f}'.format(mse))
    print('Root mean squared error: {:.2f}'.format(rmse))
    print('R-squared: {:.2f}'.format(r2))
    return best_pH

#Automation (aka Control System)

def add_pH():
    GPIO.setmode(GPIO.BCM)
    RELAY_1_GPIO = 23 #Pin 11
    GPIO.setup(RELAY_1_GPIO, GPIO.OUT) # GPIO Assign mode
    GPIO.output(RELAY_1_GPIO, GPIO.LOW) # out
    GPIO.output(RELAY_1_GPIO, GPIO.HIGH) # on
    time.sleep(10)
    GPIO.output(RELAY_1_GPIO, GPIO.LOW) # out
    GPIO.cleanup()

def subtract_pH():
    GPIO.setmode(GPIO.BCM)
    RELAY_2_GPIO = 24 #Pin 13
    GPIO.setup(RELAY_2_GPIO, GPIO.OUT) # GPIO Assign mode
    GPIO.output(RELAY_2_GPIO, GPIO.LOW) # out
    GPIO.output(RELAY_2_GPIO, GPIO.HIGH) # on
    time.sleep(150)
    GPIO.output(RELAY_2_GPIO, GPIO.LOW) # out
    GPIO.cleanup()

def Motor():
    GPIO.setmode(GPIO.BCM)
    RELAY_3_GPIO = 8 #Pin 13
    GPIO.setup(RELAY_3_GPIO, GPIO.OUT) # GPIO Assign mode
    GPIO.output(RELAY_3_GPIO, GPIO.LOW) # out
    GPIO.output(RELAY_3_GPIO, GPIO.HIGH) # on
    time.sleep(15)
    GPIO.output(RELAY_3_GPIO, GPIO.LOW) # out
    GPIO.cleanup()


#Process
def main():
    pH = pH_level()
    pH_optimized = LinearRegression()
    cycle = 0
    while cycle < 14:
        if cycle % 3 == 0:
            if pH > pH_optimized:
                subtract_pH()
                print("Optimizing pH level(-)")
                cycle = cycle +1
            elif pH < pH_optimized:
                add_pH()
                print("Optimizing pH level(+)")
                cycle = cycle + 1
        else:
            ThingSpeak_DB()
            Notification()
            local_database()
            cycle = cycle +1
            time.sleep(500)
    Motor()
    python = sys.executable
    os.execl(python, python, *sys.argv)

try:
    main()
except:
    python = sys.executable
    os.execl(python. python, *sys.argv)