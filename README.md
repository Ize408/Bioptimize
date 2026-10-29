# Bioptimize

Thesis project: monitoring and pH optimization of a biogas digester using a Raspberry Pi.

The system reads the digester's sensors, uploads the readings to ThingSpeak (viewed in the BiOptimize Android app), stores them in a local SQLite database, sends SMS alerts, and keeps the sludge pH near the value that a regression model predicts gives the most biogas.

![Schematic](SchematicDiagram(Bioptimize).png)

## Hardware

| Part | Connection |
|---|---|
| Raspberry Pi 3 Model B | |
| ADS1115 16-bit ADC (I2C) | SDA → GPIO2, SCL → GPIO3 |
| pH meter (DFRobot) | ADS1115 AIN1 |
| MQ gas sensor | ADS1115 AIN2 |
| McMaster-Carr 2957N11 pressure sensor | ADS1115 AIN3 |
| 2x DS18B20 temperature sensors (sludge, gas) | GPIO4 (1-Wire), with a pull-up resistor |
| Relay: pH up pump/valve | `RELAY_ADD_PH` (default GPIO23) |
| Relay: pH down pump/valve | `RELAY_SUBTRACT_PH` (default GPIO24) |
| Relay: mixing motor | `RELAY_MOTOR` (default GPIO8) |

The schematic shows the relays on GPIO21, GPIO20 and GPIO16, while the code uses 23, 24 and 8. Check your actual wiring and set the pins in `.env`.

## Files

| File | |
|---|---|
| `bioptimize1.py` | main program (sensors, ThingSpeak, SMS, database, model, pH control) |
| `data_0423.csv` | recorded sensor data used to train the model |
| `BiOptimize.apk` | Android app |
| `bioptimize.service` | runs the program on boot and restarts it if it stops |
| `tests/` | tests (run on a normal PC, no sensors needed) |

## Setup on the Raspberry Pi

1. Enable I2C and 1-Wire:
   ```
   sudo raspi-config     # Interface Options -> I2C -> Enable, 1-Wire -> Enable
   sudo reboot
   ```
   Check that the sensors are detected:
   ```
   i2cdetect -y 1               # ADS1115 should show up (0x48 or 0x49)
   ls /sys/bus/w1/devices/      # two folders starting with 28-
   ```

2. Get the code and install the libraries:
   ```
   git clone https://github.com/Ize408/Bioptimize.git
   cd Bioptimize
   sudo apt install python3-smbus python3-rpi.gpio
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Download `DFRobot_ADS1115.py` from DFRobot's GitHub (DFRobot_ADS1115, `python/raspberrypi` folder) and put it in the Bioptimize folder.

4. Settings:
   ```
   cp .env.example .env
   nano .env
   ```
   Fill in the ThingSpeak write key, the Twilio details, and the relay pins. Setting `SLUDGE_SENSOR_ID` and `GAS_SENSOR_ID` to the `28-...` folder names makes sure the two temperature sensors never get swapped.

5. Run it:
   ```
   python3 bioptimize1.py
   ```

6. To start it automatically on boot:
   ```
   sudo cp bioptimize.service /etc/systemd/system/
   sudo systemctl enable --now bioptimize
   journalctl -u bioptimize -f      # view the log
   ```

## How it works

Every cycle (500 s by default) the program:

1. reads all sensors once
2. uploads the readings to ThingSpeak and saves them in `database_optimize.db`
3. sends an SMS if the sludge temperature is ≥ 50 °C or the gas reading is ≥ 90000, re-checking after 5 s so that one bad reading doesn't trigger an alert, with at most one SMS per alert type every 30 minutes
4. every 3rd cycle, compares the pH with the target and runs the pH up or pH down relay if it's off by more than 0.2

After 14 cycles the mixing motor runs and the count starts over.

## pH target model

`LinearRegression()` fits a degree-2 polynomial regression of gas output on pH and sludge temperature, then picks the pH with the highest predicted gas. It only searches within the pH range that exists in the data, because predictions outside it aren't reliable. The result is always kept between `PH_SAFE_MIN` and `PH_SAFE_MAX` (6.5–7.5 by default), so the pumps never push the digester to a harmful pH.

```
python3 bioptimize1.py train                 # train the model and print the results
python3 bioptimize1.py export new_data.csv   # export the database in the same format as data_0423.csv
```

Notes on the current data (`data_0423.csv`):

- pH was saved as whole numbers (0, 1, 4, 5, 6) because the old database code used `%d`. This is fixed, and new data keeps decimals.
- Rows with pH 0 or 1 are sensor errors and are skipped during training.
- The gas reading rises steadily over time, so it likely measures gas building up in the tank rather than the production rate. That's why the model currently favors pH 4, which is too acidic for a digester, and the safe range overrides it. Retraining on new data with decimal pH values should give a more meaningful target.

## Testing without the hardware

```
BIOPTIMIZE_SIMULATE=1 python3 bioptimize1.py    # fake sensors and relays, 1 s cycles
pip install -r requirements.txt && pytest
```
On Windows CMD: `set BIOPTIMIZE_SIMULATE=1` then `python bioptimize1.py`.
