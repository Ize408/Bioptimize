import os
import sqlite3
import sys

import pytest

os.environ["BIOPTIMIZE_SIMULATE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bioptimize1 as b  # noqa: E402


def test_ph_calibration_points():
    assert b.voltage_to_pH(b.PH_NEUTRAL_VOLTAGE) == pytest.approx(7.0)
    assert b.voltage_to_pH(b.PH_ACID_VOLTAGE) == pytest.approx(4.0)


def test_gas_sensor_out_of_range_returns_none():
    assert b.voltage_to_gas_ppm(0) is None
    assert b.voltage_to_gas_ppm(3300) is None
    assert b.voltage_to_gas_ppm(1500) > 0


def test_pressure_zero_at_half_volt():
    assert b.voltage_to_pressure(500) == pytest.approx(0)


def test_adjust_ph(monkeypatch):
    calls = []
    monkeypatch.setattr(b, "add_pH", lambda: calls.append("+"))
    monkeypatch.setattr(b, "subtract_pH", lambda: calls.append("-"))
    assert b.adjust_pH(6.0, 7.0) == "+"
    assert b.adjust_pH(8.0, 7.0) == "-"
    assert b.adjust_pH(7.1, 7.0) is None   # within tolerance, leave it
    assert b.adjust_pH(None, 7.0) is None  # no reading, don't dose blindly
    assert calls == ["+", "-"]


def test_target_ph_is_kept_in_safe_range():
    target = b.LinearRegression()
    assert b.PH_SAFE_MIN <= target <= b.PH_SAFE_MAX


def test_database_keeps_decimals(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setattr(b, "DB_FILE", str(db))
    b.local_database({"sludge_temp": 28.4, "gas_temp": 31.2, "gas": 1234.5,
                      "pressure": 5000.0, "pH": 6.85})
    row = sqlite3.connect(db).execute("SELECT pH, sludge_temp FROM readings").fetchone()
    assert row == (6.85, 28.4)


def test_relay_switches_off_even_if_something_fails(monkeypatch):
    states = []

    class G:
        def output(self, pin, value):
            states.append(value)
            if value == b.ON:
                raise KeyboardInterrupt  # e.g. ctrl+c while the pump is running

    monkeypatch.setattr(b, "GPIO", G())
    with pytest.raises(KeyboardInterrupt):
        b.run_relay(23, 1)
    assert states[-1] == b.OFF
