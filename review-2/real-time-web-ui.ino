/*
 * Structural Health Monitoring - Simplified Version
 * ESP32 + 2 ADXL345 Sensors + WiFi Only
 * No SD Card Required
 */

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include "arduinoFFT.h"

// ==================== CONFIGURATION ====================
#define NUM_SENSORS 2
#define SAMPLE_RATE 200  // Hz
#define FFT_SIZE 1024

// WiFi credentials - copy config.h.example to config.h and edit
#include "config.h"

// Create two I2C buses
TwoWire I2C_1 = TwoWire(0);
TwoWire I2C_2 = TwoWire(1);

// I2C Bus pins
#define I2C1_SDA 21
#define I2C1_SCL 22
#define I2C2_SDA 23
#define I2C2_SCL 18

// ADXL345 Configuration
#define ADXL345_ADDRESS 0x53
#define ADXL345_POWER_CTL   0x2D
#define ADXL345_DATA_FORMAT 0x31
#define ADXL345_DATAX0      0x32
#define ADXL345_BW_RATE     0x2C
#define ADXL345_DEVID       0x00

// Detection thresholds
#define RMS_THRESHOLD_FACTOR 1.5
#define FREQ_SHIFT_THRESHOLD 0.02
#define KURTOSIS_THRESHOLD 5.0

// ==================== GLOBAL VARIABLES ====================
TwoWire* I2C_BUS[NUM_SENSORS] = {&I2C_1, &I2C_2};
int16_t accel_raw[NUM_SENSORS][3];
float accel_g[NUM_SENSORS][3];
float accel_ensemble[3];
float accel_offset[NUM_SENSORS][3] = {0};

double fft_input[FFT_SIZE];
double fft_output[FFT_SIZE];
ArduinoFFT FFT = ArduinoFFT(fft_input, fft_output, (double)FFT_SIZE, (double)SAMPLE_RATE);

float baseline_rms[3] = {0};
float baseline_freq[3] = {0};
float baseline_kurtosis[3] = {3.0};

unsigned long sample_count = 0;
unsigned long last_sample_time = 0;
bool calibration_complete = false;

// RMS and frequency tracking
float current_rms = 0.0;
float dominant_freq = 0.0;
int fft_sample_idx = 0;

// Alert thresholds (m/s²)
#define ALERT_WARNING_RMS  0.05
#define ALERT_CRITICAL_RMS 0.15

WebServer server(80);

// ==================== SENSOR FUNCTIONS ====================

bool writeADXL(TwoWire &wire, uint8_t reg, uint8_t data) {
  wire.beginTransmission(ADXL345_ADDRESS);
  wire.write(reg);
  wire.write(data);
  uint8_t err = wire.endTransmission();
  if (err != 0) {
    Serial.printf("ADXL write error %d (reg 0x%02X)\n", err, reg);
    return false;
  }
  return true;
}

bool readADXL(TwoWire &wire, uint8_t reg, uint8_t* buffer, uint8_t len) {
  wire.beginTransmission(ADXL345_ADDRESS);
  wire.write(reg);
  uint8_t err = wire.endTransmission(false);
  if (err != 0) {
    Serial.printf("ADXL read error %d (reg 0x%02X)\n", err, reg);
    return false;
  }
  uint8_t count = wire.requestFrom(ADXL345_ADDRESS, len);
  if (count != len) {
    Serial.printf("ADXL: expected %d bytes, got %d\n", len, count);
    return false;
  }
  for (uint8_t i = 0; i < len; i++) {
    buffer[i] = wire.read();
  }
  return true;
}

bool initADXL(uint8_t sensor_id) {
  TwoWire &wire = *I2C_BUS[sensor_id];

  // Validate sensor by reading DEVID
  uint8_t devid = 0;
  if (!readADXL(wire, ADXL345_DEVID, &devid, 1) || devid != 0xE5) {
    Serial.printf("Sensor %d: DEVID check failed (0x%02X), retrying...\n", sensor_id, devid);
    delay(50);
    if (!readADXL(wire, ADXL345_DEVID, &devid, 1) || devid != 0xE5) {
      Serial.printf("Sensor %d: DEVID retry failed (0x%02X)\n", sensor_id, devid);
      return false;
    }
  }

  writeADXL(wire, ADXL345_POWER_CTL, 0x00);
  delay(10);
  writeADXL(wire, ADXL345_DATA_FORMAT, 0x08);  // Full resolution, ±2g
  writeADXL(wire, ADXL345_BW_RATE, 0x0B);      // 200 Hz
  writeADXL(wire, ADXL345_POWER_CTL, 0x08);    // Measurement mode
  delay(10);
  return true;
}

void readSensorData(uint8_t sensor_id) {
  TwoWire &wire = *I2C_BUS[sensor_id];
  uint8_t buffer[6];
  readADXL(wire, ADXL345_DATAX0, buffer, 6);
  
  accel_raw[sensor_id][0] = (int16_t)((buffer[1] << 8) | buffer[0]);
  accel_raw[sensor_id][1] = (int16_t)((buffer[3] << 8) | buffer[2]);
  accel_raw[sensor_id][2] = (int16_t)((buffer[5] << 8) | buffer[4]);
  
  for (int axis = 0; axis < 3; axis++) {
    accel_g[sensor_id][axis] = (accel_raw[sensor_id][axis] / 256.0) - accel_offset[sensor_id][axis];
  }
}

void computeEnsembleAverage() {
  for (int axis = 0; axis < 3; axis++) {
    accel_ensemble[axis] = (accel_g[0][axis] + accel_g[1][axis]) / 2.0;
  }
}

float calculateRMS(float* data, int length) {
  float sum_sq = 0;
  for (int i = 0; i < length; i++) {
    sum_sq += data[i] * data[i];
  }
  return sqrt(sum_sq / length);
}

// ==================== CALIBRATION ====================

void performCalibration() {
  Serial.println("\n=== Calibration Starting ===");
  Serial.println("Keep sensors still for 10 seconds...");
  
  const int CAL_SAMPLES = 2000;
  float sum[NUM_SENSORS][3] = {0};
  
  for (int sample = 0; sample < CAL_SAMPLES; sample++) {
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
      TwoWire &wire = *I2C_BUS[sensor];
      uint8_t buffer[6];
      readADXL(wire, ADXL345_DATAX0, buffer, 6);
      
      int16_t x = (int16_t)((buffer[1] << 8) | buffer[0]);
      int16_t y = (int16_t)((buffer[3] << 8) | buffer[2]);
      int16_t z = (int16_t)((buffer[5] << 8) | buffer[4]);
      
      sum[sensor][0] += x / 256.0;
      sum[sensor][1] += y / 256.0;
      sum[sensor][2] += z / 256.0;
    }
    delay(5);
    if (sample % 200 == 0) Serial.print(".");
  }
  
  Serial.println("\nCalculating offsets...");
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    for (int axis = 0; axis < 3; axis++) {
      accel_offset[sensor][axis] = sum[sensor][axis] / CAL_SAMPLES;
      if (axis == 2) accel_offset[sensor][axis] -= 1.0;  // Remove gravity
    }
    Serial.printf("Sensor %d: X=%.3f Y=%.3f Z=%.3f\n", 
                  sensor+1, accel_offset[sensor][0], 
                  accel_offset[sensor][1], accel_offset[sensor][2]);
  }
  
  calibration_complete = true;
  Serial.println("=== Calibration Complete ===\n");
}

// ==================== WEB SERVER ====================

const char* getAlertStatus() {
  if (current_rms > ALERT_CRITICAL_RMS) return "CRITICAL";
  if (current_rms > ALERT_WARNING_RMS) return "WARNING";
  return "NORMAL";
}

const char* getAlertColor() {
  if (current_rms > ALERT_CRITICAL_RMS) return "#e74c3c";
  if (current_rms > ALERT_WARNING_RMS) return "#f39c12";
  return "#27ae60";
}

void handleRoot() {
  String html = R"rawliteral(<!DOCTYPE html><html><head><title>SHM Monitor</title>
<style>
body{font-family:Arial;margin:20px;background:#f0f0f0;}
.container{max-width:800px;margin:0 auto;background:white;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}
h1{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px;}
table{width:100%;border-collapse:collapse;margin:20px 0;}
th,td{border:1px solid #ddd;padding:10px;text-align:left;}
th{background:#3498db;color:white;}
.value{font-weight:bold;color:#2c3e50;}
#alert{padding:15px;border-radius:8px;color:white;font-size:1.2em;text-align:center;margin:15px 0;}
.stats{display:flex;gap:20px;margin:15px 0;}
.stat-box{flex:1;padding:15px;border-radius:8px;background:#ecf0f1;text-align:center;}
.stat-box .label{font-size:0.85em;color:#7f8c8d;}
.stat-box .val{font-size:1.4em;font-weight:bold;color:#2c3e50;}
</style>
<script>
var es = new EventSource('/events');
es.onmessage = function(e) {
  var d = JSON.parse(e.data);
  document.getElementById('samples').textContent = d.samples;
  document.getElementById('rms').textContent = d.rms;
  document.getElementById('freq').textContent = d.freq;
  document.getElementById('alert').textContent = d.alert;
  document.getElementById('alert').style.background = d.alertColor;
  for (var a = 0; a < 3; a++) {
    var ax = ['X','Y','Z'][a];
    document.getElementById('ens_ms_'+ax).textContent = d.ensemble_ms[a];
    document.getElementById('ens_mg_'+ax).textContent = d.ensemble_mg[a];
  }
  for (var s = 0; s < 2; s++) {
    for (var a = 0; a < 3; a++) {
      document.getElementById('s'+s+'_'+['X','Y','Z'][a]).textContent = d.sensors[s][a];
    }
  }
  document.getElementById('uptime').textContent = d.uptime;
};
</script>
</head><body>
<div class='container'>
<h1>Structural Health Monitor</h1>
<div id='alert' style='background:#27ae60;'>NORMAL</div>
<div class='stats'>
  <div class='stat-box'><div class='label'>RMS (m/s&sup2;)</div><div class='val' id='rms'>--</div></div>
  <div class='stat-box'><div class='label'>Dom. Freq (Hz)</div><div class='val' id='freq'>--</div></div>
  <div class='stat-box'><div class='label'>Samples</div><div class='val' id='samples'>--</div></div>
</div>
<h2>Ensemble Average</h2>
<table><tr><th>Axis</th><th>m/s&sup2;</th><th>mg</th></tr>
<tr><td>X</td><td class='value' id='ens_ms_X'>--</td><td class='value' id='ens_mg_X'>--</td></tr>
<tr><td>Y</td><td class='value' id='ens_ms_Y'>--</td><td class='value' id='ens_mg_Y'>--</td></tr>
<tr><td>Z</td><td class='value' id='ens_ms_Z'>--</td><td class='value' id='ens_mg_Z'>--</td></tr>
</table>
<h2>Individual Sensors</h2>
<table><tr><th>Sensor</th><th>X (m/s&sup2;)</th><th>Y (m/s&sup2;)</th><th>Z (m/s&sup2;)</th></tr>
<tr><td>Sensor 1</td><td class='value' id='s0_X'>--</td><td class='value' id='s0_Y'>--</td><td class='value' id='s0_Z'>--</td></tr>
<tr><td>Sensor 2</td><td class='value' id='s1_X'>--</td><td class='value' id='s1_Y'>--</td><td class='value' id='s1_Z'>--</td></tr>
</table>
<p>Uptime: <span id='uptime'>--</span>s | Sample Rate: 200 Hz</p>
</div></body></html>)rawliteral";

  server.send(200, "text/html", html);
}

void handleEvents() {
  // Build JSON payload
  String json = "{";
  json += "\"samples\":" + String(sample_count);
  json += ",\"rms\":\"" + String(current_rms, 4) + "\"";
  json += ",\"freq\":\"" + String(dominant_freq, 1) + "\"";
  json += ",\"alert\":\"" + String(getAlertStatus()) + "\"";
  json += ",\"alertColor\":\"" + String(getAlertColor()) + "\"";
  json += ",\"uptime\":" + String(millis() / 1000);

  json += ",\"ensemble_ms\":[";
  for (int a = 0; a < 3; a++) {
    if (a) json += ",";
    json += "\"" + String(accel_ensemble[a] * 9.81, 3) + "\"";
  }
  json += "],\"ensemble_mg\":[";
  for (int a = 0; a < 3; a++) {
    if (a) json += ",";
    json += "\"" + String(accel_ensemble[a] * 1000, 2) + "\"";
  }
  json += "],\"sensors\":[";
  for (int s = 0; s < NUM_SENSORS; s++) {
    if (s) json += ",";
    json += "[";
    for (int a = 0; a < 3; a++) {
      if (a) json += ",";
      json += "\"" + String(accel_g[s][a] * 9.81, 3) + "\"";
    }
    json += "]";
  }
  json += "]}";

  String payload = "data: " + json + "\n\n";
  server.send(200, "text/event-stream", payload);
}

// ==================== SETUP ====================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║  Structural Health Monitoring System   ║");
  Serial.println("║  ESP32 + Dual ADXL345 + WiFi          ║");
  Serial.println("╚════════════════════════════════════════╝\n");
  
  // Initialize I2C buses
  Serial.println("Initializing I2C buses...");
  I2C_1.begin(I2C1_SDA, I2C1_SCL, 100000);
  I2C_2.begin(I2C2_SDA, I2C2_SCL, 100000);
  delay(100);
  
  // Initialize sensors
  Serial.println("Initializing sensors...");
  for (int i = 0; i < NUM_SENSORS; i++) {
    bool ok = initADXL(i);
    Serial.printf("Sensor %d: %s\n", i+1, ok ? "✓ OK" : "✗ Failed");
  }
  
  // Connect to WiFi
  Serial.println("\nConnecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected!");
    Serial.println("IP address: " + WiFi.localIP().toString());
    Serial.println("Open browser: http://" + WiFi.localIP().toString());
    
    server.on("/", handleRoot);
    server.on("/events", handleEvents);
    server.begin();
  } else {
    Serial.println("\n✗ WiFi failed - continuing offline");
  }
  
  // Calibrate
  delay(2000);
  performCalibration();
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║         SYSTEM READY!                  ║");
  Serial.println("╚════════════════════════════════════════╝\n");
  
  last_sample_time = micros();
}

// ==================== MAIN LOOP ====================

void loop() {
  unsigned long current_time = micros();
  
  // Maintain 200 Hz sample rate
  if (current_time - last_sample_time >= 5000) {
    last_sample_time = current_time;
    
    // Read sensors
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
      readSensorData(sensor);
    }
    
    // Calculate average
    computeEnsembleAverage();
    sample_count++;

    // Compute RMS of ensemble magnitude (exclude gravity: Z - 1g)
    float mag = sqrt(accel_ensemble[0]*accel_ensemble[0] +
                     accel_ensemble[1]*accel_ensemble[1] +
                     (accel_ensemble[2]-1.0)*(accel_ensemble[2]-1.0)) * 9.81;
    current_rms = current_rms * 0.99 + mag * 0.01;  // exponential moving average

    // Accumulate samples for FFT (use Z-axis vibration)
    if (fft_sample_idx < FFT_SIZE) {
      fft_input[fft_sample_idx] = accel_ensemble[2] - 1.0;  // remove gravity
      fft_output[fft_sample_idx] = 0;
      fft_sample_idx++;
    }
    if (fft_sample_idx >= FFT_SIZE) {
      FFT.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
      FFT.Compute(FFT_FORWARD);
      FFT.ComplexToMagnitude();
      dominant_freq = FFT.MajorPeak();
      fft_sample_idx = 0;
    }

    // Print every 1000 samples (5 seconds)
    if (sample_count % 1000 == 0) {
      Serial.printf("\n[%lu] Ensemble: X=%6.3f Y=%6.3f Z=%6.3f m/s²\n",
                    sample_count,
                    accel_ensemble[0] * 9.81,
                    accel_ensemble[1] * 9.81,
                    accel_ensemble[2] * 9.81);
      
      for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
        Serial.printf("       Sensor %d: X=%6.3f Y=%6.3f Z=%6.3f m/s²\n",
                      sensor + 1,
                      accel_g[sensor][0] * 9.81,
                      accel_g[sensor][1] * 9.81,
                      accel_g[sensor][2] * 9.81);
      }
    }
  }
  
  // Handle web requests
  server.handleClient();
}
