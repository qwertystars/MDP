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

// WiFi credentials - CHANGE THESE!
const char* ssid = "DESKTOP-2LR5MU4";        // ← CHANGE THIS
const char* password = "T05*790a";     // ← CHANGE THIS

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

WebServer server(80);

// ==================== SENSOR FUNCTIONS ====================

void writeADXL(TwoWire &wire, uint8_t reg, uint8_t data) {
  wire.beginTransmission(ADXL345_ADDRESS);
  wire.write(reg);
  wire.write(data);
  wire.endTransmission();
}

void readADXL(TwoWire &wire, uint8_t reg, uint8_t* buffer, uint8_t len) {
  wire.beginTransmission(ADXL345_ADDRESS);
  wire.write(reg);
  wire.endTransmission(false);
  wire.requestFrom(ADXL345_ADDRESS, len);
  for (uint8_t i = 0; i < len; i++) {
    buffer[i] = wire.read();
  }
}

void initADXL(uint8_t sensor_id) {
  TwoWire &wire = *I2C_BUS[sensor_id];
  writeADXL(wire, ADXL345_POWER_CTL, 0x00);
  delay(10);
  writeADXL(wire, ADXL345_DATA_FORMAT, 0x08);  // Full resolution, ±2g
  writeADXL(wire, ADXL345_BW_RATE, 0x0B);      // 200 Hz
  writeADXL(wire, ADXL345_POWER_CTL, 0x08);    // Measurement mode
  delay(10);
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

void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>SHM Monitor</title>";
  html += "<meta http-equiv='refresh' content='2'>";
  html += "<style>";
  html += "body{font-family:Arial;margin:20px;background:#f0f0f0;}";
  html += ".container{max-width:800px;margin:0 auto;background:white;padding:20px;border-radius:10px;}";
  html += "h1{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px;}";
  html += "table{width:100%;border-collapse:collapse;margin:20px 0;}";
  html += "th,td{border:1px solid #ddd;padding:10px;text-align:left;}";
  html += "th{background:#3498db;color:white;}";
  html += ".value{font-weight:bold;color:#2c3e50;}";
  html += "</style></head><body>";
  html += "<div class='container'>";
  html += "<h1>Structural Health Monitor</h1>";
  html += "<p>Sample Rate: 200 Hz | Samples: " + String(sample_count) + "</p>";
  
  html += "<h2>Current Readings (Ensemble Average)</h2>";
  html += "<table><tr><th>Axis</th><th>Acceleration (m/s²)</th><th>Acceleration (mg)</th></tr>";
  for (int axis = 0; axis < 3; axis++) {
    html += "<tr><td>" + String((char)('X' + axis)) + "</td>";
    html += "<td class='value'>" + String(accel_ensemble[axis] * 9.81, 3) + "</td>";
    html += "<td class='value'>" + String(accel_ensemble[axis] * 1000, 2) + "</td></tr>";
  }
  html += "</table>";
  
  html += "<h2>Individual Sensors</h2>";
  html += "<table><tr><th>Sensor</th><th>X (m/s²)</th><th>Y (m/s²)</th><th>Z (m/s²)</th></tr>";
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    html += "<tr><td>Sensor " + String(sensor + 1) + "</td>";
    for (int axis = 0; axis < 3; axis++) {
      html += "<td class='value'>" + String(accel_g[sensor][axis] * 9.81, 3) + "</td>";
    }
    html += "</tr>";
  }
  html += "</table>";
  
  html += "<p>Last update: " + String(millis()/1000) + " seconds</p>";
  html += "</div></body></html>";
  
  server.send(200, "text/html", html);
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
    initADXL(i);
    uint8_t devid;
    readADXL(*I2C_BUS[i], ADXL345_DEVID, &devid, 1);
    Serial.printf("Sensor %d: %s (ID: 0x%02X)\n", 
                  i+1, devid == 0xE5 ? "✓ OK" : "✗ Failed", devid);
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
