/*
 * Structural Health Monitoring System - ESP32 3-Sensor Array
 * MPU-6050 MEMS Accelerometer Array with Ensemble Averaging
 * 
 * Hardware:
 * - ESP32 DevKit
 * - 3x MPU-6050 accelerometers
 * - TCA9548A I2C multiplexer (optional, or use address pins)
 * - MicroSD card module for data logging
 * 
 * Features:
 * - 200 Hz synchronized sampling across all 3 sensors
 * - Real-time ensemble averaging for noise reduction
 * - Temperature compensation
 * - FFT analysis with frequency shift detection
 * - Anomaly detection with configurable thresholds
 * - WiFi data transmission capability
 */

#include <Wire.h>
#include <SD.h>
#include <WiFi.h>
#include <WebServer.h>
#include "arduinoFFT.h"

// ==================== CONFIGURATION ====================
#define NUM_SENSORS 3
#define SAMPLE_RATE 200  // Hz
#define FFT_SIZE 1024
#define I2C_SDA 21
#define I2C_SCL 22
#define SD_CS 5

// MPU-6050 I2C addresses (using AD0 pin: LOW=0x68, HIGH=0x69)
const uint8_t MPU_ADDR[NUM_SENSORS] = {0x68, 0x69, 0x68};  // Third uses multiplexer
#define USE_MULTIPLEXER true
#define TCA9548A_ADDR 0x70

// Accelerometer configuration
#define MPU_ACCEL_RANGE 0  // 0=±2g, 1=±4g, 2=±8g, 3=±16g
#define ACCEL_SENSITIVITY 16384.0  // LSB/g for ±2g range

// Detection thresholds
#define RMS_THRESHOLD_FACTOR 1.5   // 150% of baseline
#define FREQ_SHIFT_THRESHOLD 0.02  // 2% frequency shift
#define KURTOSIS_THRESHOLD 5.0     // Early fault indicator

// WiFi credentials (optional)
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ==================== GLOBAL VARIABLES ====================
// Sensor data buffers
int16_t accel_raw[NUM_SENSORS][3];  // [sensor][x,y,z]
float accel_g[NUM_SENSORS][3];      // Converted to g
float accel_ensemble[3];             // Ensemble averaged
float temperature[NUM_SENSORS];

// Calibration offsets
float accel_offset[NUM_SENSORS][3] = {0};
float temp_offset[NUM_SENSORS] = {0};
float temp_coeff[NUM_SENSORS][3] = {0};  // Temperature compensation coefficients

// FFT buffers
double fft_input[FFT_SIZE];
double fft_output[FFT_SIZE];
arduinoFFT FFT = arduinoFFT(fft_input, fft_output, FFT_SIZE, SAMPLE_RATE);

// Baseline statistics (set during calibration)
float baseline_rms[3] = {0};
float baseline_freq[3] = {0};
float baseline_kurtosis[3] = {3.0};

// Data logging
File dataFile;
unsigned long sample_count = 0;
unsigned long last_sample_time = 0;

// Status flags
bool sensors_initialized = false;
bool calibration_complete = false;
bool sd_available = false;
bool wifi_connected = false;

WebServer server(80);

// ==================== HELPER FUNCTIONS ====================

void selectMuxChannel(uint8_t channel) {
  if (!USE_MULTIPLEXER) return;
  if (channel > 7) return;
  Wire.beginTransmission(TCA9548A_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void writeMPU(uint8_t sensor_id, uint8_t reg, uint8_t data) {
  if (USE_MULTIPLEXER && sensor_id == 2) {
    selectMuxChannel(sensor_id);
  }
  Wire.beginTransmission(MPU_ADDR[sensor_id]);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission(true);
}

void readMPU(uint8_t sensor_id, uint8_t reg, uint8_t* buffer, uint8_t len) {
  if (USE_MULTIPLEXER && sensor_id == 2) {
    selectMuxChannel(sensor_id);
  }
  Wire.beginTransmission(MPU_ADDR[sensor_id]);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR[sensor_id], len);
  for (uint8_t i = 0; i < len; i++) {
    buffer[i] = Wire.read();
  }
}

void initMPU(uint8_t sensor_id) {
  // Wake up MPU-6050
  writeMPU(sensor_id, 0x6B, 0x00);
  delay(10);
  
  // Configure accelerometer range (±2g for maximum sensitivity)
  writeMPU(sensor_id, 0x1C, MPU_ACCEL_RANGE << 3);
  
  // Configure DLPF for 94 Hz cutoff (good for 200 Hz sampling)
  writeMPU(sensor_id, 0x1A, 0x02);
  
  // Set sample rate divider (1 kHz / (1 + divider))
  // For 200 Hz: divider = 4
  writeMPU(sensor_id, 0x19, 0x04);
  
  delay(10);
}

void readSensorData(uint8_t sensor_id) {
  uint8_t buffer[14];
  readMPU(sensor_id, 0x3B, buffer, 14);
  
  // Parse accelerometer data
  accel_raw[sensor_id][0] = (int16_t)((buffer[0] << 8) | buffer[1]);
  accel_raw[sensor_id][1] = (int16_t)((buffer[2] << 8) | buffer[3]);
  accel_raw[sensor_id][2] = (int16_t)((buffer[4] << 8) | buffer[5]);
  
  // Parse temperature (TEMP_OUT_H, TEMP_OUT_L)
  int16_t temp_raw = (int16_t)((buffer[6] << 8) | buffer[7]);
  temperature[sensor_id] = temp_raw / 340.0 + 36.53;
  
  // Convert to g with calibration and temperature compensation
  for (int axis = 0; axis < 3; axis++) {
    float accel_cal = accel_raw[sensor_id][axis] / ACCEL_SENSITIVITY - accel_offset[sensor_id][axis];
    float temp_comp = temp_coeff[sensor_id][axis] * (temperature[sensor_id] - temp_offset[sensor_id]);
    accel_g[sensor_id][axis] = accel_cal - temp_comp;
  }
}

void computeEnsembleAverage() {
  // Ensemble averaging: reduces noise by factor of √N
  // Outlier rejection: Remove sensor if >3σ from median
  
  for (int axis = 0; axis < 3; axis++) {
    float values[NUM_SENSORS];
    for (int i = 0; i < NUM_SENSORS; i++) {
      values[i] = accel_g[i][axis];
    }
    
    // Calculate median and MAD (Median Absolute Deviation)
    float sorted[NUM_SENSORS];
    memcpy(sorted, values, sizeof(values));
    
    // Simple bubble sort for small array
    for (int i = 0; i < NUM_SENSORS - 1; i++) {
      for (int j = 0; j < NUM_SENSORS - i - 1; j++) {
        if (sorted[j] > sorted[j + 1]) {
          float temp = sorted[j];
          sorted[j] = sorted[j + 1];
          sorted[j + 1] = temp;
        }
      }
    }
    
    float median = sorted[NUM_SENSORS / 2];
    
    // Calculate MAD
    float deviations[NUM_SENSORS];
    for (int i = 0; i < NUM_SENSORS; i++) {
      deviations[i] = abs(values[i] - median);
    }
    
    // Sort deviations
    for (int i = 0; i < NUM_SENSORS - 1; i++) {
      for (int j = 0; j < NUM_SENSORS - i - 1; j++) {
        if (deviations[j] > deviations[j + 1]) {
          float temp = deviations[j];
          deviations[j] = deviations[j + 1];
          deviations[j + 1] = temp;
        }
      }
    }
    
    float mad = deviations[NUM_SENSORS / 2];
    float threshold = median + 3.0 * 1.4826 * mad;  // 3σ equivalent
    
    // Average valid sensors only
    float sum = 0;
    int count = 0;
    for (int i = 0; i < NUM_SENSORS; i++) {
      if (abs(values[i] - median) <= threshold) {
        sum += values[i];
        count++;
      }
    }
    
    accel_ensemble[axis] = (count > 0) ? sum / count : median;
  }
}

float calculateRMS(float* data, int length) {
  float sum_sq = 0;
  for (int i = 0; i < length; i++) {
    sum_sq += data[i] * data[i];
  }
  return sqrt(sum_sq / length);
}

float calculateKurtosis(float* data, int length) {
  float mean = 0;
  for (int i = 0; i < length; i++) {
    mean += data[i];
  }
  mean /= length;
  
  float m2 = 0, m4 = 0;
  for (int i = 0; i < length; i++) {
    float diff = data[i] - mean;
    m2 += diff * diff;
    m4 += diff * diff * diff * diff;
  }
  m2 /= length;
  m4 /= length;
  
  return (m2 > 0) ? m4 / (m2 * m2) : 3.0;
}

void performFFT(float* data, int length, float& peak_freq, float& peak_amplitude) {
  // Copy data to FFT buffer and apply Hanning window
  for (int i = 0; i < FFT_SIZE; i++) {
    if (i < length) {
      float window = 0.5 * (1.0 - cos(2.0 * PI * i / (FFT_SIZE - 1)));
      fft_input[i] = data[i] * window;
    } else {
      fft_input[i] = 0;
    }
    fft_output[i] = 0;
  }
  
  // Compute FFT
  FFT.Windowing(FFT_HANN, FFT_FORWARD);
  FFT.Compute(FFT_FORWARD);
  FFT.ComplexToMagnitude();
  
  // Find peak frequency (ignore DC component)
  peak_amplitude = 0;
  int peak_index = 0;
  for (int i = 2; i < FFT_SIZE / 2; i++) {  // Start at 2 to skip DC and first bin
    if (fft_output[i] > peak_amplitude) {
      peak_amplitude = fft_output[i];
      peak_index = i;
    }
  }
  
  peak_freq = (peak_index * SAMPLE_RATE) / (float)FFT_SIZE;
}

// ==================== CALIBRATION ====================

void performCalibration() {
  Serial.println("\n=== Starting Calibration ===");
  Serial.println("Keep sensors stationary for 10 seconds...");
  
  const int CAL_SAMPLES = 2000;  // 10 seconds at 200 Hz
  float sum[NUM_SENSORS][3] = {0};
  float temp_sum[NUM_SENSORS] = {0};
  
  for (int sample = 0; sample < CAL_SAMPLES; sample++) {
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
      readSensorData(sensor);
      for (int axis = 0; axis < 3; axis++) {
        sum[sensor][axis] += accel_raw[sensor][axis] / ACCEL_SENSITIVITY;
      }
      temp_sum[sensor] += temperature[sensor];
    }
    delay(5);  // 200 Hz = 5ms
    
    if (sample % 200 == 0) {
      Serial.print(".");
    }
  }
  
  Serial.println("\nCalculating offsets...");
  
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    for (int axis = 0; axis < 3; axis++) {
      accel_offset[sensor][axis] = sum[sensor][axis] / CAL_SAMPLES;
      // Subtract gravity from Z-axis (assuming sensor mounted horizontally)
      if (axis == 2) {
        accel_offset[sensor][axis] -= 1.0;  // Remove 1g
      }
    }
    temp_offset[sensor] = temp_sum[sensor] / CAL_SAMPLES;
    
    Serial.printf("Sensor %d offsets: X=%.4f, Y=%.4f, Z=%.4f g, Temp=%.2f°C\n",
                  sensor, accel_offset[sensor][0], accel_offset[sensor][1], 
                  accel_offset[sensor][2], temp_offset[sensor]);
  }
  
  calibration_complete = true;
  Serial.println("=== Calibration Complete ===\n");
}

void establishBaseline() {
  Serial.println("\n=== Establishing Baseline ===");
  Serial.println("Recording 30 seconds of normal vibration...");
  
  const int BASELINE_SAMPLES = 6000;  // 30 seconds at 200 Hz
  float rms_samples[BASELINE_SAMPLES][3];
  
  for (int sample = 0; sample < BASELINE_SAMPLES; sample++) {
    // Read all sensors
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
      readSensorData(sensor);
    }
    
    // Compute ensemble average
    computeEnsembleAverage();
    
    // Store for RMS calculation
    for (int axis = 0; axis < 3; axis++) {
      rms_samples[sample][axis] = accel_ensemble[axis];
    }
    
    delay(5);
    
    if (sample % 600 == 0) {
      Serial.print(".");
    }
  }
  
  Serial.println("\nCalculating baseline statistics...");
  
  // Calculate RMS for each axis
  for (int axis = 0; axis < 3; axis++) {
    float axis_data[BASELINE_SAMPLES];
    for (int i = 0; i < BASELINE_SAMPLES; i++) {
      axis_data[i] = rms_samples[i][axis];
    }
    
    baseline_rms[axis] = calculateRMS(axis_data, BASELINE_SAMPLES);
    baseline_kurtosis[axis] = calculateKurtosis(axis_data, BASELINE_SAMPLES);
    
    // Perform FFT to find dominant frequency
    float peak_freq, peak_amp;
    performFFT(axis_data, BASELINE_SAMPLES, peak_freq, peak_amp);
    baseline_freq[axis] = peak_freq;
    
    Serial.printf("Axis %c: RMS=%.4f g, Freq=%.2f Hz, Kurtosis=%.2f\n",
                  'X' + axis, baseline_rms[axis], baseline_freq[axis], baseline_kurtosis[axis]);
  }
  
  Serial.println("=== Baseline Established ===\n");
}

// ==================== ANOMALY DETECTION ====================

struct AnomalyReport {
  bool detected;
  float rms_factor[3];
  float freq_shift[3];
  float kurtosis[3];
  char message[256];
};

AnomalyReport detectAnomalies(float* current_data, int length) {
  AnomalyReport report;
  report.detected = false;
  report.message[0] = '\0';
  
  char temp_msg[128];
  
  for (int axis = 0; axis < 3; axis++) {
    // Extract axis data
    float axis_data[length];
    for (int i = 0; i < length; i++) {
      axis_data[i] = current_data[i * 3 + axis];
    }
    
    // Calculate current statistics
    float current_rms = calculateRMS(axis_data, length);
    float current_kurtosis = calculateKurtosis(axis_data, length);
    
    float peak_freq, peak_amp;
    performFFT(axis_data, length, peak_freq, peak_amp);
    
    // Calculate factors
    report.rms_factor[axis] = current_rms / baseline_rms[axis];
    report.freq_shift[axis] = (peak_freq - baseline_freq[axis]) / baseline_freq[axis];
    report.kurtosis[axis] = current_kurtosis;
    
    // Check thresholds
    if (report.rms_factor[axis] > RMS_THRESHOLD_FACTOR) {
      report.detected = true;
      snprintf(temp_msg, sizeof(temp_msg), "[%c] RMS %.1fx baseline! ", 'X' + axis, report.rms_factor[axis]);
      strcat(report.message, temp_msg);
    }
    
    if (abs(report.freq_shift[axis]) > FREQ_SHIFT_THRESHOLD) {
      report.detected = true;
      snprintf(temp_msg, sizeof(temp_msg), "[%c] Freq shift %.1f%%! ", 'X' + axis, report.freq_shift[axis] * 100);
      strcat(report.message, temp_msg);
    }
    
    if (current_kurtosis > KURTOSIS_THRESHOLD) {
      report.detected = true;
      snprintf(temp_msg, sizeof(temp_msg), "[%c] Kurtosis %.2f! ", 'X' + axis, current_kurtosis);
      strcat(report.message, temp_msg);
    }
  }
  
  return report;
}

// ==================== WEB SERVER ====================

void handleRoot() {
  String html = "<html><head><title>SHM Monitor</title>";
  html += "<meta http-equiv='refresh' content='2'>";
  html += "<style>body{font-family:Arial;margin:20px;}";
  html += "table{border-collapse:collapse;width:100%;}";
  html += "th,td{border:1px solid #ddd;padding:8px;text-align:left;}";
  html += "th{background-color:#4CAF50;color:white;}";
  html += ".alert{color:red;font-weight:bold;}</style></head><body>";
  html += "<h1>Structural Health Monitor</h1>";
  html += "<p>Sample Rate: " + String(SAMPLE_RATE) + " Hz | Samples: " + String(sample_count) + "</p>";
  
  html += "<h2>Current Readings (Ensemble Average)</h2><table>";
  html += "<tr><th>Axis</th><th>Accel (mg)</th><th>RMS Factor</th><th>Freq Shift (%)</th><th>Kurtosis</th></tr>";
  
  for (int axis = 0; axis < 3; axis++) {
    html += "<tr><td>" + String((char)('X' + axis)) + "</td>";
    html += "<td>" + String(accel_ensemble[axis] * 1000, 2) + "</td>";
    html += "<td>-</td><td>-</td><td>-</td></tr>";
  }
  
  html += "</table>";
  
  html += "<h2>Individual Sensors</h2><table>";
  html += "<tr><th>Sensor</th><th>X (mg)</th><th>Y (mg)</th><th>Z (mg)</th><th>Temp (°C)</th></tr>";
  
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    html += "<tr><td>" + String(sensor) + "</td>";
    for (int axis = 0; axis < 3; axis++) {
      html += "<td>" + String(accel_g[sensor][axis] * 1000, 2) + "</td>";
    }
    html += "<td>" + String(temperature[sensor], 1) + "</td></tr>";
  }
  
  html += "</table></body></html>";
  
  server.send(200, "text/html", html);
}

// ==================== SETUP ====================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== Structural Health Monitoring System ===");
  Serial.println("ESP32 3-Sensor Array with Ensemble Averaging\n");
  
  // Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);  // 400 kHz Fast Mode
  
  // Initialize sensors
  Serial.println("Initializing sensors...");
  for (int i = 0; i < NUM_SENSORS; i++) {
    initMPU(i);
    Serial.printf("Sensor %d initialized at address 0x%02X\n", i, MPU_ADDR[i]);
  }
  sensors_initialized = true;
  
  // Initialize SD card
  Serial.println("\nInitializing SD card...");
  if (SD.begin(SD_CS)) {
    sd_available = true;
    Serial.println("SD card ready");
    
    // Create new log file
    char filename[32];
    int filenum = 0;
    do {
      snprintf(filename, sizeof(filename), "/shm_log_%03d.csv", filenum++);
    } while (SD.exists(filename));
    
    dataFile = SD.open(filename, FILE_WRITE);
    if (dataFile) {
      dataFile.println("timestamp,s0_x,s0_y,s0_z,s1_x,s1_y,s1_z,s2_x,s2_y,s2_z,ens_x,ens_y,ens_z,t0,t1,t2");
      dataFile.close();
      Serial.printf("Logging to: %s\n", filename);
    }
  } else {
    Serial.println("SD card initialization failed");
  }
  
  // Initialize WiFi (optional)
  Serial.println("\nConnecting to WiFi...");
  WiFi.begin(ssid, password);
  int wifi_attempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_attempts < 20) {
    delay(500);
    Serial.print(".");
    wifi_attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    wifi_connected = true;
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    
    server.on("/", handleRoot);
    server.begin();
    Serial.println("Web server started");
  } else {
    Serial.println("\nWiFi not connected (continuing without network)");
  }
  
  // Perform calibration
  delay(2000);
  performCalibration();
  
  // Establish baseline
  delay(2000);
  establishBaseline();
  
  Serial.println("\n=== System Ready ===");
  Serial.println("Monitoring started...\n");
  
  last_sample_time = micros();
}

// ==================== MAIN LOOP ====================

void loop() {
  unsigned long current_time = micros();
  unsigned long elapsed = current_time - last_sample_time;
  
  // Maintain 200 Hz sample rate (5000 microseconds per sample)
  if (elapsed >= 5000) {
    last_sample_time = current_time;
    
    // Read all sensors
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
      readSensorData(sensor);
    }
    
    // Compute ensemble average
    computeEnsembleAverage();
    
    sample_count++;
    
    // Log data every 100 samples (0.5 seconds)
    if (sd_available && sample_count % 100 == 0) {
      dataFile = SD.open("/shm_log_000.csv", FILE_APPEND);
      if (dataFile) {
        dataFile.printf("%lu,", millis());
        for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
          dataFile.printf("%.6f,%.6f,%.6f,", 
                         accel_g[sensor][0], accel_g[sensor][1], accel_g[sensor][2]);
        }
        dataFile.printf("%.6f,%.6f,%.6f,", 
                       accel_ensemble[0], accel_ensemble[1], accel_ensemble[2]);
        dataFile.printf("%.2f,%.2f,%.2f\n", 
                       temperature[0], temperature[1], temperature[2]);
        dataFile.close();
      }
    }
    
    // Perform anomaly detection every 1024 samples (~5 seconds)
    static float detection_buffer[FFT_SIZE * 3];
    static int buffer_index = 0;
    
    for (int axis = 0; axis < 3; axis++) {
      detection_buffer[buffer_index * 3 + axis] = accel_ensemble[axis];
    }
    buffer_index++;
    
    if (buffer_index >= FFT_SIZE) {
      AnomalyReport report = detectAnomalies(detection_buffer, FFT_SIZE);
      
      if (report.detected) {
        Serial.println("\n*** ANOMALY DETECTED ***");
        Serial.println(report.message);
        Serial.println("************************\n");
      }
      
      // Print status update
      Serial.printf("[%lu] RMS: X=%.2f Y=%.2f Z=%.2f mg | ", 
                    sample_count,
                    calculateRMS(&detection_buffer[0], FFT_SIZE) * 1000,
                    calculateRMS(&detection_buffer[1], FFT_SIZE) * 1000,
                    calculateRMS(&detection_buffer[2], FFT_SIZE) * 1000);
      Serial.printf("Temp: %.1f %.1f %.1f °C\n",
                    temperature[0], temperature[1], temperature[2]);
      
      buffer_index = 0;
    }
  }
  
  // Handle web server requests
  if (wifi_connected) {
    server.handleClient();
  }
}
