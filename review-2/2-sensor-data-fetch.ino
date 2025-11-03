#include <Wire.h>

// Create two I2C buses
TwoWire I2C_1 = TwoWire(0);
TwoWire I2C_2 = TwoWire(1);

// ADXL345 I2C address
#define ADXL345_ADDRESS 0x53

// ADXL345 Registers
#define ADXL345_POWER_CTL   0x2D
#define ADXL345_DATAX0      0x32
#define ADXL345_DATA_FORMAT 0x31

void setup() {
  Serial.begin(115200);
  Serial.println("Dual ADXL345 on Separate I2C Buses");
  
  // Initialize I2C bus 1 (GPIO 21, 22)
  I2C_1.begin(21, 22, 100000);
  
  // Initialize I2C bus 2 (GPIO 23, 18)
  I2C_2.begin(23, 18, 100000);
  
  delay(100);
  
  // Initialize Sensor 1
  writeByte(I2C_1, ADXL345_ADDRESS, ADXL345_POWER_CTL, 0x08);   // Power on
  writeByte(I2C_1, ADXL345_ADDRESS, ADXL345_DATA_FORMAT, 0x0B); // Full resolution, ±16g
  Serial.println("Sensor 1 initialized!");
  
  // Initialize Sensor 2
  writeByte(I2C_2, ADXL345_ADDRESS, ADXL345_POWER_CTL, 0x08);   // Power on
  writeByte(I2C_2, ADXL345_ADDRESS, ADXL345_DATA_FORMAT, 0x0B); // Full resolution, ±16g
  Serial.println("Sensor 2 initialized!");
  
  Serial.println("\nBoth sensors ready!\n");
}

void loop() {
  // Read Sensor 1
  int16_t x1, y1, z1;
  readAccel(I2C_1, ADXL345_ADDRESS, &x1, &y1, &z1);
  
  Serial.print("Sensor 1 -> X: ");
  Serial.print(x1 * 0.0039 * 9.81, 2); // Convert to m/s²
  Serial.print(" \tY: ");
  Serial.print(y1 * 0.0039 * 9.81, 2);
  Serial.print(" \tZ: ");
  Serial.println(z1 * 0.0039 * 9.81, 2);
  
  // Read Sensor 2
  int16_t x2, y2, z2;
  readAccel(I2C_2, ADXL345_ADDRESS, &x2, &y2, &z2);
  
  Serial.print("Sensor 2 -> X: ");
  Serial.print(x2 * 0.0039 * 9.81, 2);
  Serial.print(" \tY: ");
  Serial.print(y2 * 0.0039 * 9.81, 2);
  Serial.print(" \tZ: ");
  Serial.println(z2 * 0.0039 * 9.81, 2);
  
  Serial.println("------------------------");
  delay(500);
}

// Write a byte to a register
void writeByte(TwoWire &wire, uint8_t address, uint8_t reg, uint8_t data) {
  wire.beginTransmission(address);
  wire.write(reg);
  wire.write(data);
  wire.endTransmission();
}

// Read acceleration data
void readAccel(TwoWire &wire, uint8_t address, int16_t *x, int16_t *y, int16_t *z) {
  wire.beginTransmission(address);
  wire.write(ADXL345_DATAX0);
  wire.endTransmission(false);
  
  wire.requestFrom(address, (uint8_t)6);
  
  *x = wire.read() | (wire.read() << 8);
  *y = wire.read() | (wire.read() << 8);
  *z = wire.read() | (wire.read() << 8);
}
