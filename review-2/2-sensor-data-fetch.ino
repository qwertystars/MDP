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
#define ADXL345_DEVID       0x00

bool initSensor(TwoWire &wire, const char* name) {
  // Validate DEVID register
  wire.beginTransmission(ADXL345_ADDRESS);
  wire.write(ADXL345_DEVID);
  uint8_t err = wire.endTransmission(false);
  if (err != 0) {
    Serial.printf("%s: I2C error %d during DEVID read\n", name, err);
    return false;
  }
  uint8_t count = wire.requestFrom(ADXL345_ADDRESS, (uint8_t)1);
  if (count != 1) {
    Serial.printf("%s: requestFrom returned %d bytes, expected 1\n", name, count);
    return false;
  }
  uint8_t devid = wire.read();
  if (devid != 0xE5) {
    Serial.printf("%s: unexpected DEVID 0x%02X (expected 0xE5)\n", name, devid);
    return false;
  }

  if (!writeByte(wire, ADXL345_ADDRESS, ADXL345_POWER_CTL, 0x08)) return false;
  if (!writeByte(wire, ADXL345_ADDRESS, ADXL345_DATA_FORMAT, 0x0B)) return false;
  Serial.printf("%s initialized (DEVID: 0x%02X)\n", name, devid);
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.println("Dual ADXL345 on Separate I2C Buses");

  // Initialize I2C bus 1 (GPIO 21, 22)
  I2C_1.begin(21, 22, 100000);

  // Initialize I2C bus 2 (GPIO 23, 18)
  I2C_2.begin(23, 18, 100000);

  delay(100);

  if (!initSensor(I2C_1, "Sensor 1")) {
    Serial.println("WARNING: Sensor 1 init failed!");
  }
  if (!initSensor(I2C_2, "Sensor 2")) {
    Serial.println("WARNING: Sensor 2 init failed!");
  }

  Serial.println("\nSetup complete!\n");
}

void loop() {
  // Read Sensor 1
  int16_t x1, y1, z1;
  if (readAccel(I2C_1, ADXL345_ADDRESS, &x1, &y1, &z1)) {
    Serial.print("Sensor 1 -> X: ");
    Serial.print(x1 * 0.0039 * 9.81, 2); // Convert to m/s²
    Serial.print(" \tY: ");
    Serial.print(y1 * 0.0039 * 9.81, 2);
    Serial.print(" \tZ: ");
    Serial.println(z1 * 0.0039 * 9.81, 2);
  }

  // Read Sensor 2
  int16_t x2, y2, z2;
  if (readAccel(I2C_2, ADXL345_ADDRESS, &x2, &y2, &z2)) {
    Serial.print("Sensor 2 -> X: ");
    Serial.print(x2 * 0.0039 * 9.81, 2);
    Serial.print(" \tY: ");
    Serial.print(y2 * 0.0039 * 9.81, 2);
    Serial.print(" \tZ: ");
    Serial.println(z2 * 0.0039 * 9.81, 2);
  }

  Serial.println("------------------------");
  delay(500);
}

// Write a byte to a register, returns true on success
bool writeByte(TwoWire &wire, uint8_t address, uint8_t reg, uint8_t data) {
  wire.beginTransmission(address);
  wire.write(reg);
  wire.write(data);
  uint8_t err = wire.endTransmission();
  if (err != 0) {
    Serial.printf("I2C write error %d (reg 0x%02X)\n", err, reg);
    return false;
  }
  return true;
}

// Read acceleration data, returns true on success
bool readAccel(TwoWire &wire, uint8_t address, int16_t *x, int16_t *y, int16_t *z) {
  wire.beginTransmission(address);
  wire.write(ADXL345_DATAX0);
  uint8_t err = wire.endTransmission(false);
  if (err != 0) {
    Serial.printf("I2C read error %d\n", err);
    return false;
  }

  uint8_t count = wire.requestFrom(address, (uint8_t)6);
  if (count != 6) {
    Serial.printf("I2C: expected 6 bytes, got %d\n", count);
    return false;
  }

  *x = wire.read() | (wire.read() << 8);
  *y = wire.read() | (wire.read() << 8);
  *z = wire.read() | (wire.read() << 8);
  return true;
}
