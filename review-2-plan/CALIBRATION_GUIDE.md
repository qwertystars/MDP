# Sensor Calibration Guide
## Step-by-Step Procedure for MPU-6050 Array Calibration

---

## Overview

Proper calibration is **critical** for accurate structural health monitoring. This guide covers:
1. Zero-g offset calibration (automatic in firmware)
2. Multi-point temperature calibration (optional but recommended)
3. Validation and testing procedures

---

## Part 1: Basic Zero-G Calibration (Required)

### Equipment Needed
- Assembled sensor array
- Level surface
- Serial monitor (Arduino IDE or similar)

### Procedure

**Step 1: Pre-Installation Check**
```
Before mounting sensors on building:
1. Place sensor array on flat, level surface
2. Ensure surface is stable (not on table that wobbles)
3. Verify no vibration sources nearby (machinery, foot traffic)
4. Allow 10 minutes for temperature stabilization
```

**Step 2: Run Auto-Calibration**
```
1. Power on ESP32
2. Open Serial Monitor (115200 baud)
3. Wait for calibration sequence to start automatically
4. System will display:
   
   === Starting Calibration ===
   Keep sensors stationary for 10 seconds...
   ..........
   
5. Do NOT move or touch sensors during this time!
```

**Step 3: Verify Results**
```
System will display offsets for each sensor:

Sensor 0 offsets: X=0.0123, Y=-0.0087, Z=0.9876 g, Temp=24.5°C
Sensor 1 offsets: X=-0.0098, Y=0.0134, Z=0.9891 g, Temp=24.7°C
Sensor 2 offsets: X=0.0156, Y=-0.0076, Z=0.9823 g, Temp=24.3°C

=== Calibration Complete ===

Expected values:
✓ X, Y offsets: -0.05 to +0.05 g
✓ Z offset: 0.95 to 1.05 g (1g ± sensor noise)
✓ Temperature: Ambient temp ± 2°C

⚠️ If offsets exceed ±0.1g, check:
   - Surface is truly level
   - Sensors are not damaged
   - No vibrations during calibration
```

**Step 4: Mount Sensors**
```
After successful calibration:
1. Transport carefully to installation location
2. Mount using structural epoxy or fasteners
3. Do NOT power off ESP32 (calibration stored in RAM)
4. Allow 24 hours for epoxy cure if applicable
```

---

## Part 2: Multi-Point Temperature Calibration (Recommended)

**Why?** Reduces temperature drift from ±35mg to ±2-3mg

### Equipment Needed
- Ice bath (ice + water in container)
- Heating pad or warm environment (40°C)
- Thermometer
- Notebook for recording data
- Patience (2-3 hours total)

### Procedure

**Calibration Point 1: Cold (0°C)**
```
1. Prepare ice bath:
   - Mix ice and water thoroughly
   - Verify temperature ~0°C with thermometer
   - Let mixture stabilize for 5 minutes

2. Place sensor array in waterproof bag
   - Use Ziploc or similar
   - Remove air to ensure good thermal contact
   - Seal completely

3. Submerge in ice bath
   - Ensure sensors fully immersed
   - Wait 10 minutes for thermal equilibrium
   - Monitor temperature sensor readings

4. Record 100 samples:
   - Open Serial Monitor
   - Copy/paste 100 lines of data
   - Calculate average X, Y, Z for each sensor
   - Note: Temperature should read ~0°C

Example data to record:
Temp: 0.5°C
Sensor 0: X=-0.0452, Y=-0.0381, Z=1.0624
Sensor 1: X=-0.0398, Y=0.0214, Z=1.0358
Sensor 2: X=-0.0423, Y=-0.0176, Z=1.0781
```

**Calibration Point 2: Room Temperature (20°C)**
```
1. Remove from ice bath
2. Dry thoroughly
3. Place in room temperature environment
4. Wait 15 minutes for stabilization
5. Record 100 samples as above

Example:
Temp: 20.3°C
Sensor 0: X=-0.0125, Y=-0.0083, Z=0.9987
Sensor 1: X=-0.0098, Y=0.0134, Z=0.9891
Sensor 2: X=-0.0156, Y=-0.0076, Z=0.9823
```

**Calibration Point 3: Warm (40°C)**
```
1. Place on heating pad set to 40°C
   OR place in warm location (direct sunlight, oven at 40°C)
2. Monitor temperature sensor reading
3. Wait 15 minutes after temp reaches 40°C
4. Record 100 samples

Example:
Temp: 40.1°C
Sensor 0: X=0.0186, Y=0.0214, Z=0.9358
Sensor 1: X=0.0142, Y=0.0348, Z=0.9487
Sensor 2: X=0.0203, Y=0.0098, Z=0.9215
```

**Optional Point 4: Hot (60°C)**
```
Only if sensors will experience >50°C in operation:
- Use oven set to 60°C
- Same procedure as above
- Record after 15 min stabilization
```

### Data Processing

**Calculate Temperature Coefficients:**

```python
# Example Python script
import numpy as np

# Your recorded data
temps = [0, 20, 40]  # Celsius
sensor0_x = [-0.0452, -0.0125, 0.0186]  # Offsets at each temp

# Fit polynomial (2nd order recommended)
coeffs = np.polyfit(temps, sensor0_x, deg=2)
print(f"Sensor 0 X-axis coefficients: {coeffs}")

# Repeat for each sensor, each axis (9 total: 3 sensors × 3 axes)

# Update firmware with coefficients:
# In esp32_3sensor_array.ino, find:
float temp_coeff[NUM_SENSORS][3] = {
  {a2_s0x, a2_s0y, a2_s0z},  // Sensor 0
  {a2_s1x, a2_s1y, a2_s1z},  // Sensor 1  
  {a2_s2x, a2_s2y, a2_s2z}   // Sensor 2
};

// Replace with your calculated coefficients
```

**Or use automated script:**
```bash
python process_calibration.py --input calibration_data.csv --output coefficients.h
```

---

## Part 3: Validation Testing

### Test 1: Repeatability Check

**Purpose:** Verify calibration is stable

**Procedure:**
```
1. Mount sensor array on stable surface
2. Record 1000 samples (5 seconds at 200 Hz)
3. Calculate RMS for each axis
4. Repeat 3 times

Expected:
✓ RMS variation between runs: <5%
✓ Mean values within ±2mg of zero (X,Y) or ±20mg of 1g (Z)

If failed:
- Recalibrate
- Check mounting stability
- Verify no temperature changes during test
```

### Test 2: Known Excitation Response

**Purpose:** Verify frequency response is correct

**Procedure:**
```
1. Mount sensor array on test structure (or hold rigidly)
2. Apply known excitation:
   Option A: Heel drop test
   - Stand on structure
   - Rise on toes
   - Drop heels suddenly
   - Should see damped oscillation
   
   Option B: Impact test
   - Strike structure with rubber mallet
   - Single sharp impact
   - Observe ringdown

3. Analyze FFT:
   - Identify dominant frequency
   - Compare to expected (use online calculators or FEM)
   
Expected:
✓ Clear frequency peaks visible
✓ Frequency within 10% of theoretical
✓ Exponential decay after excitation

If failed:
- Check sampling rate settings
- Verify FFT window functions
- Ensure rigid mounting
```

### Test 3: Ensemble Averaging Verification

**Purpose:** Confirm noise reduction from multiple sensors

**Procedure:**
```
1. Record 10 seconds of stationary data
2. Calculate RMS for:
   - Each individual sensor
   - Ensemble average
3. Compare noise levels

Expected:
✓ Ensemble RMS < individual sensor RMS
✓ Improvement factor: 1.5-2× (for 3 sensors)

Example:
Sensor 0 RMS: 4.8 mg
Sensor 1 RMS: 5.2 mg
Sensor 2 RMS: 4.6 mg
Ensemble RMS: 2.9 mg → 1.7× improvement ✓

If failed:
- Check sensor wiring/connections
- Verify sensors are same model/revision
- Recalibrate individual sensors
```

---

## Part 4: Field Calibration After Installation

### Initial Site Baseline

**Required before operation begins**

```
Timeline: 24-48 hours continuous recording

Procedure:
1. Mount sensors permanently on structure
2. Power on system
3. Run auto-calibration (will adjust for mounting orientation)
4. Record continuously for 24-48 hours
5. System will establish:
   - Baseline RMS levels for each axis
   - Dominant structural frequencies  
   - Temperature variation range
   - Normal vibration patterns (daily cycles)

Critical: Structure must be in "healthy" state
- No known damage
- No construction activity
- Typical occupancy/operations
- Representative weather conditions
```

### Validation Against Professional Sensor (Optional)

**Gold standard calibration check**

```
Equipment: Rent/borrow professional accelerometer
Examples:
- PCB Piezotronics 603C01 ($100-200/day rental)
- Brüel & Kjær portable accelerometer
- Any sensor with <10 µg/√Hz noise

Procedure:
1. Co-locate professional sensor with your array
2. Record simultaneously for 2-7 days
3. Compare:
   - Dominant frequencies (should match within 2%)
   - RMS levels (should match within 15%)
   - Time-domain waveforms (correlation >0.7)
   - Modal parameters (if analysis performed)

If deviations exceed these ranges:
→ Your system may not be suitable for this structure
→ Consider adding more sensors or upgrading to ADXL355
→ Implement advanced error reduction techniques
```

---

## Part 5: Ongoing Calibration Maintenance

### Monthly Checks

```
Quick validation (30 minutes):

1. Visual inspection:
   ✓ Sensors securely mounted
   ✓ No physical damage
   ✓ Cables intact
   ✓ Enclosure sealed

2. Data quality check:
   ✓ No sensor dropouts in logs
   ✓ Temperature readings reasonable
   ✓ RMS levels within 2× baseline
   ✓ Dominant frequency within 5% of baseline

3. If temperature varied >20°C since last calibration:
   → Run zero-g calibration again
   → Update baseline statistics
```

### Quarterly Recalibration

```
Full procedure (2-3 hours):

1. Download all data from SD card
2. Analyze trends:
   - Has baseline drifted?
   - New vibration sources?
   - Seasonal effects?

3. Re-run zero-g calibration
4. Update baseline statistics
5. Validate against historical data
6. Document any changes in log book

Generate calibration report:
- Date
- Temperature range during calibration
- New vs old offsets
- Reason for recalibration
- Any adjustments made
```

### Annual Full Calibration

```
Complete validation (1 day):

1. Multi-point temperature calibration (repeat Part 2)
2. Validation testing (repeat Part 3)
3. Professional sensor comparison (if budget allows)
4. Update all calibration coefficients in firmware
5. Re-establish baseline if building conditions changed
6. Generate comprehensive calibration certificate

Documentation:
- Calibration procedures followed
- All test results
- Comparison to previous year
- System performance assessment
- Recommendations for next year
```

---

## Troubleshooting Calibration Issues

### Issue: Large Offset Values (>0.1g)

**Possible Causes:**
- Surface not level
- Sensor damaged
- Electrical interference
- Software bug

**Solutions:**
1. Verify surface with spirit level
2. Try different sensor from same batch
3. Move away from power supplies, motors
4. Update firmware to latest version
5. Try manual calibration (measure in all 6 orientations)

### Issue: High Temperature Sensitivity

**Possible Causes:**
- Inadequate thermal calibration
- Poor thermal coupling with structure
- Direct sunlight on sensors
- HVAC vents nearby

**Solutions:**
1. Perform 3-4 point temperature calibration
2. Ensure good thermal contact with structure
3. Add thermal insulation/enclosure
4. Relocate sensors away from thermal extremes
5. Implement advanced thermal compensation (see NOVEL_ERROR_REDUCTION.md)

### Issue: Inconsistent Results Between Sensors

**Possible Causes:**
- Manufacturing variation
- Different mounting conditions
- One sensor faulty
- Electrical noise on one channel

**Solutions:**
1. Perform individual sensor characterization
2. Ensure identical mounting for all sensors
3. Replace suspect sensor ($2-6 cost)
4. Check I2C pull-up resistors
5. Use shielded cables if EMI suspected

### Issue: Drift Over Time

**Possible Causes:**
- Temperature cycling effects
- Mechanical stress on solder joints
- Component aging
- Baseline shift due to building changes

**Solutions:**
1. More frequent recalibration (monthly vs quarterly)
2. Strain relief on cables
3. Accept 1-2% drift as normal for consumer sensors
4. Update baseline if structure legitimately changed
5. Consider upgrade to ADXL355 if drift unacceptable

---

## Calibration Best Practices

### DO:
✅ Calibrate in environment similar to deployment  
✅ Allow adequate thermal stabilization time
✅ Record calibration data for future reference
✅ Perform validation testing after calibration
✅ Recalibrate after any physical changes
✅ Document everything

### DON'T:
❌ Calibrate with sensors vibrating
❌ Skip temperature calibration for variable environments
❌ Assume all sensors from same batch are identical
❌ Forget to update baseline after recalibration
❌ Ignore calibration drift warnings
❌ Calibrate only once and never again

---

## Summary Checklist

**Initial Calibration:**
- [ ] Zero-g calibration complete
- [ ] Temperature calibration performed (if temp varies >20°C)
- [ ] Validation testing passed
- [ ] Sensors mounted correctly
- [ ] 24-48 hour baseline established
- [ ] All data documented

**Ongoing Maintenance:**
- [ ] Monthly visual inspections
- [ ] Quarterly recalibration
- [ ] Annual full calibration
- [ ] Professional validation (if available)
- [ ] Calibration records maintained

**Quality Metrics:**
- [ ] Individual sensor RMS: <6mg
- [ ] Ensemble average RMS: <3mg
- [ ] Temperature drift: <20mg over operating range
- [ ] Frequency accuracy: Within 2% of reference
- [ ] Sensor-to-sensor consistency: Within 20%

---

## Additional Resources

**Calibration Tools:**
- Six-position tumble test jig (DIY plans available online)
- Temperature chamber (or improvise with ice bath + heating pad)
- Reference accelerometer rental services
- Vibration calibrators (expensive but precise)

**Online Calculators:**
- Building natural frequency estimator
- Accelerometer noise calculator
- Statistical calibration processors

**Further Reading:**
- IEEE standards for accelerometer calibration
- ASTM E1318: Standard Specification for Highway Weigh-In-Motion Systems
- Manufacturer datasheets (MPU-6050, ADXL345, ADXL355)

---

**Ready to calibrate?** Follow the steps carefully, document everything, and your system will provide reliable structural monitoring for years to come!
