# HC-SR04 Ultrasonic Sensors for Indoor Positioning: Technical Feasibility Assessment

**Bottom line: HC-SR04 sensors are marginally feasible for coarse room-level positioning (±1-2m) but face severe practical limitations.** While triangulation with 3-4 sensors can theoretically achieve room-level accuracy, real-world performance is compromised by material absorption, interference management complexity, and environmental sensitivity. The sensors work adequately for simple obstacle detection but require extensive mitigation strategies and realistic expectations for positioning applications.

## Accuracy and precision: The specification gap

The manufacturer's ±3mm accuracy claim is **achievable only under ideal laboratory conditions** with static, hard, flat targets perpendicular to the sensor. Real-world performance tells a different story.

**Static targets** achieve **±1-2cm accuracy** in typical indoor environments with proper filtering. Independent testing across 5-200cm ranges showed relative errors of 0-5% and absolute errors of -0.5 to -1.5cm, with standard deviations of 0.1-0.5cm. One extensive test bluntly concluded performance is "worse in all regards than standard specifications."

**Moving targets face fundamental constraints.** The sensor's 60ms minimum measurement cycle (16.7 Hz maximum rate) creates tracking limitations. A target moving at 24 cm/s travels 14.4mm between readings—already exceeding the claimed 3mm accuracy. The 40 kHz wavelength (8.5mm) creates a theoretical resolution limit due to wave diffraction. For motion detection, minimum detectable velocities range from 3.7 to 67.7 cm/s depending on buffer size and filtering approach, with practical thresholds around 15 cm/s.

**Distance significantly affects accuracy.** Near range (20-100cm) delivers ±10mm typical performance. Mid-range (100-200cm) degrades to ±20mm. Long range (200-400cm) shows ±50-100mm errors or complete failures. The best controlled laboratory experiments—using custom 6-transducer triangulation systems—achieved 0.83-1.10mm standard deviation, but this required millimetric positioning precision and calibration impractical for most applications.

Temperature compensation is **critical and often ignored.** Sound speed changes 0.17% per degree Celsius, creating 1.8% measurement error across a 10°C temperature range. At 3m distance, this translates to 6cm error without compensation. Indoor temperature gradients from floor to ceiling can easily span 5-10°C, introducing systematic errors that overwhelm the sensor's theoretical precision.

## Maximum reliable range: The indoor reality check

While datasheets claim 2-400cm range, **practical indoor performance tops out at 10-250cm reliably.** The optimal sweet spot is 30-200cm, where performance is most consistent.

Target characteristics dominate range capabilities. Large, hard, flat targets enable detection up to 3-4m. Medium targets (24" × 48" flat panels) work reliably to approximately 2.77m. Small targets (1.3" × 1.5" aluminum) max out around 1.5-1.7m. Narrow objects like yardsticks manage roughly 2.45m maximum. **Human bodies**, covered in sound-absorbing clothing, become unreliable beyond 2m depending on approach angle and clothing material.

The **20cm near-field blind zone** is a critical limitation. Transducer ringing after the ultrasonic burst causes direct transmitter-to-receiver coupling for approximately 1.2ms, making measurements below 20cm unreliable despite the 2cm specification. Some implementations report practical minimums of 5-8cm even in best cases.

Indoor environments create **range-limiting acoustic challenges.** Sound absorption by air at 40 kHz increases with distance. Multipath reflections from walls consume signal energy. The inverse square law reduces echo amplitude rapidly. Testing in a residential garage showed reliable vehicle detection started only at 230cm (42% of the theoretical 550cm range to the back wall), with massive signal dropouts due to the curved bumper surface.

## Interference management: The simultaneous operation myth

**Four HC-SR04 sensors cannot operate simultaneously without sophisticated mitigation.** All HC-SR04 units transmit identical 8-cycle bursts at exactly 40 kHz with no frequency variability. They cannot distinguish their own echoes from other sensors' transmissions, creating guaranteed cross-talk interference.

**Time-division multiplexing is the only practical solution** for hobbyist and production applications. This requires triggering sensors sequentially with **minimum 29-60ms delays** between activations. The NewPing library, tested with 15-sensor arrays, recommends 29ms minimum spacing. Manufacturers specify 60ms measurement cycles. Real-world implementations typically use 33-50ms intervals as a reliability compromise.

**Update rate degrades linearly** with sensor count. Four sensors at 50ms intervals require 200ms total cycle time, yielding just 5 Hz system update rate. Eight sensors at 50ms intervals drop to 2.5 Hz. For robots moving at 1 m/s, a 200ms update cycle means the robot travels 20cm between complete sensor sweeps—introducing substantial position uncertainty.

Interference manifests in several ways. **Direct cross-talk** occurs when Sensor A's pulse triggers false echoes in Sensor B's receiver, causing readings of zero, incorrect distances, or complete sensor lockup. **Echo misattribution** happens when sensors receive the wrong echo from overlapping pulses. **Multipath reflections** in enclosed spaces cause ultrasonic energy to persist 10-100ms, creating confusion between current and previous measurements. Critically, **some low-quality HC-SR04 clones lack proper timeout implementation**—when no echo returns, the echo pin stays HIGH indefinitely, requiring power cycling to recover.

Spatial separation provides minor benefits but doesn't eliminate interference. Minimum physical spacing of 36-50mm between sensors helps reduce side-lobe coupling. Directional placement with sensors pointed 30°+ apart improves isolation. Acoustic shielding using cardboard tubes, metal baffles, or foam narrowly focuses beams. However, even with these measures, **sequential triggering remains mandatory.**

The research-grade solution—stochastic coding with adaptive filtering—enables true simultaneous operation by transmitting unique random-noise patterns per sensor and using DSP-based correlation filtering. Bosch demonstrated 8 sensors operating concurrently in automotive applications. This approach requires custom transducers, DSP processors, and complex algorithms, making it impractical for HC-SR04-based systems.

**Maximum practical sensor count**: 6-8 sensors for real-time robotics requiring 5-10 Hz updates, up to 10-12 for slower moving applications, and 15+ for stationary monitoring where 1-2 Hz suffices.

## Failure modes: When ultrasonic sensing breaks down

### Material absorption creates blind spots

**Soft materials represent complete detection failure.** Clothing, curtains, upholstered furniture, and foam absorb ultrasonic waves rather than reflecting them. Multiple sources document robots with HC-SR04 sensors crashing into fabric-covered sofas because the sensor literally cannot "see" these obstacles. Human detection becomes unreliable at best—people wearing heavy clothing may be invisible to the sensor while those in rain jackets reflect clearly.

The physics is unforgiving. Soft, porous materials have low acoustic impedance mismatch with air, causing sound energy to dissipate into the material structure rather than reflect. **Detection reliability by material type:**
- Hard perpendicular walls/metal/plastic: \>95% reliable
- Textured walls: 70-85% reliable  
- Wood furniture: 60-80% reliable
- Angled smooth surfaces (\>15°): \<30% reliable
- Fabric/foam: \<20% reliable, often 0%

**For indoor positioning, this means entire categories of room features are invisible**—fabric seating, window treatments, textile wall coverings, and clothing on people create unpredictable coverage gaps.

### Multipath reflections and wall interference

Indoor environments generate **complex multipath propagation** that creates false targets and positioning errors. Sound waves bounce off walls, ceilings, floors, and furniture before returning to the sensor, with some paths 2-4× longer than the direct path. Research in multipath environments documented **16-29mm RMSE positioning errors** even with advanced compensation algorithms.

**Corner reflections** are particularly problematic—sound bouncing wall-to-wall-to-sensor creates phantom objects at incorrect distances. Parallel walls generate reverberation and standing waves. Highly reflective surfaces like glass or metal furniture act as acoustic mirrors, reflecting sound at oblique angles or creating duplicate target artifacts.

The sensor's 60ms minimum cycle time stems partly from multipath decay requirements. Even in empty rooms, echoes persist \>1.2ms, and furnished indoor spaces require longer settling to avoid measuring residual energy from previous pulses.

### Angular limitations constrain coverage

The advertised 15° beam angle is **both optimistic and distance-dependent.** Comprehensive beam pattern testing reveals actual horizontal detection spans ~20° and vertical ~13° at optimal distances, but this varies dramatically with range.

**Detection angle by distance** (for perpendicular targets):
- 15-45cm: Drops rapidly from 20° to 9°
- 45-145cm: Relatively constant ~10°  
- 145-250cm: Gradually decreases at 0.06°/cm
- \>250cm: Narrows to \<5°, requiring precision alignment

**Range collapses catastrophically with angle.** At 30° off-axis, effective range drops from 3m to just 50cm—an 83% reduction. The sensor requires near-perpendicular alignment (within ±4° tolerance) for maximum-range performance. For indoor positioning, this means **walls or objects angled \>15° from perpendicular may not return echoes**, creating coverage gaps in room corners or near angled furniture.

### Specular reflection from angled surfaces

Smooth surfaces (walls, glass, metal) produce **specular reflections** like acoustic mirrors—sound reflects at an angle equal to incidence. If the surface isn't perpendicular to the sensor (±15° tolerance), the echo reflects away and is lost entirely. Even a 30° wall angle can prevent detection completely.

This contrasts with diffuse reflection from rough surfaces (\>8.5mm roughness, matching the 40 kHz wavelength), which scatter sound in multiple directions. Diffuse reflectors produce weaker but more omnidirectional echoes. Indoor environments mix both reflection types unpredictably, creating **object detection that varies based on approach angle and surface finish.**

Curved surfaces like car bumpers or rounded furniture present the worst case—they reflect sound tangentially rather than back toward the sensor, causing detection failures beyond 1-2m despite the objects being well within nominal range.

## ESP32 update rate and practical throughput

**Single-sensor maximum: 16.7 Hz** (60ms minimum cycle). The HC-SR04 requires a 10μs trigger pulse, transmits an 8-cycle burst (~300μs), waits for the echo (up to 23.5ms for 400cm max range), and needs echo decay time (~6-36ms). Total minimum cycle: 60ms per manufacturer recommendations, though 29ms is achievable with reduced maximum range.

**ESP32 integration** uses straightforward GPIO control. Typical implementations use 500ms intervals (2 Hz) in example code, though this is overly conservative. **Practical recommendations: 10 Hz (100ms intervals) for stable robotics applications**, balancing responsiveness with measurement reliability. The ESP32's 1μs timer resolution provides adequate precision for the ~58μs per centimeter pulse width.

**Critical implementation detail:** The standard pulseIn() function blocks code execution during measurement, preventing concurrent motor control or communications. Interrupt-driven approaches using timer capture enable non-blocking operation, crucial for real-time robotics.

**Multi-sensor arrays face arithmetic degradation.** Four sensors sequentially triggered at recommended 50ms spacing require 200ms total cycle time (5 Hz system update rate). Eight sensors demand 400ms (2.5 Hz). For a robot moving at 1 m/s with 200ms update cycles, position uncertainty reaches 20cm simply from measurement latency—already exceeding the ±1-2m room-level target but problematic for precise navigation.

## Real-world positioning implementations: Successes and instructive failures

### Two-sensor triangulation (LEGO Robotics project)

Configuration using 2 HC-SR04 sensors spaced 20.5cm apart on a motorized tracking platform achieved **±2-4cm positioning accuracy** at ranges up to approximately 2m using law-of-cosines triangulation. However, the system showed critical limitations: offset errors restricted sideways range to just 88mm, scale errors limited effective distance, and **numerical resolution constraints** from the sensor's 1cm resolution caused errors at measurement boundaries.

**Maximum tracking speed was limited to \<240 mm/s at 60 Hz** to prevent wavelength-jump errors. The motorized tracking setup (actively pointing sensors at target) outperformed fixed-mount configurations, suggesting active tracking compensates for angular sensitivity.

### Three-sensor triangulation (ROBIAN humanoid robot)

The most sophisticated implementation used **6 ultrasonic transducers** (3 emitters, 3 receivers) in equilateral triangle arrangements spaced 48mm apart. This custom system achieved **0.83-1.10mm standard deviation** for X, Y, Z axes with sub-millimeter static resolution and 60 Hz sampling frequency over 1-700mm working range.

**However, this is not a HC-SR04 system**—it employed specialized transducers with time-of-flight measurement across 9 emitter-receiver pairs, sophisticated triangulation algorithms, and precise mechanical mounting with millimetric calibration. The results demonstrate ultrasonic triangulation's theoretical potential but require engineering investment far beyond commodity HC-SR04 sensors.

### Four-sensor room mapping (Hackaday Project #19241)

This attempt **FAILED completely** to map room geometry despite using 4 HC-SR04 sensors with magnetic compass for 360° rotation. Instead of measuring square room dimensions, the system produced circular patterns because the **wide beam spread caused echoes from the nearest walls regardless of pointing direction.**

**Critical lesson:** Wide, distance-dependent beam patterns and multipath reflections make single-point ultrasonic sensors unsuitable for mapping without beam-forming arrays. The project documented specific failures including timeout inconsistencies (38ms vs 200ms actual), intermittent pulse misses, and walls \>15° off-perpendicular producing no response.

### Automotive parking detection (Comprehensive garage study)

Exhaustively documented parking assist system **ultimately failed** despite extensive testing and analysis. The curved vehicle bumper surface failed to reflect sufficient energy beyond 230cm (7.5 feet)—just 42% of the expected range to the garage back wall. Detection suffered massive dropouts until the vehicle straightened to near-perpendicular approach.

The project provided valuable quantitative data: **static target standard deviation of 0.56cm** for foam at 2.66m, but highly skewed (non-normal) distribution. The researcher's detailed beam spread analysis at 2.66m showed detection angle maximum of just 15-20° with rapid narrowing at longer distances. Curved surfaces proved fundamentally incompatible with HC-SR04 sensors beyond close range.

### PIR sensor integration for presence detection

Multiple implementations successfully combine PIR motion sensors with HC-SR04 distance measurement for **presence detection and security applications.** The hybrid approach leverages complementary strengths: PIR triggers on infrared motion (working through clothing and temperature changes), then HC-SR04 confirms distance and provides spatial localization.

Typical configuration: PIR continuously monitors for motion, triggering HC-SR04 distance measurement only when motion detected. If distance \<200cm threshold, alarm activates. This reduces false positives from both sensors while minimizing HC-SR04 power consumption. PIR's wide 90-120° detection angle compensates for HC-SR04's narrow beam, while ultrasonic distance measurement overcomes PIR's limitation to binary presence without range information.

**Human detection reliability**: \>95% detection probability at \<3m for PIR+HC-SR04 combined systems, compared to 50-80% for HC-SR04 alone due to clothing absorption issues. The PIR's immunity to acoustic properties makes it the superior primary motion detector, with ultrasonics providing secondary distance validation.

## Feasibility assessment: Can triangulation achieve ±1-2m room-level accuracy?

**Yes, but with substantial caveats and system complexity.** The theoretical answer is clearly positive—documented implementations achieve centimeter-level accuracy. The practical answer depends heavily on environmental control and expectations management.

### Technical feasibility considerations

**Minimum sensor array: 3-4 HC-SR04 sensors** for 2D room-level positioning. Sensors must be wall-mounted or ceiling-mounted at known positions, covering the tracking area with overlapping detection zones. **Recommended spacing: 40-50mm between sensors** with directional arrangements to minimize interference.

**Expected positioning accuracy** with realistic implementation:
- **Best case (controlled environment, hard targets, averaging):** ±5-10cm
- **Typical case (furnished room, mixed targets):** ±10-30cm  
- **Worst case (soft furnishings, multipath, poor calibration):** ±50cm-1m or failures

Room-level positioning of **±1-2m accuracy is achievable** but represents relatively coarse localization—effectively determining which quadrant or zone of a room contains the target rather than precise coordinates.

### Required mitigation strategies

**Time-division multiplexing** with 50-60ms sensor spacing reduces system update rate to 5 Hz for 4 sensors, introducing 200ms latency. A person walking at 1.5 m/s moves 30cm between updates, requiring motion prediction or Kalman filtering to compensate.

**Temperature compensation is essential** for sub-10cm accuracy. A DHT22 or BME280 sensor measuring room temperature enables speed-of-sound correction, reducing systematic errors from 2-6cm per meter to \<1cm per meter. Implementation requires measuring temperature at multiple room heights to account for thermal stratification.

**Multi-sample averaging and outlier rejection**—collecting 5-10 readings per sensor, calculating median or filtered mean, and discarding outliers exceeding 1.5cm from median—improves precision from ±2cm single-shot to ±0.5-1cm averaged. This multiplies measurement time by the sample count.

**Triangulation algorithm sophistication** matters significantly. Simple geometric intersection produces 2-4cm errors. Least-squares optimization across multiple sensors reduces errors to 1-2cm. Kalman filtering incorporating motion models achieves 0.5-1cm tracking accuracy but requires tuning and computational resources.

### Minimum detectable object size

**Human bodies:** Reliably detectable at 0.5-3m when facing or backing toward sensors, but **clothing absorption reduces reliability to \<50-80% detection probability**. Heavy coats, loose fabrics, and soft materials can render humans partially or completely invisible to ultrasonic sensing.

**Small robots:** Objects \<1 foot × 1 foot struggle beyond 1m range. At 2m, the beam spot diameter spans ~70cm, and targets should occupy significant portions (20-40%) of the beam for reliable detection. Small robots (\<10cm cross-section) become undetectable beyond 50cm unless constructed of highly reflective materials and maintained perpendicular to sensor beams.

**Target material dominates size requirements.** A 20cm foam cube may not detect at 1m, while a 20cm aluminum plate reflects reliably to 3m. For positioning applications tracking humans or mobile robots, **hard mounting of reflector tags** (flat metal or plastic plates) dramatically improves detection reliability and range.

### Practical accuracy in cluttered indoor environments

**Cluttered environments degrade performance significantly.** Furniture, curtains, walls, and miscellaneous objects within the sensor's beam create multipath reflections, false echoes, and coverage gaps. The 15-20° beam width at typical room distances (2-4m) creates 50-140cm diameter detection zones—any object within this cone can produce confusing echoes.

Testing in furnished rooms shows **accuracy degradation of 2-5× compared to controlled open spaces.** Simple obstacle avoidance that works reliably at ±10cm in open areas may degrade to ±30-50cm in living rooms with multiple reflective surfaces and soft furnishings.

**Room geometry matters intensely.** Square rooms with parallel walls create strongest multipath interference and standing waves. Irregular room shapes, angled walls, and sound-absorbing furnishings paradoxically improve performance by reducing coherent reflections. High ceilings (\>3m) reduce ceiling reflection interference compared to low ceilings (\<2.5m).

The room mapping project failure illustrates this: attempting to measure room dimensions with rotating sensor arrays produced circular artifacts from nearest-wall bias rather than accurate room geometry, demonstrating that **conventional HC-SR04 arrays cannot reliably distinguish target echo from environmental echoes** without sophisticated signal processing.

## Integration difficulty with PIR sensors

**Integration is straightforward and highly recommended.** PIR sensors cost $2-5, require simple 3-pin connections (VCC, GND, signal), and complement HC-SR04 limitations perfectly. Standard implementations use PIR for motion detection triggering, followed by HC-SR04 distance confirmation.

**Code integration example:**
```
IF PIR_motion_detected:
    distance = HC_SR04_measure()
    IF distance < threshold (50-200cm):
        trigger_action()
```

PIR handles initial wide-angle motion detection (90-120° coverage vs. 15-20° for HC-SR04), working reliably through clothing that blocks ultrasonic detection. HC-SR04 provides distance confirmation and spatial localization that PIR cannot. Combined detection reduces false positives from either sensor alone—PIR won't trigger on HVAC temperature changes if no object is in ultrasonic range; HC-SR04 won't mistake distant wall echoes for presence when PIR shows no infrared motion.

**Power optimization:** PIR operates continuously at \<1mA, HC-SR04 draws 15-20mA when actively measuring. Triggering ultrasonics only on PIR events extends battery life 10-20× in presence detection applications.

**Limitations:** PIR sensors fail in extreme temperature environments where room temperature approaches body temperature, become less sensitive to slow motion (\<0.3 m/s), and provide no distance information. HC-SR04 sensors work across temperature extremes but fail with soft targets. The combination covers both sensors' weaknesses.

## Final verdict: Feasibility for indoor positioning

HC-SR04 sensors are **marginally feasible** for coarse indoor positioning with substantial qualifications:

**Appropriate applications:**
- Room-level zone detection (which quadrant/area contains target)
- Presence detection combined with PIR for human occupancy
- Simple obstacle avoidance for slow-moving robots (\<0.5 m/s)
- Educational demonstrations of ultrasonic triangulation principles
- Cost-constrained projects where ±50cm-1m accuracy suffices

**Inappropriate applications:**
- Precise indoor positioning requiring \<10cm accuracy
- Tracking humans wearing normal clothing (reliability \<80%)
- Fast-moving object tracking (\>1 m/s)
- Environments with extensive soft furnishings or curtains
- Safety-critical applications requiring reliable detection
- Mapping room geometry or creating floor plans

**Alternative technologies** should be considered for demanding applications. Time-of-Flight LIDAR (VL53L0X, TF-Luna) provides 1-5cm accuracy with immunity to soft materials at $5-15 per sensor. UWB (Ultra-Wideband) tags deliver centimeter-level indoor positioning with multipath resistance at $20-50 per tag. Vision-based systems using cameras and ArUco markers achieve millimeter precision with broader applicability despite higher computational requirements.

The HC-SR04's primary advantage—**$1.50 cost per sensor**—makes it attractive for hobbyist projects and educational applications where perfect performance isn't required and limitations can be accepted. For production systems or applications requiring reliable \<20cm positioning accuracy, the engineering effort required to work around HC-SR04 limitations quickly exceeds the cost savings from more capable sensor technologies.

**System design recommendation:** If constrained to HC-SR04 sensors, implement 4 wall/ceiling-mounted sensors with time-division multiplexing (50ms spacing), temperature compensation via DHT22, 5-10 sample averaging with median filtering, Kalman filter motion tracking, and hard reflector tags on tracked objects. Expect ±10-30cm realistic accuracy in furnished indoor environments with 5 Hz update rate, accept complete detection failures with soft targets, and plan manual calibration procedures for sensor positioning and timing. This configuration can achieve room-level positioning of ±1-2m for hard targets, meeting the stated goal but requiring substantial development effort relative to the sensor cost savings.