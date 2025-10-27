# Deep Research Prompts for Feasibility Analysis

## Category 1: Structural Health Monitoring (Accelerometer-Based)

### Prompt 1.1: Sensor Capability & Sensitivity
```
Research Query:
"What is the minimum detectable vibration amplitude and frequency range required to identify structural damage (cracks, settling, joint weakness) in residential and small commercial buildings? Compare this with the specifications of MEMS accelerometers like MPU-6050 (±2g to ±16g range, 0-256 Hz bandwidth) and ADXL345. Are these consumer-grade sensors sensitive enough to detect early-stage structural anomalies, or do we need industrial-grade accelerometers? Include case studies or research papers where low-cost MEMS sensors were successfully used for structural health monitoring."

Key things to verify:
- Typical building vibration frequencies (normal vs damaged)
- MPU-6050 noise floor vs signal strength from structural issues
- Success stories of DIY/research projects using similar sensors
- Failure modes: what CAN'T these sensors detect?
```

### Prompt 1.2: False Positive Discrimination
```
Research Query:
"How can we distinguish between structural damage vibrations and environmental noise (traffic, wind, HVAC systems, footsteps, door slams) using signal processing techniques? What are the characteristic frequency signatures of:
1. Structural damage (resonant frequencies, damping changes)
2. Normal building activity
3. External disturbances

Specifically investigate: Can Fast Fourier Transform (FFT), wavelet analysis, or machine learning algorithms running on a standard laptop effectively filter false positives from MPU-6050 accelerometer data? What sampling rates and processing windows are recommended?"

Key things to verify:
- Frequency separation between signal and noise
- Real-time processing feasibility on laptop CPU
- Published algorithms/code for vibration analysis
- Required training data for ML approaches
```

### Prompt 1.3: Sensor Placement Strategy
```
Research Query:
"In structural health monitoring systems, what are the optimal locations to place accelerometers on building structures (columns, beams, joints, walls)? How many sensors are needed for a small-scale model (1m x 1m, 2-floor structure) vs a real room (4m x 4m)? Research wireless sensor network (WSN) architectures for structural monitoring - what are best practices for:
1. Sensor density (how many per square meter)
2. Mounting methods (adhesive, magnetic, bolted)
3. Spatial distribution patterns
4. Redundancy requirements

Focus on low-cost implementations and academic research projects, not commercial systems."

Key things to verify:
- Minimum viable sensor count for proof-of-concept
- Critical structural points to monitor
- Impact of poor mounting on data quality
- Sensor synchronization requirements
```

### Prompt 1.4: Data Transmission & Latency
```
Research Query:
"For real-time structural health monitoring, what are the acceptable data transmission latencies and sampling rates? Specifically:
1. Can ESP32 microcontrollers reliably stream accelerometer data over WiFi (MQTT protocol) at 100-200 Hz sampling rate to a laptop without significant packet loss?
2. What is the maximum number of ESP32 nodes that can simultaneously transmit to one MQTT broker on a standard home WiFi router?
3. What happens during WiFi congestion or connection drops?
4. Are there published ESP32 + MPU-6050 projects demonstrating multi-node vibration monitoring networks?

Include power consumption analysis: can battery-powered ESP32 nodes last 24+ hours while continuously sampling at 100 Hz?"

Key things to verify:
- WiFi bandwidth limitations
- ESP32 processing overhead for MQTT
- Battery life calculations
- Network reliability statistics
```

---

## Category 2: Occupancy Detection & Localization

### Prompt 2.1: PIR Sensor Limitations
```
Research Query:
"What are the fundamental limitations of Passive Infrared (PIR) motion sensors (HC-SR501) for indoor occupancy tracking? Specifically investigate:
1. Detection range and field of view (typically 7m range, 120° cone)
2. Blind spots and false triggers (pets, sunlight, heating vents)
3. Response time (typical 2-3 second reset time)
4. Can PIR sensors differentiate between one person vs multiple people in the same zone?
5. How does room temperature affect IR detection accuracy?

Find research papers or projects that used PIR sensor grids for room-level occupancy tracking. What grid spacing (distance between sensors) is recommended for a 1m x 1m model vs 4m x 4m room?"

Key things to verify:
- Whether 4 PIR sensors are sufficient for zone tracking
- Expected false positive/negative rates
- Multi-person detection capability
- Temperature compensation needs
```

### Prompt 2.2: Ultrasonic Distance Measurement
```
Research Query:
"Evaluate the feasibility of using HC-SR04 ultrasonic sensors for coarse indoor positioning. Research:
1. Accuracy and precision (typically ±3mm to ±5mm, but what about moving targets?)
2. Maximum reliable range in indoor environments (claimed 2-400cm)
3. Interference between multiple ultrasonic sensors (can 4 sensors operate simultaneously without cross-talk?)
4. Failure modes: absorption by clothing, reflections from walls, angular limitations
5. Update rate when used with ESP32

Find examples of ultrasonic sensor arrays used for presence detection or simple localization. Can triangulation from 3-4 HC-SR04 sensors provide room-level (±1-2 meter) position accuracy?"

Key things to verify:
- Sensor interference mitigation strategies
- Minimum detectable object size (human vs small robot)
- Practical accuracy in cluttered environments
- Integration difficulty with PIR sensors
```

### Prompt 2.3: Sensor Fusion for Zone Localization
```
Research Query:
"How can PIR motion sensors and ultrasonic distance sensors be fused to achieve zone-level occupancy tracking (e.g., 'person is in Room A' or 'person is in northwest quadrant')? Research sensor fusion algorithms suitable for low-cost embedded systems:
1. Bayesian filtering approaches
2. Simple rule-based logic (if PIR triggers AND ultrasonic detects object at distance X...)
3. Occupancy grid mapping techniques
4. Kalman filtering for tracking moving targets

What is the state-of-the-art accuracy for such systems? Find academic papers or GitHub projects implementing similar sensor combinations. Can this run on a laptop in Python with <100ms latency?"

Key things to verify:
- Computational complexity of fusion algorithms
- Expected localization accuracy (±0.5m, ±1m, ±2m?)
- Handling of occlusions and multiple occupants
- Open-source code availability
```

### Prompt 2.4: Privacy & Ethical Comparison
```
Research Query:
"Compare the privacy implications and ethical considerations of different occupancy tracking technologies:
1. Camera-based (computer vision, face detection)
2. WiFi/Bluetooth tracking (MAC address sniffing)
3. UWB radar
4. PIR + Ultrasonic (our approach)

Find guidelines or regulations (IEEE, ISO, GDPR) regarding indoor monitoring systems. Specifically: Does our sensor-only approach (no cameras, no personal identifiers) qualify as 'privacy-preserving'? Are there legal or ethical concerns we should address in our project documentation?"

Key things to verify:
- Whether our approach is truly privacy-friendly
- Consent requirements for building monitoring
- Data retention best practices
- Ethical review board considerations
```

---

## Category 3: Mobile Robot Navigation & Mapping

### Prompt 3.1: Low-Cost SLAM Feasibility
```
Research Query:
"Is it feasible to implement SLAM (Simultaneous Localization and Mapping) on a budget robot using only:
- 2WD chassis with wheel encoders
- HC-SR04 ultrasonic sensors (or array of 4-6 sensors)
- MPU-6050 IMU for orientation
- ESP32 microcontroller

Research lightweight SLAM algorithms like:
1. EKF-SLAM (Extended Kalman Filter)
2. Particle Filter-based SLAM
3. Occupancy grid mapping with ultrasonic sensors

Find Arduino/ESP32 projects that achieved basic SLAM without expensive lidar. What map accuracy (±5cm, ±10cm?) can we expect in a 1m x 1m environment? Can the ESP32 handle SLAM computations, or must we offload to laptop?"

Key things to verify:
- Whether ultrasonic-only SLAM is viable (or do we need cheap lidar?)
- ESP32 memory/processing constraints
- Open-source libraries (ROSSerial, Arduino SLAM implementations)
- Minimum sensor suite for 2D mapping
```

### Prompt 3.2: Autonomous Navigation in Small Spaces
```
Research Query:
"What are the challenges of autonomous robot navigation in confined spaces (1m x 1m model building with obstacles)? Research:
1. Minimum turning radius requirements for 2WD robots
2. Obstacle detection reliability with ultrasonic sensors (can robot avoid 5cm x 5cm obstacles?)
3. Path planning algorithms suitable for resource-constrained systems (A*, DWA, potential fields)
4. Collision recovery strategies

Find examples of small robots (10cm x 10cm footprint) navigating complex indoor environments. What percentage success rate can we expect for tasks like 'drive from Point A to Point B without collision'?"

Key things to verify:
- Robot size vs environment scale feasibility
- Sensor coverage blind spots
- Mechanical limitations (wheels slipping on cardboard?)
- Failsafe mechanisms (manual override)
```

### Prompt 3.3: ESP32-CAM for Visual Inspection
```
Research Query:
"Evaluate the ESP32-CAM module for real-time visual inspection tasks:
1. Image quality and resolution (typically 1600x1200 max)
2. Frame rate when streaming over WiFi (5-15 fps typical)
3. Low-light performance (does robot need LED illumination?)
4. Can basic computer vision run on ESP32 (edge detection, anomaly highlighting) or must images be sent to laptop?
5. Power consumption impact on battery life

Find projects where ESP32-CAM was used for inspection or surveillance. Can we capture clear enough images to visually verify 'damage' markers (red tape on model) from 20-30cm distance?"

Key things to verify:
- Image quality sufficient for proof-of-concept
- WiFi bandwidth for video streaming
- Whether we need on-board image processing
- Alternative: store images on SD card vs real-time streaming
```

### Prompt 3.4: Robot-Sensor Network Coordination
```
Research Query:
"How can a mobile robot coordinate with a fixed wireless sensor network in real-time? Research architectures where:
1. Fixed sensors (accelerometers, PIR) detect an event
2. Central system (laptop) sends waypoint commands to robot
3. Robot navigates to event location autonomously
4. Robot reports back status (arrived, image captured)

Investigate communication protocols:
- MQTT for sensor network + robot control (single protocol)
- ROS (Robot Operating System) for laptop-robot communication
- Direct ESP32-to-ESP32 messaging

What is the typical end-to-end latency from 'event detected' to 'robot at location' in research systems? Find similar projects integrating stationary sensors with mobile agents."

Key things to verify:
- Protocol compatibility (can everything use MQTT?)
- Command latency (<5 seconds feasible?)
- Handling of robot task failures
- Multi-device coordination complexity
```

---

## Category 4: Digital Twin & Simulation

### Prompt 4.1: Real-Time Data Visualization
```
Research Query:
"What are the best tools/frameworks for creating real-time 3D visualizations of sensor data on a laptop? Compare:
1. Python-based: Matplotlib (3D plots), Pygame, PyQt + OpenGL, VPython
2. Game engines: Unity3D, Unreal Engine (can they import live sensor data?)
3. Web-based: Three.js, Babylon.js (browser-based 3D)

Requirements:
- Update visualization at 10-20 Hz based on incoming sensor data
- Color-code building elements (green/yellow/red for health status)
- Overlay live graphs (accelerometer waveforms)
- Show robot position in real-time

Find tutorials or GitHub projects showing sensor data visualization in 3D. What's easier for students: Python GUI or Unity3D scripting?"

Key things to verify:
- Learning curve for each tool
- Performance on mid-range laptop (integrated graphics)
- Ease of importing 3D models
- Data streaming integration
```

### Prompt 4.2: Simplified Structural Simulation
```
Research Query:
"For a student-level project, what is the simplest physics simulation approach to demonstrate 'structural failure analysis'? Research:
1. Finite Element Analysis (FEA) - is it too complex for real-time simulation?
2. Simplified beam theory (Euler-Bernoulli equations)
3. Lookup table approach (pre-compute failure scenarios)
4. Probabilistic risk assessment (simple scoring: if vibration > threshold, risk = high)

Find Python libraries or tools:
- PyFEM, FEniCS (open-source FEA)
- Simple physics engines (Pymunk, PyBullet)
- Rule-based systems

Goal: Given sensor input 'Column A vibrating at 15 Hz with 0.5g amplitude', system should classify risk level (low/medium/high) within 1 second. Does this require complex simulation or can we use simplified heuristics?"

Key things to verify:
- Whether FEA is overkill for proof-of-concept
- Computational cost of real-time physics
- Accuracy needed for demonstration purposes
- Existing code examples we can adapt
```

### Prompt 4.3: Evacuation Path Planning
```
Research Query:
"Research path planning algorithms for emergency evacuation routing:
1. A* (A-star) algorithm for grid-based maps
2. Dijkstra's algorithm with dynamic weights (avoid damaged zones)
3. Potential field methods
4. Multi-agent evacuation simulation (if tracking multiple occupants)

Requirements:
- Compute path from 'Person at Location X' to 'Exit' avoiding zones marked as 'damaged'
- Update path in real-time if new damage detected
- Visualize path on 3D Digital Twin

Find Python implementations (NetworkX library?) and examples of dynamic path replanning. What is typical computation time for a 10x10 grid map on a standard laptop (<1 second feasible?)?"

Key things to verify:
- Algorithm complexity vs map size
- Dynamic obstacle handling
- Python library availability
- Integration with visualization
```

### Prompt 4.4: Data Logging & Post-Event Analysis
```
Research Query:
"What are best practices for logging and analyzing time-series sensor data in structural monitoring systems? Research:
1. Database options: SQLite (lightweight), InfluxDB (time-series optimized), CSV files
2. Data retention strategies (how long to keep historical data?)
3. Post-event analysis tools (Jupyter notebooks, Grafana dashboards)
4. Anomaly detection techniques (statistical thresholds, machine learning)

Specifically: If our system runs for 24 hours collecting data from 4 accelerometers + 4 PIR + 4 ultrasonic sensors at ~10 Hz, how much storage space is needed? Can a laptop handle this data volume? Find examples of academic projects with sensor data analysis pipelines."

Key things to verify:
- Data storage requirements (MB per hour)
- Query performance for historical playback
- Tools for generating reports/graphs
- Data export formats for documentation
```

---

## Category 5: System Integration & Reliability

### Prompt 5.1: ESP32 Network Scalability
```
Research Query:
"What is the maximum number of ESP32 devices that can reliably connect to a single WiFi access point and communicate with one MQTT broker? Research:
1. WiFi AP client limits (typically 10-20 for home routers)
2. MQTT broker throughput (Mosquitto on laptop)
3. ESP32 WiFi stability under congestion
4. Packet loss rates in dense ESP32 networks

Our system needs: 3 ESP32 sensor nodes + 1 ESP32 robot = 4 total. Is this well within safe limits? Find stress tests or benchmarks of ESP32 WSNs. What happens if we scale to 10 or 20 nodes in a real building?"

Key things to verify:
- 4-device network is definitely feasible
- Scalability bottlenecks for future expansion
- WiFi channel selection importance
- Alternative protocols (ESP-NOW mesh) as backup
```

### Prompt 5.2: Power Budget Analysis
```
Research Query:
"Calculate detailed power consumption for battery-operated ESP32 sensor nodes:
1. ESP32 in WiFi active mode: ~160-260 mA
2. MPU-6050 accelerometer: ~3.5 mA
3. Transmission duty cycle (1 packet per second)
4. Deep sleep mode power savings (~10 µA)

Given 2x 18650 batteries (3.7V, 2500 mAh each = 5000 mAh total):
- How long can a sensor node run continuously at 100 Hz sampling?
- Can we extend battery life to 24+ hours using sleep modes?

Research ESP32 power optimization techniques and low-power sensor network architectures. Find projects with actual measured battery life data."

Key things to verify:
- 24-hour operation feasibility
- Whether deep sleep is compatible with real-time monitoring
- Battery recharging strategy
- Power failure handling
```

### Prompt 5.3: Environmental Testing Requirements
```
Research Query:
"What environmental factors affect sensor accuracy and system reliability in our proposed setup?
1. Temperature: Do MPU-6050/PIR/Ultrasonic sensors need calibration across 10-40°C range?
2. Humidity: Can electronics handle 60-80% relative humidity (monsoon conditions)?
3. Vibration isolation: Does mounting method affect accelerometer readings?
4. Electromagnetic interference: Can WiFi/Bluetooth from other devices cause problems?

Find environmental testing standards (IP ratings, operating temperature ranges) for the components we're using. Do we need any protective enclosures or climate compensation in our code?"

Key things to verify:
- Indoor lab environment is sufficient for prototype
- Whether outdoor/harsh conditions need special consideration
- Calibration procedures required
- Failure modes under stress
```

### Prompt 5.4: Fail-Safe Mechanisms
```
Research Query:
"Design fault-tolerance and fail-safe strategies for a safety-critical monitoring system:
1. What happens if one sensor fails? (Single point of failure analysis)
2. How to detect sensor malfunction vs genuine anomaly?
3. Watchdog timers for ESP32 (auto-restart on hang)
4. Data validation (sanity checks for impossible readings)
5. Backup communication paths if WiFi fails
6. Manual override mechanisms

Research safety-critical embedded systems best practices. Find examples of redundancy strategies in academic/hobbyist projects. Since this is a student prototype, what minimum fail-safes are acceptable vs full commercial system?"

Key things to verify:
- Critical vs non-critical failure modes
- Error detection mechanisms
- System degradation gracefully vs catastrophic failure
- Ethical considerations for safety applications
```

---

## Category 6: Validation & Testing Methodologies

### Prompt 6.1: Ground Truth Establishment
```
Research Query:
"How can we create 'ground truth' data to validate our system's accuracy?
1. For structural monitoring: How to simulate realistic vibration patterns? (shaker table, controlled impacts, speaker-based excitation)
2. For occupancy tracking: How to measure true position vs detected position? (marked floor grid, video recording)
3. For robot localization: How to measure actual vs estimated position? (external tracking system, manual measurement)

Find low-cost validation methods used in academic research. Can we use smartphone sensors as reference? What statistical metrics should we report (precision, recall, F1-score for detection; RMSE for position)?"

Key things to verify:
- DIY validation setup within budget
- Acceptable accuracy thresholds for student project
- Comparison baselines (random guessing, simple heuristics)
- Documentation requirements for academic rigor
```

### Prompt 6.2: Benchmark Scenarios
```
Research Query:
"Design test scenarios to systematically evaluate our system:
1. Structural monitoring tests:
   - Controlled vibration at known frequencies (1 Hz, 5 Hz, 10 Hz)
   - Impact tests (dropping weight from fixed height)
   - Baseline noise measurement (no activity)
2. Occupancy tracking tests:
   - Single person walking predefined path
   - Person stationary in each zone
   - Empty room (false positive test)
3. Robot navigation tests:
   - Point-to-point navigation (5 trials, measure success rate)
   - Obstacle avoidance (introduce random obstacles)
   - Emergency response time (alert to inspection complete)

Find academic papers describing similar test protocols. What sample sizes (10 trials? 100 trials?) are needed for statistical significance?"

Key things to verify:
- Reproducible test procedures
- Pass/fail criteria for each test
- Expected vs actual performance gaps
- Comprehensive coverage of failure modes
```

### Prompt 6.3: User Acceptance Criteria
```
Research Query:
"What constitutes a 'successful' demonstration for academic reviewers and potential end-users? Research:
1. Industry standards for structural health monitoring (what accuracy do commercial systems achieve?)
2. User experience expectations (how fast should alerts appear? How intuitive should UI be?)
3. Comparison with existing solutions (our ₹5,000 system vs ₹5,00,000 commercial system - what compromises are acceptable?)

Survey similar student projects or startup prototypes. What impressed reviewers? What were common criticisms? Define SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound) for our Review-II and Review-V milestones."

Key things to verify:
- Realistic vs over-ambitious goals
- What reviewers prioritize (novelty, execution, presentation?)
- Documentation quality expectations
- Comparison fairness (prototype vs mature product)
```

---

## Category 7: Future Scalability & Real-World Deployment

### Prompt 7.1: Scaling to Real Building
```
Research Query:
"What are the technical and economic challenges of scaling our ₹5,000 prototype to a real building (e.g., 4-room apartment, 150 sq meters)? Research:
1. Sensor density requirements (how many nodes per 10 sq meters?)
2. Communication infrastructure (can home WiFi support 20+ ESP32s? Need for mesh network?)
3. Installation complexity (mounting sensors, cable management, power supply)
4. Maintenance requirements (battery replacement, sensor calibration)
5. Cost projection (per-room vs whole-building)

Find case studies of deployed sensor networks in buildings. What were unexpected challenges? How much does installation labor cost compared to hardware?"

Key things to verify:
- Feasibility of DIY installation vs professional
- Total cost for real-world deployment
- Regulatory approvals needed (electrical safety, building codes)
- Long-term operational costs
```

### Prompt 7.2: Commercial Viability Assessment
```
Research Query:
"Could this project be commercialized as a product or service? Research:
1. Market analysis: Who would buy this? (homeowners, schools, small businesses?)
2. Competitive landscape: Existing products in ₹10,000-₹50,000 range?
3. Value proposition: What problems does this solve better/cheaper than alternatives?
4. Business models: One-time purchase vs subscription (cloud storage, alerts)?
5. Certifications needed: CE/FCC (electronics), building safety standards

Find examples of academic projects that became startups. What were key success factors? Are there incubators or competitions (TiE, IIT startup programs) we could target?"

Key things to verify:
- Whether there's genuine market need
- Competitive advantages we can claim
- Intellectual property considerations (patents, open-source)
- Realistic path from prototype to product
```

### Prompt 7.3: Integration with Existing Systems
```
Research Query:
"How could our system integrate with existing building management infrastructure?
1. Fire alarm systems (can our sensor data trigger fire panel alerts?)
2. Security systems (occupancy data to security company monitoring)
3. Smart home platforms (Google Home, Alexa integration for alerts)
4. Building management software (API for facility managers)
5. Emergency services (automatic 911 call with occupant location data)

Research interoperability standards (BACnet, MQTT, RESTful APIs). Are there legal/liability issues with automated emergency calls? Find examples of IoT safety devices and their integration approaches."

Key things to verify:
- Technical interfaces available
- Liability concerns for safety-critical integration
- User consent and privacy policies needed
- Industry standards we should follow
```

---

## How to Use These Prompts

### **Step 1: Prioritize by Project Phase**
For **TRL-3 (Review-II)**, focus on:
- Prompts 1.1, 1.2 (accelerometer feasibility)
- Prompts 2.1, 2.2 (PIR/ultrasonic basics)
- Prompt 3.1 (robot navigation basics)

For **TRL 6-7 (Review-V)**, additionally cover:
- All Category 4 (Digital Twin)
- All Category 5 (System Integration)
- Category 6 (Validation)

### **Step 2: Research Methodology**
For each prompt:
1. **Google Scholar search** (academic papers with citations)
2. **GitHub search** (real code examples: "ESP32 MPU-6050 vibration")
3. **YouTube tutorials** (practical implementation videos)
4. **Reddit/Forum discussions** (r/arduino, r/esp32 - real user experiences)
5. **Technical datasheets** (manufacturer specs for sensors)

### **Step 3: Document Findings**
Create a research log:
```
Prompt 1.1 - Sensor Capability
✅ Found paper: "Low-cost MEMS for bridge monitoring" (2019)
✅ Key finding: MPU-6050 detected 0.1 Hz vibrations successfully
❌ Limitation: Noise floor of 0.02g might mask small cracks
📊 Data: 95% accuracy for magnitude > 0.1g
🔗 Link: [URL]
💡 Action: Need to test in lab environment first
```

### **Step 4: Create Feasibility Matrix**
After researching all prompts, summarize:

| Component/Feature | Feasibility (1-5) | Confidence | Risk Mitigation |
|-------------------|-------------------|------------|-----------------|
| MPU-6050 for vibration | 4/5 | High | Proven in literature |
| PIR zone tracking | 3/5 | Medium | May need more sensors |
| Ultrasonic SLAM | 2/5 | Low | Consider cheap lidar alternative |
| ... | ... | ... | ... |

---
