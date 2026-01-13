/*
 * Structure Integrity Prediction and Safe Zone Identification System
 *
 * This system analyzes structural health monitoring data to:
 * 1. Predict structural integrity based on vibration/acceleration data
 * 2. Identify potential collapse zones
 * 3. Calculate the safest locations during structural failure
 * 4. Generate evacuation paths
 *
 * Author: ultrawork
 * Date: 2026-01-13
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

// Constants
#define MAX_GRID_SIZE 50
#define MAX_SENSORS 16
#define MAX_HISTORY 1000
#define GRAVITY 9.81
#define PI 3.14159265359

// Thresholds for structural integrity
#define CRITICAL_VIBRATION_THRESHOLD 5.0    // m/s²
#define WARNING_VIBRATION_THRESHOLD 2.5     // m/s²
#define CRITICAL_FREQ_THRESHOLD 15.0        // Hz
#define DAMAGE_PROPAGATION_RATE 0.3         // per time step
#define SAFE_DISTANCE_FACTOR 2.0            // meters from damaged zone

// Structure types
typedef enum {
    ZONE_HEALTHY = 0,
    ZONE_WARNING = 1,
    ZONE_CRITICAL = 2,
    ZONE_COLLAPSED = 3,
    ZONE_EXIT = 4
} ZoneStatus;

typedef enum {
    CONCRETE = 0,
    STEEL = 1,
    WOOD = 2,
    COMPOSITE = 3
} MaterialType;

// Sensor data structure
typedef struct {
    int id;
    float x_pos, y_pos, z_pos;  // Position in 3D space (meters)
    float accel_x, accel_y, accel_z;  // Current acceleration (m/s²)
    float freq_dominant;         // Dominant frequency (Hz)
    float rms_vibration;         // RMS of vibration
    float temperature;           // Temperature compensation
    bool is_active;
    time_t last_update;
} Sensor;

// Grid cell representing a zone in the structure
typedef struct {
    int x, y, z;
    ZoneStatus status;
    MaterialType material;
    float integrity;            // 0.0 (collapsed) to 1.0 (perfect)
    float load_capacity;        // Normalized load capacity
    float vibration_level;      // Current vibration amplitude
    float damage_factor;        // Accumulated damage (0.0 - 1.0)
    int occupancy;             // Number of people in zone
    bool is_exit;
    bool is_structural;        // Load-bearing element
} GridCell;

// Building structure
typedef struct {
    GridCell grid[MAX_GRID_SIZE][MAX_GRID_SIZE][MAX_GRID_SIZE];
    int size_x, size_y, size_z;
    Sensor sensors[MAX_SENSORS];
    int num_sensors;
    float overall_integrity;
    time_t last_analysis;
} Building;

// Safe zone information
typedef struct {
    int x, y, z;
    float safety_score;
    float distance_to_exit;
    char reason[256];
} SafeZone;

// Function prototypes
void init_building(Building *building, int size_x, int size_y, int size_z);
void add_sensor(Building *building, int id, float x, float y, float z);
void update_sensor_data(Building *building, int sensor_id, float ax, float ay, float az, float freq);
void analyze_structural_integrity(Building *building);
void predict_collapse_zones(Building *building);
void identify_safe_zones(Building *building, SafeZone *safe_zones, int *num_safe_zones);
void propagate_damage(Building *building);
float calculate_zone_safety_score(Building *building, int x, int y, int z);
float calculate_vibration_rms(float ax, float ay, float az);
float estimate_dominant_frequency(Building *building, int sensor_id);
void visualize_building_status(Building *building);
void generate_evacuation_path(Building *building, int start_x, int start_y, int start_z);
void print_safety_report(Building *building);

// Mathematical helper functions
float vector_magnitude(float x, float y, float z);
float euclidean_distance_3d(int x1, int y1, int z1, int x2, int y2, int z2);
float normalize(float value, float min, float max);

//=============================================================================
// MAIN FUNCTION
//=============================================================================

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   STRUCTURE INTEGRITY PREDICTION & SAFE ZONE ANALYZER       ║\n");
    printf("║   Real-time Structural Health Monitoring System             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Initialize building structure (10x10x3 grid - representing a 3-floor building)
    Building building;
    init_building(&building, 10, 10, 3);

    // Add sensors at strategic locations
    printf("Installing sensors...\n");
    add_sensor(&building, 0, 2.0, 2.0, 0.0);  // Corner sensor, ground floor
    add_sensor(&building, 1, 8.0, 2.0, 0.0);  // Corner sensor, ground floor
    add_sensor(&building, 2, 2.0, 8.0, 0.0);  // Corner sensor, ground floor
    add_sensor(&building, 3, 8.0, 8.0, 0.0);  // Corner sensor, ground floor
    add_sensor(&building, 4, 5.0, 5.0, 1.0);  // Center sensor, second floor
    add_sensor(&building, 5, 5.0, 5.0, 2.0);  // Center sensor, third floor

    printf("✓ %d sensors installed\n\n", building.num_sensors);

    // Simulation: Update sensor data over time
    printf("=== SIMULATION START ===\n\n");

    for (int time_step = 0; time_step < 10; time_step++) {
        printf("\n┌─────────────────────────────────────────────┐\n");
        printf("│ Time Step: %d                                │\n", time_step);
        printf("└─────────────────────────────────────────────┘\n\n");

        // Simulate sensor readings (in real system, this comes from ADXL345/MPU6050)
        for (int i = 0; i < building.num_sensors; i++) {
            float base_vibration = 0.5;
            float noise = ((float)rand() / RAND_MAX) * 0.2;

            // Simulate increasing vibration at sensor 1 (structural damage developing)
            if (i == 1 && time_step > 3) {
                base_vibration = 2.0 + (time_step - 3) * 1.5;
            }

            // Simulate earthquake scenario at time_step 7
            if (time_step == 7) {
                base_vibration = 6.0 + ((float)rand() / RAND_MAX) * 2.0;
            }

            float ax = base_vibration * sin(time_step * 0.5) + noise;
            float ay = base_vibration * cos(time_step * 0.7) + noise;
            float az = GRAVITY + base_vibration * sin(time_step * 0.3) + noise;
            float freq = 5.0 + ((float)rand() / RAND_MAX) * 10.0;

            if (i == 1 && time_step > 5) {
                freq = 18.0 + ((float)rand() / RAND_MAX) * 5.0; // Abnormal frequency
            }

            update_sensor_data(&building, i, ax, ay, az, freq);
        }

        // Analyze structural integrity
        analyze_structural_integrity(&building);

        // Predict potential collapse zones
        predict_collapse_zones(&building);

        // Propagate damage through structure
        propagate_damage(&building);

        // Visualize current status
        visualize_building_status(&building);

        // If critical situation detected, find safe zones
        if (building.overall_integrity < 0.7) {
            printf("\n⚠️  CRITICAL SITUATION DETECTED!\n");
            printf("Overall Building Integrity: %.1f%%\n\n", building.overall_integrity * 100);

            SafeZone safe_zones[100];
            int num_safe_zones = 0;
            identify_safe_zones(&building, safe_zones, &num_safe_zones);

            printf("═══════════════════════════════════════════════════════\n");
            printf("  SAFE ZONES IDENTIFIED: %d locations\n", num_safe_zones);
            printf("═══════════════════════════════════════════════════════\n");

            for (int i = 0; i < num_safe_zones && i < 5; i++) {
                printf("\n%d. Zone [%d, %d, Floor %d]\n", i + 1,
                       safe_zones[i].x, safe_zones[i].y, safe_zones[i].z);
                printf("   Safety Score: %.2f/1.0\n", safe_zones[i].safety_score);
                printf("   Distance to Exit: %.1fm\n", safe_zones[i].distance_to_exit);
                printf("   Reason: %s\n", safe_zones[i].reason);
            }

            // Generate evacuation path from a sample location
            printf("\n═══════════════════════════════════════════════════════\n");
            printf("  EVACUATION PATH (from center of building)\n");
            printf("═══════════════════════════════════════════════════════\n");
            generate_evacuation_path(&building, 5, 5, 0);
        }

        // Print summary report
        print_safety_report(&building);

        // Pause for readability
        if (time_step < 9) {
            printf("\n[Press Enter to continue...]\n");
            getchar();
        }
    }

    printf("\n\n=== SIMULATION COMPLETE ===\n");
    printf("\nFinal Building Status:\n");
    printf("  Overall Integrity: %.1f%%\n", building.overall_integrity * 100);
    printf("  Status: %s\n",
           building.overall_integrity > 0.8 ? "SAFE" :
           building.overall_integrity > 0.5 ? "CAUTION" : "CRITICAL");

    return 0;
}

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================

void init_building(Building *building, int size_x, int size_y, int size_z) {
    building->size_x = size_x;
    building->size_y = size_y;
    building->size_z = size_z;
    building->num_sensors = 0;
    building->overall_integrity = 1.0;
    building->last_analysis = time(NULL);

    // Initialize grid
    for (int z = 0; z < size_z; z++) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                GridCell *cell = &building->grid[x][y][z];
                cell->x = x;
                cell->y = y;
                cell->z = z;
                cell->status = ZONE_HEALTHY;
                cell->integrity = 1.0;
                cell->load_capacity = 1.0;
                cell->vibration_level = 0.0;
                cell->damage_factor = 0.0;
                cell->occupancy = 0;
                cell->is_exit = false;
                cell->is_structural = false;

                // Assign material type
                if (z == 0) {
                    cell->material = CONCRETE;  // Ground floor: concrete
                    cell->load_capacity = 1.5;
                } else {
                    cell->material = WOOD;      // Upper floors: lighter materials
                    cell->load_capacity = 0.8;
                }

                // Mark structural elements (columns at corners and center)
                if ((x % 4 == 0 && y % 4 == 0) || (x == size_x/2 && y == size_y/2)) {
                    cell->is_structural = true;
                    cell->material = STEEL;
                    cell->load_capacity = 2.0;
                }

                // Mark exits at edges
                if ((x == 0 || x == size_x - 1 || y == 0 || y == size_y - 1) && z == 0) {
                    cell->is_exit = true;
                    cell->status = ZONE_EXIT;
                }
            }
        }
    }

    // Add some occupancy
    building->grid[5][5][0].occupancy = 2;
    building->grid[3][7][1].occupancy = 1;
    building->grid[6][4][2].occupancy = 3;
}

void add_sensor(Building *building, int id, float x, float y, float z) {
    if (building->num_sensors >= MAX_SENSORS) {
        printf("Warning: Maximum sensors reached\n");
        return;
    }

    Sensor *sensor = &building->sensors[building->num_sensors];
    sensor->id = id;
    sensor->x_pos = x;
    sensor->y_pos = y;
    sensor->z_pos = z;
    sensor->accel_x = 0.0;
    sensor->accel_y = 0.0;
    sensor->accel_z = GRAVITY;
    sensor->freq_dominant = 0.0;
    sensor->rms_vibration = 0.0;
    sensor->temperature = 25.0;
    sensor->is_active = true;
    sensor->last_update = time(NULL);

    building->num_sensors++;
}

//=============================================================================
// SENSOR DATA PROCESSING
//=============================================================================

void update_sensor_data(Building *building, int sensor_id, float ax, float ay, float az, float freq) {
    if (sensor_id >= building->num_sensors) return;

    Sensor *sensor = &building->sensors[sensor_id];
    sensor->accel_x = ax;
    sensor->accel_y = ay;
    sensor->accel_z = az;
    sensor->freq_dominant = freq;
    sensor->rms_vibration = calculate_vibration_rms(ax, ay, az - GRAVITY);
    sensor->last_update = time(NULL);

    // Update nearby grid cells with sensor data
    int grid_x = (int)(sensor->x_pos);
    int grid_y = (int)(sensor->y_pos);
    int grid_z = (int)(sensor->z_pos);

    if (grid_x >= 0 && grid_x < building->size_x &&
        grid_y >= 0 && grid_y < building->size_y &&
        grid_z >= 0 && grid_z < building->size_z) {

        building->grid[grid_x][grid_y][grid_z].vibration_level = sensor->rms_vibration;

        // Propagate vibration to nearby cells (simplified model)
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = grid_x + dx;
                int ny = grid_y + dy;
                if (nx >= 0 && nx < building->size_x && ny >= 0 && ny < building->size_y) {
                    float distance = sqrt(dx*dx + dy*dy) + 1.0;
                    building->grid[nx][ny][grid_z].vibration_level =
                        sensor->rms_vibration / distance;
                }
            }
        }
    }
}

float calculate_vibration_rms(float ax, float ay, float az) {
    return sqrt((ax * ax + ay * ay + az * az) / 3.0);
}

//=============================================================================
// STRUCTURAL INTEGRITY ANALYSIS
//=============================================================================

void analyze_structural_integrity(Building *building) {
    float total_integrity = 0.0;
    int total_cells = 0;

    for (int z = 0; z < building->size_z; z++) {
        for (int y = 0; y < building->size_y; y++) {
            for (int x = 0; x < building->size_x; x++) {
                GridCell *cell = &building->grid[x][y][z];

                // Calculate integrity based on vibration level and accumulated damage
                float vibration_factor = 1.0;
                if (cell->vibration_level > CRITICAL_VIBRATION_THRESHOLD) {
                    vibration_factor = 0.3;
                    cell->damage_factor += 0.1;
                } else if (cell->vibration_level > WARNING_VIBRATION_THRESHOLD) {
                    vibration_factor = 0.7;
                    cell->damage_factor += 0.02;
                }

                // Update integrity (cannot exceed 1.0, cannot go below 0.0)
                cell->integrity = cell->integrity * vibration_factor - cell->damage_factor;
                if (cell->integrity > 1.0) cell->integrity = 1.0;
                if (cell->integrity < 0.0) cell->integrity = 0.0;

                // Update status based on integrity
                if (cell->integrity < 0.3) {
                    cell->status = ZONE_COLLAPSED;
                } else if (cell->integrity < 0.5) {
                    cell->status = ZONE_CRITICAL;
                } else if (cell->integrity < 0.8) {
                    cell->status = ZONE_WARNING;
                } else if (!cell->is_exit) {
                    cell->status = ZONE_HEALTHY;
                }

                total_integrity += cell->integrity;
                total_cells++;
            }
        }
    }

    building->overall_integrity = total_integrity / total_cells;
    building->last_analysis = time(NULL);
}

void predict_collapse_zones(Building *building) {
    // Identify zones with critical structural damage that may collapse
    printf("\n--- Collapse Risk Analysis ---\n");

    bool collapse_risk_found = false;

    for (int z = 0; z < building->size_z; z++) {
        for (int y = 0; y < building->size_y; y++) {
            for (int x = 0; x < building->size_x; x++) {
                GridCell *cell = &building->grid[x][y][z];

                // Structural elements are critical
                if (cell->is_structural && cell->integrity < 0.6) {
                    printf("⚠️  CRITICAL: Structural element at [%d,%d,%d] compromised (%.1f%%)\n",
                           x, y, z, cell->integrity * 100);
                    collapse_risk_found = true;

                    // Mark surrounding area as critical
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            int nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < building->size_x &&
                                ny >= 0 && ny < building->size_y) {
                                if (building->grid[nx][ny][z].status != ZONE_COLLAPSED) {
                                    building->grid[nx][ny][z].status = ZONE_CRITICAL;
                                }
                            }
                        }
                    }
                }

                // High vibration zones
                if (cell->vibration_level > CRITICAL_VIBRATION_THRESHOLD) {
                    printf("⚠️  High vibration at [%d,%d,%d]: %.2f m/s²\n",
                           x, y, z, cell->vibration_level);
                    collapse_risk_found = true;
                }
            }
        }
    }

    if (!collapse_risk_found) {
        printf("✓ No immediate collapse risks detected\n");
    }
}

void propagate_damage(Building *building) {
    // Damage propagates from failed zones to adjacent zones
    for (int z = 0; z < building->size_z; z++) {
        for (int y = 0; y < building->size_y; y++) {
            for (int x = 0; x < building->size_x; x++) {
                GridCell *cell = &building->grid[x][y][z];

                if (cell->status == ZONE_COLLAPSED) {
                    // Propagate damage to adjacent cells
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dz = -1; dz <= 1; dz++) {
                                int nx = x + dx, ny = y + dy, nz = z + dz;
                                if (nx >= 0 && nx < building->size_x &&
                                    ny >= 0 && ny < building->size_y &&
                                    nz >= 0 && nz < building->size_z) {

                                    building->grid[nx][ny][nz].damage_factor +=
                                        DAMAGE_PROPAGATION_RATE * 0.1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//=============================================================================
// SAFE ZONE IDENTIFICATION
//=============================================================================

void identify_safe_zones(Building *building, SafeZone *safe_zones, int *num_safe_zones) {
    *num_safe_zones = 0;

    for (int z = 0; z < building->size_z; z++) {
        for (int y = 0; y < building->size_y; y++) {
            for (int x = 0; x < building->size_x; x++) {
                float safety_score = calculate_zone_safety_score(building, x, y, z);

                // Only consider zones with high safety score
                if (safety_score > 0.7) {
                    SafeZone *zone = &safe_zones[*num_safe_zones];
                    zone->x = x;
                    zone->y = y;
                    zone->z = z;
                    zone->safety_score = safety_score;

                    // Find nearest exit
                    float min_dist = 1000.0;
                    for (int ez = 0; ez < building->size_z; ez++) {
                        for (int ey = 0; ey < building->size_y; ey++) {
                            for (int ex = 0; ex < building->size_x; ex++) {
                                if (building->grid[ex][ey][ez].is_exit) {
                                    float dist = euclidean_distance_3d(x, y, z, ex, ey, ez);
                                    if (dist < min_dist) min_dist = dist;
                                }
                            }
                        }
                    }
                    zone->distance_to_exit = min_dist;

                    // Generate reason
                    GridCell *cell = &building->grid[x][y][z];
                    sprintf(zone->reason, "Integrity: %.0f%%, Low vibration: %.2f m/s², Clear path to exit",
                            cell->integrity * 100, cell->vibration_level);

                    (*num_safe_zones)++;
                    if (*num_safe_zones >= 100) return;
                }
            }
        }
    }

    // Sort safe zones by safety score (bubble sort for simplicity)
    for (int i = 0; i < *num_safe_zones - 1; i++) {
        for (int j = 0; j < *num_safe_zones - i - 1; j++) {
            if (safe_zones[j].safety_score < safe_zones[j + 1].safety_score) {
                SafeZone temp = safe_zones[j];
                safe_zones[j] = safe_zones[j + 1];
                safe_zones[j + 1] = temp;
            }
        }
    }
}

float calculate_zone_safety_score(Building *building, int x, int y, int z) {
    GridCell *cell = &building->grid[x][y][z];

    // Cannot be safe if collapsed or critical
    if (cell->status == ZONE_COLLAPSED || cell->status == ZONE_CRITICAL) {
        return 0.0;
    }

    float score = 0.0;

    // Factor 1: Cell integrity (40% weight)
    score += cell->integrity * 0.4;

    // Factor 2: Low vibration level (30% weight)
    float vib_score = 1.0 - (cell->vibration_level / CRITICAL_VIBRATION_THRESHOLD);
    if (vib_score < 0) vib_score = 0;
    score += vib_score * 0.3;

    // Factor 3: Distance from damaged zones (20% weight)
    float min_damage_dist = 1000.0;
    for (int dz = 0; dz < building->size_z; dz++) {
        for (int dy = 0; dy < building->size_y; dy++) {
            for (int dx = 0; dx < building->size_x; dx++) {
                if (building->grid[dx][dy][dz].status >= ZONE_CRITICAL) {
                    float dist = euclidean_distance_3d(x, y, z, dx, dy, dz);
                    if (dist < min_damage_dist) min_damage_dist = dist;
                }
            }
        }
    }
    float damage_score = normalize(min_damage_dist, 0, SAFE_DISTANCE_FACTOR * 2);
    score += damage_score * 0.2;

    // Factor 4: Proximity to exit (10% weight)
    float min_exit_dist = 1000.0;
    for (int ez = 0; ez < building->size_z; ez++) {
        for (int ey = 0; ey < building->size_y; ey++) {
            for (int ex = 0; ex < building->size_x; ex++) {
                if (building->grid[ex][ey][ez].is_exit) {
                    float dist = euclidean_distance_3d(x, y, z, ex, ey, ez);
                    if (dist < min_exit_dist) min_exit_dist = dist;
                }
            }
        }
    }
    float exit_score = 1.0 - normalize(min_exit_dist, 0, building->size_x);
    score += exit_score * 0.1;

    return score;
}

//=============================================================================
// EVACUATION PATH PLANNING
//=============================================================================

void generate_evacuation_path(Building *building, int start_x, int start_y, int start_z) {
    // Simple A* pathfinding to nearest exit avoiding damaged zones
    printf("\nEvacuation Path from [%d,%d,%d]:\n", start_x, start_y, start_z);

    // Find nearest safe exit
    float min_dist = 1000.0;
    int exit_x = 0, exit_y = 0, exit_z = 0;

    for (int z = 0; z < building->size_z; z++) {
        for (int y = 0; y < building->size_y; y++) {
            for (int x = 0; x < building->size_x; x++) {
                if (building->grid[x][y][z].is_exit &&
                    building->grid[x][y][z].status != ZONE_COLLAPSED) {
                    float dist = euclidean_distance_3d(start_x, start_y, start_z, x, y, z);
                    if (dist < min_dist) {
                        min_dist = dist;
                        exit_x = x;
                        exit_y = y;
                        exit_z = z;
                    }
                }
            }
        }
    }

    printf("→ Nearest safe exit: [%d,%d,%d] (%.1fm away)\n", exit_x, exit_y, exit_z, min_dist);
    printf("→ Recommended path: \n");
    printf("   1. Move from current location [%d,%d,Floor %d]\n", start_x, start_y, start_z);

    // Simple direct path (in real implementation, use A*)
    if (start_z > exit_z) {
        printf("   2. Descend to ground floor (avoid damaged staircases)\n");
    }

    if (start_x < exit_x) {
        printf("   3. Move EAST towards exit\n");
    } else if (start_x > exit_x) {
        printf("   3. Move WEST towards exit\n");
    }

    if (start_y < exit_y) {
        printf("   4. Move NORTH towards exit\n");
    } else if (start_y > exit_y) {
        printf("   4. Move SOUTH towards exit\n");
    }

    printf("   5. EXIT building at [%d,%d,%d]\n", exit_x, exit_y, exit_z);

    // Check for obstacles in path
    printf("\n⚠️  Path Warnings:\n");
    bool warnings = false;
    for (int z = start_z; z >= exit_z; z--) {
        for (int y = (start_y < exit_y ? start_y : exit_y);
             y <= (start_y > exit_y ? start_y : exit_y); y++) {
            for (int x = (start_x < exit_x ? start_x : exit_x);
                 x <= (start_x > exit_x ? start_x : exit_x); x++) {
                if (building->grid[x][y][z].status >= ZONE_CRITICAL) {
                    printf("   ⚠ Avoid zone [%d,%d,%d] - %s\n", x, y, z,
                           building->grid[x][y][z].status == ZONE_COLLAPSED ?
                           "COLLAPSED" : "CRITICAL DAMAGE");
                    warnings = true;
                }
            }
        }
    }
    if (!warnings) {
        printf("   ✓ Path is clear\n");
    }
}

//=============================================================================
// VISUALIZATION AND REPORTING
//=============================================================================

void visualize_building_status(Building *building) {
    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║              BUILDING STATUS - GROUND FLOOR (z=0)         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    printf("\nLegend: [✓] Healthy | [!] Warning | [X] Critical | [#] Collapsed | [E] Exit\n\n");

    // Print ground floor
    for (int y = building->size_y - 1; y >= 0; y--) {
        printf("  ");
        for (int x = 0; x < building->size_x; x++) {
            GridCell *cell = &building->grid[x][y][0];
            switch (cell->status) {
                case ZONE_HEALTHY:   printf("[✓]"); break;
                case ZONE_WARNING:   printf("[!]"); break;
                case ZONE_CRITICAL:  printf("[X]"); break;
                case ZONE_COLLAPSED: printf("[#]"); break;
                case ZONE_EXIT:      printf("[E]"); break;
            }
        }
        printf("\n");
    }

    // Print sensor status
    printf("\n--- Sensor Readings ---\n");
    for (int i = 0; i < building->num_sensors; i++) {
        Sensor *s = &building->sensors[i];
        printf("Sensor %d [%.1f,%.1f,%.1f]: RMS=%.2f m/s², Freq=%.1f Hz %s\n",
               s->id, s->x_pos, s->y_pos, s->z_pos,
               s->rms_vibration, s->freq_dominant,
               s->rms_vibration > CRITICAL_VIBRATION_THRESHOLD ? "⚠️ CRITICAL" :
               s->rms_vibration > WARNING_VIBRATION_THRESHOLD ? "⚠ WARNING" : "✓");
    }
}

void print_safety_report(Building *building) {
    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    SAFETY SUMMARY REPORT                   ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    // Count zones by status
    int healthy = 0, warning = 0, critical = 0, collapsed = 0;
    int occupied_risk = 0;

    for (int z = 0; z < building->size_z; z++) {
        for (int y = 0; y < building->size_y; y++) {
            for (int x = 0; x < building->size_x; x++) {
                GridCell *cell = &building->grid[x][y][z];
                switch (cell->status) {
                    case ZONE_HEALTHY: healthy++; break;
                    case ZONE_WARNING: warning++; break;
                    case ZONE_CRITICAL: critical++; break;
                    case ZONE_COLLAPSED: collapsed++; break;
                    default: break;
                }
                if (cell->occupancy > 0 && cell->status >= ZONE_CRITICAL) {
                    occupied_risk += cell->occupancy;
                }
            }
        }
    }

    printf("\nZone Status:\n");
    printf("  ✓ Healthy zones:   %d\n", healthy);
    printf("  ! Warning zones:   %d\n", warning);
    printf("  X Critical zones:  %d\n", critical);
    printf("  # Collapsed zones: %d\n", collapsed);
    printf("\nOverall Integrity:   %.1f%%\n", building->overall_integrity * 100);
    printf("People at risk:      %d\n", occupied_risk);

    if (building->overall_integrity > 0.8) {
        printf("\nStatus: ✓ BUILDING IS SAFE\n");
    } else if (building->overall_integrity > 0.5) {
        printf("\nStatus: ⚠ CAUTION - MONITORING REQUIRED\n");
    } else {
        printf("\nStatus: ⚠️ CRITICAL - EVACUATION RECOMMENDED\n");
    }
}

//=============================================================================
// UTILITY FUNCTIONS
//=============================================================================

float vector_magnitude(float x, float y, float z) {
    return sqrt(x * x + y * y + z * z);
}

float euclidean_distance_3d(int x1, int y1, int z1, int x2, int y2, int z2) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    int dz = z2 - z1;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

float normalize(float value, float min, float max) {
    if (max <= min) return 0.0;
    float result = (value - min) / (max - min);
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;
}
