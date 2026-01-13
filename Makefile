# Makefile for Structure Integrity Prediction System
# Author: ultrawork
# Date: 2026-01-13

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
LDFLAGS = -lm
TARGET = structure_integrity
SOURCE = structure_integrity_prediction.c

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	@echo "Compiling Structure Integrity Prediction System..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo "✓ Build successful! Run with: ./$(TARGET)"

# Run the program
run: $(TARGET)
	@echo "Running Structure Integrity Prediction System..."
	@echo ""
	./$(TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	@echo "✓ Clean complete"

# Debug build with symbols
debug: CFLAGS += -g -DDEBUG
debug: clean $(TARGET)
	@echo "✓ Debug build complete"

# Help message
help:
	@echo "Structure Integrity Prediction System - Build Instructions"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build the program"
	@echo "  make run      - Build and run the program"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make debug    - Build with debug symbols"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  make && ./structure_integrity"

.PHONY: all run clean debug help
