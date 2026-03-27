CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -DHAVE_OPENMP
GLUT_FLAGS = $(shell pkg-config --cflags --libs glut)
OPENGL_FLAGS = -lGL -lGLU
OPENMP_FLAGS = -fopenmp

# Source files
SOURCES = GLView.cpp StatsWindow.cpp main.cpp MLPBrain.cpp Agent.cpp World.cpp vmath.cpp Config.cpp SpatialGrid.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
TARGET = scriptbots

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(GLUT_FLAGS) $(OPENGL_FLAGS) $(OPENMP_FLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Generate version files
generate-version:
	chmod +x generate_version.sh
	./generate_version.sh

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt update
	sudo apt install -y build-essential freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev

.PHONY: all clean install-deps 