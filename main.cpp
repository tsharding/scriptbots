#include "GLView.h"
#include "StatsWindow.h"
#include "World.h"
#include "Config.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include <GL/glut.h>

#include <stdio.h>

// Version functions
std::string getVersion() {
    std::ifstream file("version.txt");
    if (file.is_open()) {
        std::string version;
        std::getline(file, version);
        file.close();
        // Remove any whitespace and newlines
        version.erase(std::remove_if(version.begin(), version.end(), ::isspace), version.end());
        return version;
    }
    return "2.0.0"; // fallback
}

std::string getBuildTag() {
    static std::string buildTag = "";
    if (buildTag.empty()) {
        // Use current time to generate a pseudo-random 7-character string
        time_t now = time(nullptr);
        srand(now);
        const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        buildTag = "";
        for (int i = 0; i < 7; ++i) {
            buildTag += charset[rand() % (sizeof(charset) - 1)];
        }
    }
    return buildTag;
}

std::string getFullVersion() {
    std::string version = getVersion();
    std::string buildTag = getBuildTag();
    return "v" + version + "+" + buildTag;
}

// Helper function to calculate window positions that ensure top bars are visible
void calculateWindowPositions(int worldWidth, int statsWidth, 
                             int& worldX, int& worldY, int& statsX, int& statsY) {
    // Get screen dimensions
    int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
    
    // Calculate world window position (top right corner at middle of top right quadrant)
    // Top right quadrant center: (3/4 * screenWidth, 1/4 * screenHeight)
    worldX = (3 * screenWidth / 4) - worldWidth;
    worldY = (screenHeight / 4) - 30; // 30 pixels from top to account for window decorations
    
    // Calculate stats window position (a little bit away from top right of screen)
    statsX = screenWidth - statsWidth - 50; // 50 pixels from right edge
    statsY = 50; // 50 pixels from top
    
    // Ensure windows don't go off screen
    if (worldX < 0) worldX = 0;
    if (worldY < 0) worldY = 0;
    if (statsX < 0) statsX = 0;
    if (statsY < 0) statsY = 0;
    
    // Ensure windows don't overlap too much
    if (worldX + worldWidth > statsX) {
        worldX = statsX - worldWidth - 20; // 20 pixel gap
        if (worldX < 0) worldX = 0;
    }
}

GLView* GLVIEW = new GLView(0);
StatsWindow* STATSWINDOW = nullptr;

int main(int argc, char **argv) {
    srand(time(0));
    
    // Load configuration
    if (!g_config.load()) {
        printf("Warning: Could not load configuration file. Using default values.\n");
    }
    // Cache values from loaded config (or defaults if loading failed)
    g_config.cacheValues();
    
    if (conf::WIDTH()%conf::CZ()!=0 || conf::HEIGHT()%conf::CZ()!=0) printf("CAREFUL! The cell size variable conf::CZ should divide evenly into both conf::WIDTH and conf::HEIGHT! It doesn't right now!");
    
    World* world = nullptr;
    
    // Handle command-line arguments
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "--help") {
            // Help requested
            printf("ScriptBots %s\n", getFullVersion().c_str());
            printf("ScriptBots now runs in two windows:\n");
            printf("- Main window: World visualization\n");
            printf("- Stats window: Population chart and controls\n");
            printf("Both windows respond to keyboard input.\n");
            printf("Usage: %s [--help] [--version] [--load <save_file>]\n", argv[0]);
            printf("  --help: Show this help message and exit\n");
            printf("  --version: Show version information and exit\n");
            printf("  --load <save_file>: Load a save file from the save directory on startup (e.g., manual_save_0.sav)\n");
            return 0;
        } else if (arg1 == "--version") {
            // Version requested
            printf("ScriptBots %s\n", getFullVersion().c_str());
            return 0;
        } else if (arg1 == "--load") {
            if (argc < 3) {
                printf("ERROR: --load requires a filename argument.\n");
                return 1;
            }
            std::string saveFile = argv[2];
            printf("Attempting to load save file: %s\n", saveFile.c_str());
            world = new World();
            GLVIEW->setWorld(world);
            if (world->loadFromFile(saveFile)) {
                printf("Successfully loaded save file: %s\n", saveFile.c_str());
            } else {
                printf("ERROR: Failed to load save file: %s\n", saveFile.c_str());
                printf("Exiting. Please check the file name and try again.\n");
                exit(1);
            }
            // Continue to GLUT setup below
        } else {
            printf("ERROR: Unknown argument '%s'. Use --help for usage.\n", arg1.c_str());
            return 1;
        }
    } else {
        // No arguments provided, create new simulation
        printf("ScriptBots %s\n", getFullVersion().c_str());
        printf("ScriptBots now runs in two windows:\n");
        printf("- Main window: World visualization\n");
        printf("- Stats window: Population chart and controls\n");
        printf("Both windows respond to keyboard input.\n");
        printf("Usage: %s [--help] [--version] [--load <save_file>]\n", argv[0]);
        printf("  --help: Show this help message and exit\n");
        printf("  --version: Show version information and exit\n");
        printf("  --load <save_file>: Load a save file from the save directory on startup (e.g., manual_save_0.sav)\n");
        
        world = new World();
        GLVIEW->setWorld(world);
    }
    
    //GLUT SETUP
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    
    // Calculate proper window positions to ensure top bars are visible
    int worldX, worldY, statsX, statsY;
    calculateWindowPositions(conf::WWIDTH(), 500, 
                           worldX, worldY, statsX, statsY);
    
    printf("Screen dimensions: %dx%d\n", glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT));
    printf("World window position: (%d, %d)\n", worldX, worldY);
    printf("Stats window position: (%d, %d)\n", statsX, statsY);
    
    // Create main world window
    glutInitWindowPosition(worldX, worldY);
    glutInitWindowSize(conf::WWIDTH(),conf::WHEIGHT());
    int mainWindowId = glutCreateWindow(("ScriptBots " + getFullVersion() + " - World View").c_str());
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glutDisplayFunc(gl_renderScene);
    glutIdleFunc(gl_handleIdle);
    glutReshapeFunc(gl_changeSize);
    glutKeyboardFunc(gl_processNormalKeys);
    glutMouseFunc(gl_processMouse);
    glutMotionFunc(gl_processMouseActiveMotion);

    // Create stats window
    STATSWINDOW = new StatsWindow(world);
    StatsWindow::createWindow(statsX, statsY);
    
    // Set the main window as the current window
    glutSetWindow(mainWindowId);

    glutMainLoop();
    return 0;
}
