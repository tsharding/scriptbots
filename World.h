#ifndef WORLD_H
#define WORLD_H

#include "Agent.h"
#include "SpatialGrid.h"
#include "settings.h"
#include <vector>
#include <string>
#include <fstream>

// Forward declaration
class GLView;

class World
{
public:
    World();
    ~World();
    
    void update();
    void reset();
    
    void draw(GLView* view, bool drawfood);
    
    bool isClosed() const;
    void setClosed(bool close);
    
    /**
     * Returns the number of herbivores and 
     * carnivores in the world.
     * first : num herbs
     * second : num carns
     */
    std::pair<int,int> numHerbCarnivores() const;
    
    int numAgents() const;
    int epoch() const;
    
    //mouse interaction
    void processMouse(int button, int state, int x, int y);

    void addNewByCrossover();
    void addRandomBots(int num);
    void addCarnivore();
    void addHerbivore();
    void addRandomAgents(int num);
    
    void positionOfInterest(int type, float &xi, float &yi);
    
    // Save/Load functionality
    std::string getSaveDirectory() const;
    bool saveToFile(const std::string& filename);
    bool loadFromFile(const std::string& filename);
    
    std::vector<int> numCarnivore;
    std::vector<int> numHerbivore;
    
    // Food statistics methods
    float getTotalFood() const;
    float getFoodTilePercentage() const;
    
    // Agent access methods
    const std::vector<Agent>& getAgents() const { return agents; }
    
private:
    void setInputs();
    void processOutputs();
    void brainsTick();  //takes in[] to out[] for every agent
    
    void writeReport();
    
    void reproduce(int ai, float MR, float MR2);
    
    int modcounter;
    int current_epoch;
    int idcounter;
    
    std::vector<Agent> agents;
    
    // Spatial grid for efficient neighbor queries
    SpatialGrid* spatialGrid;
    
    // food
    int FW;
    int FH;
    int fx;
    int fy;
    std::vector<std::vector<float>> food;
    bool CLOSED; //if environment is closed, then no random bots are added per time interval
};

#endif // WORLD_H
