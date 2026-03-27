#ifndef SPATIALGRID_H
#define SPATIALGRID_H

#include <vector>
#include <unordered_map>
#include "Agent.h"

/**
 * Spatial grid for efficient neighbor queries.
 * Divides the world into a grid and tracks which agents are in each cell.
 * This reduces O(N²) operations to O(N) for neighbor finding.
 */
class SpatialGrid {
public:
    SpatialGrid(float worldWidth, float worldHeight, float cellSize);
    ~SpatialGrid();
    
    // Clear all agents from the grid
    void clear();
    
    // Add an agent to the grid based on its position
    void addAgent(Agent* agent);
    
    // Remove an agent from the grid
    void removeAgent(Agent* agent);
    
    // Update agent position in grid (call when agent moves)
    void updateAgent(Agent* agent, float oldX, float oldY);
    
    // Get all agents within interaction distance of the given position
    void getNearbyAgents(float x, float y, float radius, std::vector<Agent*>& result);
    
    // Get all agents in the same cell and neighboring cells
    void getAgentsInRegion(float x, float y, std::vector<Agent*>& result);
    
    // Get statistics for debugging
    void getStats(int& totalCells, int& occupiedCells, float& avgAgentsPerCell, int& maxAgentsInCell);

private:
    float worldWidth, worldHeight;
    float cellSize;
    int gridWidth, gridHeight;
    
    // Grid of agent lists
    std::vector<std::vector<std::vector<Agent*>>> grid;
    
    // Helper functions
    int getGridX(float x) const;
    int getGridY(float y) const;
    bool isValidCell(int gx, int gy) const;
    
    // Agent tracking for efficient removal
    std::unordered_map<Agent*, std::pair<int, int>> agentCellMap;
};

#endif // SPATIALGRID_H 