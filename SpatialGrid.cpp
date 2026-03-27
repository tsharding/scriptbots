#include "SpatialGrid.h"
#include "settings.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

SpatialGrid::SpatialGrid(float worldWidth, float worldHeight, float cellSize)
    : worldWidth(worldWidth), worldHeight(worldHeight), cellSize(cellSize)
{
    gridWidth = static_cast<int>(std::ceil(worldWidth / cellSize));
    gridHeight = static_cast<int>(std::ceil(worldHeight / cellSize));
    
    // Initialize grid
    grid.resize(gridWidth);
    for (int i = 0; i < gridWidth; i++) {
        grid[i].resize(gridHeight);
    }
}

SpatialGrid::~SpatialGrid() {
    clear();
}

void SpatialGrid::clear() {
    for (int i = 0; i < gridWidth; i++) {
        for (int j = 0; j < gridHeight; j++) {
            grid[i][j].clear();
        }
    }
    agentCellMap.clear();
}

int SpatialGrid::getGridX(float x) const {
    // Handle world wrapping
    while (x < 0) x += worldWidth;
    while (x >= worldWidth) x -= worldWidth;
    
    int gx = static_cast<int>(x / cellSize);
    return std::max(0, std::min(gx, gridWidth - 1));
}

int SpatialGrid::getGridY(float y) const {
    // Handle world wrapping
    while (y < 0) y += worldHeight;
    while (y >= worldHeight) y -= worldHeight;
    
    int gy = static_cast<int>(y / cellSize);
    return std::max(0, std::min(gy, gridHeight - 1));
}

bool SpatialGrid::isValidCell(int gx, int gy) const {
    return gx >= 0 && gx < gridWidth && gy >= 0 && gy < gridHeight;
}

void SpatialGrid::addAgent(Agent* agent) {
    if (!agent) return;
    
    int gx = getGridX(agent->pos.x);
    int gy = getGridY(agent->pos.y);
    
    grid[gx][gy].push_back(agent);
    agentCellMap[agent] = std::make_pair(gx, gy);
}

void SpatialGrid::removeAgent(Agent* agent) {
    if (!agent) return;
    
    auto it = agentCellMap.find(agent);
    if (it != agentCellMap.end()) {
        int gx = it->second.first;
        int gy = it->second.second;
        
        auto& cellAgents = grid[gx][gy];
        cellAgents.erase(std::remove(cellAgents.begin(), cellAgents.end(), agent), cellAgents.end());
        
        agentCellMap.erase(it);
    }
}

void SpatialGrid::updateAgent(Agent* agent, float oldX, float oldY) {
    if (!agent) return;
    
    int oldGx = getGridX(oldX);
    int oldGy = getGridY(oldY);
    int newGx = getGridX(agent->pos.x);
    int newGy = getGridY(agent->pos.y);
    
    // Only update if agent moved to a different cell
    if (oldGx != newGx || oldGy != newGy) {
        // Remove from old cell
        auto& oldCellAgents = grid[oldGx][oldGy];
        oldCellAgents.erase(std::remove(oldCellAgents.begin(), oldCellAgents.end(), agent), oldCellAgents.end());
        
        // Add to new cell
        grid[newGx][newGy].push_back(agent);
        agentCellMap[agent] = std::make_pair(newGx, newGy);
    }
}

void SpatialGrid::getNearbyAgents(float x, float y, float radius, std::vector<Agent*>& result) {
    result.clear();
    
    // Calculate the range of cells to check
    int minGx = getGridX(x - radius);
    int maxGx = getGridX(x + radius);
    int minGy = getGridY(y - radius);
    int maxGy = getGridY(y + radius);
    
    // Handle world wrapping for grid boundaries
    for (int gx = minGx; gx <= maxGx; gx++) {
        for (int gy = minGy; gy <= maxGy; gy++) {
            int wrappedGx = gx;
            int wrappedGy = gy;
            
            // Handle wrapping
            if (wrappedGx < 0) wrappedGx += gridWidth;
            if (wrappedGx >= gridWidth) wrappedGx -= gridWidth;
            if (wrappedGy < 0) wrappedGy += gridHeight;
            if (wrappedGy >= gridHeight) wrappedGy -= gridHeight;
            
            if (isValidCell(wrappedGx, wrappedGy)) {
                const auto& cellAgents = grid[wrappedGx][wrappedGy];
                for (Agent* agent : cellAgents) {
                    // Calculate distance considering world wrapping
                    float dx = agent->pos.x - x;
                    float dy = agent->pos.y - y;
                    
                    // Handle wrapping in distance calculation
                    if (dx > worldWidth/2) dx -= worldWidth;
                    else if (dx < -worldWidth/2) dx += worldWidth;
                    if (dy > worldHeight/2) dy -= worldHeight;
                    else if (dy < -worldHeight/2) dy += worldHeight;
                    
                    float distance = std::sqrt(dx*dx + dy*dy);
                    if (distance <= radius) {
                        result.push_back(agent);
                    }
                }
            }
        }
    }
}

void SpatialGrid::getAgentsInRegion(float x, float y, std::vector<Agent*>& result) {
    result.clear();
    
    int centerGx = getGridX(x);
    int centerGy = getGridY(y);
    
    // Check center cell and 8 neighboring cells
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int gx = centerGx + dx;
            int gy = centerGy + dy;
            
            // Handle wrapping
            if (gx < 0) gx += gridWidth;
            else if (gx >= gridWidth) gx -= gridWidth;
            if (gy < 0) gy += gridHeight;
            else if (gy >= gridHeight) gy -= gridHeight;
            
            if (isValidCell(gx, gy)) {
                const auto& cellAgents = grid[gx][gy];
                result.insert(result.end(), cellAgents.begin(), cellAgents.end());
            }
        }
    }
}

void SpatialGrid::getStats(int& totalCells, int& occupiedCells, float& avgAgentsPerCell, int& maxAgentsInCell) {
    totalCells = gridWidth * gridHeight;
    occupiedCells = 0;
    int totalAgents = 0;
    maxAgentsInCell = 0;
    
    for (int i = 0; i < gridWidth; i++) {
        for (int j = 0; j < gridHeight; j++) {
            int cellCount = static_cast<int>(grid[i][j].size());
            if (cellCount > 0) {
                occupiedCells++;
                totalAgents += cellCount;
                maxAgentsInCell = std::max(maxAgentsInCell, cellCount);
            }
        }
    }
    
    avgAgentsPerCell = occupiedCells > 0 ? static_cast<float>(totalAgents) / occupiedCells : 0.0f;
} 