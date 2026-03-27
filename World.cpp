#include "World.h"
#include "GLView.h"
#include "StatsWindow.h"

#include <ctime>

#include "settings.h"
#include "helpers.h"
#include "vmath.h"
#include <stdio.h>

using namespace std;

World::World() :
        modcounter(0),
        current_epoch(0),
        idcounter(0),
        FW(conf::WIDTH()/conf::CZ()),
        FH(conf::HEIGHT()/conf::CZ()),
        CLOSED(conf::INITIAL_CLOSED_ENVIRONMENT())
{
    // Initialize spatial grid for efficient neighbor queries
    // Cell size should be roughly the interaction distance for optimal performance
    float cellSize = conf::DIST() * 0.8f; // Slightly smaller than vision distance
    spatialGrid = new SpatialGrid(conf::WIDTH(), conf::HEIGHT(), cellSize);
    
    // Initialize food layer
    food.resize(FW, std::vector<float>(FH, 0));
    
    // Fill random food tiles based on configuration
    int totalTiles = FW * FH;
    int tilesToFill = static_cast<int>(totalTiles * conf::PROP_INIT_FOOD_FILLED());
    
    printf("Initializing world with %d food tiles (%.1f%% of total tiles)\n", 
           tilesToFill, conf::PROP_INIT_FOOD_FILLED() * 100.0f);
    
    for (int i = 0; i < tilesToFill; i++) {
        int fx = randi(0, FW);
        int fy = randi(0, FH);
        food[fx][fy] = conf::FOODMAX();
    }
    
    // Add initial agents after food is placed
    addRandomBots(conf::NUMBOTS());
    
    numCarnivore.resize(200, 0);
    numHerbivore.resize(200, 0);
}

World::~World() {
    delete spatialGrid;
}

void World::update()
{
    modcounter++;

    //Process periodic events
    //Age goes up!
    if (modcounter%100==0) {
        for (int i=0;i<agents.size();i++) {
            agents[i].age+= 1;    //agents age...
        }        
    }
    
    if(modcounter%1000==0){
        std::pair<int,int> num_herbs_carns = numHerbCarnivores();
        
        // Shift all data left by one position
        for(int i = 0; i < numHerbivore.size() - 1; i++) {
            numHerbivore[i] = numHerbivore[i + 1];
            numCarnivore[i] = numCarnivore[i + 1];
        }
        
        // Add new data at the rightmost position (end of array)
        numHerbivore[numHerbivore.size() - 1] = num_herbs_carns.first;
        numCarnivore[numCarnivore.size() - 1] = num_herbs_carns.second;
    }
    if (modcounter%1000==0) writeReport();
    if (modcounter>=10000) {
        modcounter=0;
        current_epoch++;
        
        // Spawn random agents at specified epoch intervals
        if (current_epoch % conf::RANDOM_SPAWN_EPOCH_INTERVAL() == 0) {
            printf("Epoch %d: Spawning %d random agents\n", current_epoch, conf::RANDOM_SPAWN_COUNT());
            addRandomAgents(conf::RANDOM_SPAWN_COUNT());
        }
        
        // Population recovery: if carnivores or herbivores are at 0, spawn new random agents of that type
        std::pair<int,int> num_herbs_carns = numHerbCarnivores();
        
        if (num_herbs_carns.second == 0) { // No carnivores
            printf("Carnivore population extinct! Spawning %d new carnivores.\n", conf::CARNIVORE_EXTINCTION_REPOPULATION_COUNT());
            for (int i = 0; i < conf::CARNIVORE_EXTINCTION_REPOPULATION_COUNT(); i++) {
                addCarnivore();
            }
        }
        
        if (num_herbs_carns.first == 0) { // No herbivores
            printf("Herbivore population extinct! Spawning %d new herbivores.\n", conf::HERBIVORE_EXTINCTION_REPOPULATION_COUNT());
            for (int i = 0; i < conf::HERBIVORE_EXTINCTION_REPOPULATION_COUNT(); i++) {
                addHerbivore();
            }
        }
    }
    
    if(modcounter%1000==0){
        // Auto-save based on frequency setting
        if (modcounter == 0 && current_epoch % conf::AUTOSAVE_FREQUENCY() == 0 && current_epoch > 0) {
            std::string autosaveName = "autosave_epoch_" + std::to_string(current_epoch) + ".sav";
            saveToFile(autosaveName);
            printf("Auto-saved simulation state to %s\n", autosaveName.c_str());
        }
    }
    
    // Food addition - happens at the configured frequency
    if (modcounter%conf::FOODADDFREQ()==0) {
        fx=randi(0,FW);
        fy=randi(0,FH);
        food[fx][fy]= conf::FOODMAX();
    }
    
    //reset any counter variables per agent
    for(int i=0;i<agents.size();i++){
        agents[i].spiked= false;
    }

    //give input to every agent. Sets in[] array
    setInputs();

    //brains tick. computes in[] -> out[]
    brainsTick();

    //read output and process consequences of bots on environment. requires out[]
    processOutputs();

    //process bots: health and deaths
    for (int i=0;i<agents.size();i++) {
        float baseloss= 0.0002; // + 0.0001*(abs(agents[i].w1) + abs(agents[i].w2))/2;
        //if (agents[i].w1<0.1 && agents[i].w2<0.1) baseloss=0.0001; //hibernation :p
        //baseloss += 0.00005*agents[i].soundmul; //shouting costs energy. just a tiny bit

        if (agents[i].boost) {
            //boost carries its price, and it's pretty heavy!
            agents[i].health -= baseloss*conf::BOOSTSIZEMULT()*1.3;
        } else {
            agents[i].health -= baseloss;
        }
    }
    
    //process indicator (used in drawing)
    for (int i=0;i<agents.size();i++){
        if(agents[i].indicator>0) agents[i].indicator -= 1;
    }

    //remove dead agents.
    //first distribute foods
    for (int i=0;i<agents.size();i++) {
        //if this agent was spiked this round as well (i.e. killed). This will make it so that
        //natural deaths can't be capitalized on. I feel I must do this or otherwise agents
        //will sit on spot and wait for things to die around them. They must do work!
        if (agents[i].health<=0 && agents[i].spiked) { 
        
            //distribute its food. It will be erased soon
            //first figure out how many are around, to distribute this evenly
            int numaround=0;
            for (int j=0;j<agents.size();j++) {
                if (agents[j].health>0) {
                    float d= (agents[i].pos-agents[j].pos).length();
                    if (d<conf::FOOD_DISTRIBUTION_RADIUS()) {
                        numaround++;
                    }
                }
            }
            
            //young killed agents should give very little resources
            //at age 5, they mature and give full. This can also help prevent
            //agents eating their young right away
            float agemult= 1.0;
            if(agents[i].age<5) agemult= agents[i].age*0.2;
            
            if (numaround>0) {
                //distribute its food evenly
                for (int j=0;j<agents.size();j++) {
                    if (agents[j].health>0) {
                        float d= (agents[i].pos-agents[j].pos).length();
                        if (d<conf::FOOD_DISTRIBUTION_RADIUS()) {
                            agents[j].health += 5*(1-agents[j].herbivore)*(1-agents[j].herbivore)/pow(numaround,1.25)*agemult;
                            agents[j].repcounter -= conf::REPMULT()*(1-agents[j].herbivore)*(1-agents[j].herbivore)/pow(numaround,1.25)*agemult; //good job, can use spare parts to make copies
                            if (agents[j].health>2) agents[j].health=2; //cap it!
                            agents[j].initEvent(30,1,1,1); //white means they ate! nice
                        }
                    }
                }
            }

        }
    }
    vector<Agent>::iterator iter= agents.begin();
    while (iter != agents.end()) {
        if (iter->health <=0) {
            // Remove from spatial grid before erasing from vector
            spatialGrid->removeAgent(&(*iter));
            iter= agents.erase(iter);
        } else {
            ++iter;
        }
    }
    
    // Track lineage extinctions after agents are removed
    if (STATSWINDOW) {
        // Get current lineage populations
        std::map<std::string, int> currentLineagePopulations;
        for (const auto& agent : agents) {
            currentLineagePopulations[agent.lineageTag]++;
        }
        
        // Check for extinctions by comparing with previous state
        static std::map<std::string, int> previousLineagePopulations;
        for (const auto& pair : previousLineagePopulations) {
            const std::string& lineageTag = pair.first;
            int previousPop = pair.second;
            int currentPop = currentLineagePopulations[lineageTag];
            
            // Update max population before extinction
            STATSWINDOW->trackCurrentPopulation(lineageTag, previousPop);
            
            // If population went from >0 to 0, lineage went extinct
            if (previousPop > 0 && currentPop == 0) {
                STATSWINDOW->trackLineageExtinction(lineageTag, current_epoch, modcounter);
            }
        }
        
        // Update previous state
        previousLineagePopulations = currentLineagePopulations;
    }

    //handle reproduction
    for (int i=0;i<agents.size();i++) {
        if (agents[i].repcounter<0 && agents[i].health>0.65 && modcounter%15==0 && randf(0,1)<0.1) { //agent is healthy and is ready to reproduce. Also inject a bit non-determinism
            //agents[i].health= 0.8; //the agent is left vulnerable and weak, a bit
            reproduce(i, agents[i].MUTRATE1, agents[i].MUTRATE2); //this adds conf::BABIES() new agents to agents[]
            agents[i].repcounter= agents[i].herbivore*randf(conf::REPRATEH()-0.1,conf::REPRATEH()+0.1) + (1-agents[i].herbivore)*randf(conf::REPRATEC()-0.1,conf::REPRATEC()+0.1);
        }
    }

    //add new agents, if environment isn't closed
    if (!CLOSED) {
        //make sure environment is always populated with at least NUMBOTS bots
        if (agents.size()<conf::NUMBOTS()
           ) {
            //add new agent
            addRandomBots(1);
        }
        if (modcounter%100==0) {
            if (randf(0,1)<0.5){
                addRandomBots(1); //every now and then add random bots in
            }else
                addNewByCrossover(); //or by crossover
        }
    }


}

void World::setInputs()
{
    //P1 R1 G1 B1 FOOD P2 R2 G2 B2 SOUND SMELL HEALTH P3 R3 G3 B3 CLOCK1 CLOCK 2 HEARING     BLOOD_SENSOR
    //0   1  2  3  4   5   6  7 8   9     10     11   12 13 14 15 16       17      18           19

    float PI8=M_PI/8/2; //pi/8/2
    float PI38= 3*PI8; //3pi/8/2
    float PI4= M_PI/4;
    
    // Temporary vector to store nearby agents (reused for efficiency)
    std::vector<Agent*> nearbyAgents;
    
    for (int i=0;i<agents.size();i++) {
        Agent* a= &agents[i];

        //HEALTH
        a->in[11]= cap(a->health/2); //divide by 2 since health is in [0,2]

        //FOOD
        int cx= (int) a->pos.x/conf::CZ();
        int cy= (int) a->pos.y/conf::CZ();
        a->in[4]= food[cx][cy]/conf::FOODMAX();

        //SOUND SMELL EYES
        vector<float> p(conf::NUMEYES(),0);
        vector<float> r(conf::NUMEYES(),0);
        vector<float> g(conf::NUMEYES(),0);
        vector<float> b(conf::NUMEYES(),0);
                       
        float soaccum=0;
        float smaccum=0;
        float hearaccum=0;

        //BLOOD ESTIMATOR
        float blood= 0;

        // Use spatial grid to get nearby agents instead of checking all agents
        spatialGrid->getNearbyAgents(a->pos.x, a->pos.y, conf::DIST(), nearbyAgents);
        
        for (Agent* a2 : nearbyAgents) {
            if (a == a2) continue; // Skip self

            float d= (a->pos-a2->pos).length();

            if (d<conf::DIST()) {

                //smell
                smaccum+= (conf::DIST()-d)/conf::DIST();

                //sound
                soaccum+= (conf::DIST()-d)/conf::DIST()*(max(fabs(a2->w1),fabs(a2->w2)));

                //hearing. Listening to other agents
                hearaccum+= a2->soundmul*(conf::DIST()-d)/conf::DIST();

                float ang= (a2->pos- a->pos).get_angle(); //current angle between bots
                
                for(int q=0;q<conf::NUMEYES();q++){
                    float aa = a->angle + a->eyedir[q];
                    if (aa<-M_PI) aa += 2*M_PI;
                    if (aa>M_PI) aa -= 2*M_PI;
                    
                    float diff1= aa- ang;
                    if (fabs(diff1)>M_PI) diff1= 2*M_PI- fabs(diff1);
                    diff1= fabs(diff1);
                    
                    float fov = a->eyefov[q];
                    if (diff1<fov) {
                        //we see a2 with this eye. Accumulate stats
                        float mul1= a->eyesensmod*(fabs(fov-diff1)/fov)*((conf::DIST()-d)/conf::DIST());
                        p[q] += mul1*(d/conf::DIST());
                        r[q] += mul1*a2->red;
                        g[q] += mul1*a2->gre;
                        b[q] += mul1*a2->blu;
                    }
                }
                
                //blood sensor
                float forwangle= a->angle;
                float diff4= forwangle- ang;
                if (fabs(forwangle)>M_PI) diff4= 2*M_PI- fabs(forwangle);
                diff4= fabs(diff4);
                if (diff4<PI38) {
                    float mul4= ((PI38-diff4)/PI38)*((conf::DIST()-d)/conf::DIST());
                    //if we can see an agent close with both eyes in front of us
                    blood+= mul4*(1-a2->health/2); //remember: health is in [0 2]
                    //agents with high life dont bleed. low life makes them bleed more
                }
            }
        }
        
        smaccum *= a->smellmod;
        soaccum *= a->soundmod;
        hearaccum *= a->hearmod;
        blood *= a->bloodmod;
        
        a->in[0]= cap(p[0]);
        a->in[1]= cap(r[0]);
        a->in[2]= cap(g[0]);
        a->in[3]= cap(b[0]);
        
        a->in[5]= cap(p[1]);
        a->in[6]= cap(r[1]);
        a->in[7]= cap(g[1]);
        a->in[8]= cap(b[1]);
        a->in[9]= cap(soaccum);
        a->in[10]= cap(smaccum);
        
        a->in[12]= cap(p[2]);
        a->in[13]= cap(r[2]);
        a->in[14]= cap(g[2]);
        a->in[15]= cap(b[2]);
        a->in[16]= abs(sin(modcounter/a->clockf1));
        a->in[17]= abs(sin(modcounter/a->clockf2));
        a->in[18]= cap(hearaccum);
        a->in[19]= cap(blood);
        
        a->in[20]= cap(p[3]);
        a->in[21]= cap(r[3]);
        a->in[22]= cap(g[3]);
        a->in[23]= cap(b[3]);
                
    }
}

void World::processOutputs()
{
    //assign meaning
    //LEFT RIGHT R G B SPIKE BOOST SOUND_MULTIPLIER GIVING
    // 0    1    2 3 4   5     6         7             8
    for (int i=0;i<agents.size();i++) {
        Agent* a= &agents[i];

        a->red= a->out[2];
        a->gre= a->out[3];
        a->blu= a->out[4];
        a->w1= a->out[0]; //-(2*a->out[0]-1);
        a->w2= a->out[1]; //-(2*a->out[1]-1);
        a->boost= a->out[6]>0.5;
        a->soundmul= a->out[7];
        a->give= a->out[8];

        //spike length should slowly tend towards out[5]
        float g= a->out[5];
        if (a->spikeLength<g)
            a->spikeLength+=conf::SPIKESPEED();
        else if (a->spikeLength>g)
            a->spikeLength= g; //its easy to retract spike, just hard to put it up
    }

    //move bots
    //#pragma omp parallel for
    for (int i=0;i<agents.size();i++) {
        Agent* a= &agents[i];

        // Store old position for spatial grid update
        float oldX = a->pos.x;
        float oldY = a->pos.y;

        Vector2f v(conf::BOTRADIUS()/2, 0);
        v.rotate(a->angle + M_PI/2);

        Vector2f w1p= a->pos+ v; //wheel positions
        Vector2f w2p= a->pos- v;

        float BW1= conf::BOTSPEED()*a->w1;
        float BW2= conf::BOTSPEED()*a->w2;
        if (a->boost) {
            BW1=BW1*conf::BOOSTSIZEMULT();
        }
        if (a->boost) {
            BW2=BW2*conf::BOOSTSIZEMULT();
        }

        //move bots
        Vector2f vv= w2p- a->pos;
        vv.rotate(-BW1);
        a->pos= w2p-vv;
        a->angle -= BW1;
        if (a->angle<-M_PI) a->angle= M_PI - (-M_PI-a->angle);
        vv= a->pos - w1p;
        vv.rotate(BW2);
        a->pos= w1p+vv;
        a->angle += BW2;
        if (a->angle>M_PI) a->angle= -M_PI + (a->angle-M_PI);

        //wrap around the map
        if (a->pos.x<0) a->pos.x= conf::WIDTH()+a->pos.x;
        if (a->pos.x>=conf::WIDTH()) a->pos.x= a->pos.x-conf::WIDTH();
        if (a->pos.y<0) a->pos.y= conf::HEIGHT()+a->pos.y;
        if (a->pos.y>=conf::HEIGHT()) a->pos.y= a->pos.y-conf::HEIGHT();
        
        // Update spatial grid with new position
        spatialGrid->updateAgent(a, oldX, oldY);
    }

    //process food intake for herbivors
    for (int i=0;i<agents.size();i++) {

        int cx= (int) agents[i].pos.x/conf::CZ();
        int cy= (int) agents[i].pos.y/conf::CZ();
        float f= food[cx][cy];
        if (f>0 && agents[i].health<2) {
            //agent eats the food
            float itk=min(f,conf::FOODINTAKE());
            float speedmul= (1-(abs(agents[i].w1)+abs(agents[i].w2))/2)*0.7 + 0.3;
            itk= itk*agents[i].herbivore*speedmul; //herbivores gain more from ground food
            agents[i].health+= itk;
            agents[i].repcounter -= 3*itk;
            food[cx][cy]-= min(f,conf::FOODWASTE());
        }
    }

    //process giving and receiving of food
    for (int i=0;i<agents.size();i++) {
        agents[i].dfood=0;
    }
    for (int i=0;i<agents.size();i++) {
        if (agents[i].give>0.5) {
            // Use spatial grid for food sharing instead of checking all agents
            std::vector<Agent*> nearbyForFood;
            spatialGrid->getNearbyAgents(agents[i].pos.x, agents[i].pos.y, conf::FOOD_SHARING_DISTANCE(), nearbyForFood);
            
            for (Agent* a2 : nearbyForFood) {
                if (&agents[i] == a2) continue; // Skip self
                
                float d= (agents[i].pos-a2->pos).length();
                if (d<conf::FOOD_SHARING_DISTANCE()) {
                    //initiate transfer
                    if (a2->health<2) a2->health += conf::FOODTRANSFER();
                    agents[i].health -= conf::FOODTRANSFER();
                    a2->dfood += conf::FOODTRANSFER(); //only for drawing
                    agents[i].dfood -= conf::FOODTRANSFER();
                }
            }
        }
    }

    //process spike dynamics for carnivors
    if (modcounter%2==0) { //we dont need to do this TOO often. can save efficiency here since this is n^2 op in #agents
        // Temporary vector to store collision candidates (reused for efficiency)
        std::vector<Agent*> collisionCandidates;
        
        for (int i=0;i<agents.size();i++) {

            //NOTE: herbivore cant attack. TODO: hmmmmm
            //fot now ok: I want herbivores to run away from carnivores, not kill them back
            if(agents[i].herbivore>0.8 || agents[i].spikeLength<0.2 || agents[i].w1<0.5 || agents[i].w2<0.5) continue; 
            
            // Use spatial grid to get nearby agents for collision detection
            spatialGrid->getNearbyAgents(agents[i].pos.x, agents[i].pos.y, 2*conf::BOTRADIUS(), collisionCandidates);
            
            for (Agent* a2 : collisionCandidates) {
                if (&agents[i] == a2) continue; // Skip self
                
                float d= (agents[i].pos-a2->pos).length();

                if (d<2*conf::BOTRADIUS()) {
                    //these two are in collision and agent i has extended spike and is going decent fast!
                    Vector2f v(1,0);
                    v.rotate(agents[i].angle);
                    float diff= v.angle_between(a2->pos-agents[i].pos);
                    if (fabs(diff)<M_PI/8) {
                        //bot i is also properly aligned!!! that's a hit
                        float mult=1;
                        if (agents[i].boost) mult= conf::BOOSTSIZEMULT();
                        float DMG= conf::SPIKEMULT()*agents[i].spikeLength*max(fabs(agents[i].w1),fabs(agents[i].w2))*conf::BOOSTSIZEMULT();

                        a2->health-= DMG;

                        if (agents[i].health>2) agents[i].health=2; //cap health at 2
                        agents[i].spikeLength= 0; //retract spike back down

                        agents[i].initEvent(10*DMG,1,1,0); //yellow event means bot has spiked other bot. nice!

                        Vector2f v2(1,0);
                        v2.rotate(a2->angle);
                        float adiff= v.angle_between(v2);
                        if (fabs(adiff)<M_PI/2) {
                            //this was attack from the back. Retract spike of the other agent (startle!)
                            //this is done so that the other agent cant right away "by accident" attack this agent
                            a2->spikeLength= 0;
                        }
                        
                        a2->spiked= true; //set a flag saying that this agent was hit this turn
                    }
                }
            }
        }
    }
}

void World::brainsTick()
{
    #pragma omp parallel for
    for (int i=0;i<agents.size();i++) {
        agents[i].tick();
    }
}

void World::addRandomBots(int num)
{
    for (int i=0;i<num;i++) {
        Agent a;
        a.id= idcounter;
        idcounter++;
        agents.push_back(a);
        
        // Add new agent to spatial grid
        spatialGrid->addAgent(&agents.back());
        
        // Track agent creation for lineage statistics
        if (STATSWINDOW) {
            STATSWINDOW->trackAgentCreation(a.lineageTag, a.gencount, current_epoch, modcounter);
        }
    }
}

void World::positionOfInterest(int type, float &xi, float &yi) {
    if(type==1){
        //the interest of type 1 is the oldest agent
        int maxage=-1;
        int maxi=-1;
        for(int i=0;i<agents.size();i++){
           if(agents[i].age>maxage) { maxage = agents[i].age; maxi=i; }
        }
        if(maxi!=-1) {
            xi = agents[maxi].pos.x;
            yi = agents[maxi].pos.y;
        }
    } else if(type==2){
        //interest of type 2 is the selected agent
        int maxi=-1;
        for(int i=0;i<agents.size();i++){
           if(agents[i].selectflag==1) {maxi=i; break; }
        }
        if(maxi!=-1) {
            xi = agents[maxi].pos.x;
            yi = agents[maxi].pos.y;
        }
    }
    
}

void World::addCarnivore()
{
    Agent a;
    a.id= idcounter;
    idcounter++;
    a.herbivore= randf(0, 0.1);
    agents.push_back(a);
    
    // Track agent creation for lineage statistics
    if (STATSWINDOW) {
        STATSWINDOW->trackAgentCreation(a.lineageTag, a.gencount, current_epoch, modcounter);
    }
}

void World::addHerbivore()
{
    Agent a;
    a.id= idcounter;
    idcounter++;
    a.herbivore= randf(0.9, 1);
    agents.push_back(a);
    
    // Track agent creation for lineage statistics
    if (STATSWINDOW) {
        STATSWINDOW->trackAgentCreation(a.lineageTag, a.gencount, current_epoch, modcounter);
    }
}

void World::addRandomAgents(int num)
{
    for (int i = 0; i < num; i++) {
        Agent a;
        a.id = idcounter;
        idcounter++;
        // Random herbivore value between 0 and 1 (0 = carnivore, 1 = herbivore)
        a.herbivore = randf(0, 1);
        agents.push_back(a);
        
        // Track agent creation for lineage statistics
        if (STATSWINDOW) {
            STATSWINDOW->trackAgentCreation(a.lineageTag, a.gencount, current_epoch, modcounter);
        }
    }
}


void World::addNewByCrossover()
{

    //find two success cases
    int i1= randi(0, agents.size());
    int i2= randi(0, agents.size());
    for (int i=0;i<agents.size();i++) {
        if (agents[i].age > agents[i1].age && randf(0,1)<0.1) {
            i1= i;
        }
        if (agents[i].age > agents[i2].age && randf(0,1)<0.1 && i!=i1) {
            i2= i;
        }
    }

    Agent* a1= &agents[i1];
    Agent* a2= &agents[i2];


    //cross brains
    Agent anew = a1->crossover(*a2);


    //maybe do mutation here? I dont know. So far its only crossover
    anew.id= idcounter;
    idcounter++;
    agents.push_back(anew);
    
    // Track agent creation for lineage statistics
    if (STATSWINDOW) {
        STATSWINDOW->trackAgentCreation(anew.lineageTag, anew.gencount, current_epoch, modcounter);
    }
}

void World::reproduce(int ai, float MR, float MR2)
{
    if (randf(0,1)<0.04) MR= MR*randf(1, 10);
    if (randf(0,1)<0.04) MR2= MR2*randf(1, 10);

    agents[ai].initEvent(30,0,0.8,0); //green event means agent reproduced.
    for (int i=0;i<conf::BABIES();i++) {

        Agent a2 = agents[ai].reproduce(MR,MR2);
        a2.id= idcounter;
        idcounter++;
        agents.push_back(a2);
        
        // Track agent creation for lineage statistics
        if (STATSWINDOW) {
            STATSWINDOW->trackAgentCreation(a2.lineageTag, a2.gencount, current_epoch, modcounter);
        }

        //TODO fix recording
        //record this
        //FILE* fp = fopen("log.txt", "a");
        //fprintf(fp, "%i %i %i\n", 1, this->id, a2.id); //1 marks the event: child is born
        //fclose(fp);
    }
}

void World::writeReport()
{
    //TODO fix reporting
    //save all kinds of nice data stuff
//     int numherb=0;
//     int numcarn=0;
//     int topcarn=0;
//     int topherb=0;
//     for(int i=0;i<agents.size();i++){
//         if(agents[i].herbivore>0.5) numherb++;
//         else numcarn++;
// 
//         if(agents[i].herbivore>0.5 && agents[i].gencount>topherb) topherb= agents[i].gencount;
//         if(agents[i].herbivore<0.5 && agents[i].gencount>topcarn) topcarn= agents[i].gencount;
//     }
// 
//     FILE* fp = fopen("report.txt", "a");
//     fprintf(fp, "%i %i %i %i\n", numherb, numcarn, topcarn, topherb);
//     fclose(fp);
}


void World::reset()
{
    agents.clear();
    addRandomBots(conf::NUMBOTS());
}

void World::setClosed(bool close)
{
    CLOSED = close;
}

bool World::isClosed() const
{
    return CLOSED;
}


void World::processMouse(int button, int state, int x, int y)
{
     if (state==0) {        
         float mind=1e10;
         float mini=-1;
         float d;

         for (int i=0;i<agents.size();i++) {
             d= pow(x-agents[i].pos.x,2)+pow(y-agents[i].pos.y,2);
                 if (d<mind) {
                     mind=d;
                     mini=i;
                 }
             }
         //toggle selection of this agent
         for (int i=0;i<agents.size();i++) agents[i].selectflag=false;
         agents[mini].selectflag= true;
         agents[mini].printSelf();
     }
}
     
void World::draw(GLView* view, bool drawfood)
{
    //draw food
    if(drawfood) {
        for(int i=0;i<FW;i++) {
            for(int j=0;j<FH;j++) {
                float f= 0.5*food[i][j]/conf::FOODMAX();
                view->drawFood(i,j,f);
            }
        }
    }
    
    //draw all agents
    vector<Agent>::const_iterator it;
    for ( it = agents.begin(); it != agents.end(); ++it) {
        view->drawAgent(*it);
    }
    
    view->drawMisc();
}

std::pair< int,int > World::numHerbCarnivores() const
{
    int numherb=0;
    int numcarn=0;
    for (int i=0;i<agents.size();i++) {
        if (agents[i].herbivore>0.5) numherb++;
        else numcarn++;
    }
    
    return std::pair<int,int>(numherb,numcarn);
}

int World::numAgents() const
{
    return agents.size();
}

int World::epoch() const
{
    return current_epoch;
}

std::string World::getSaveDirectory() const
{
    return "scriptbots_save_files/";
}

bool World::saveToFile(const std::string& filename)
{
    std::string fullPath = getSaveDirectory() + filename;
    std::ofstream file(fullPath, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Could not open file %s for writing\n", fullPath.c_str());
        return false;
    }
    
    // Write file header
    const char* header = "SCRIPTBOTS_SAVE";
    file.write(header, 15);
    
    // Write version number (for backward compatibility)
    int version = 3; // Version 3 includes lineage tracking data
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write configuration values to ensure save/load compatibility
    // World dimensions
    file.write(reinterpret_cast<const char*>(&conf_cache::WIDTH), sizeof(conf_cache::WIDTH));
    file.write(reinterpret_cast<const char*>(&conf_cache::HEIGHT), sizeof(conf_cache::HEIGHT));
    file.write(reinterpret_cast<const char*>(&conf_cache::WWIDTH), sizeof(conf_cache::WWIDTH));
    file.write(reinterpret_cast<const char*>(&conf_cache::WHEIGHT), sizeof(conf_cache::WHEIGHT));
    file.write(reinterpret_cast<const char*>(&conf_cache::CZ), sizeof(conf_cache::CZ));
    
    // Agent settings
    file.write(reinterpret_cast<const char*>(&conf_cache::NUMBOTS), sizeof(conf_cache::NUMBOTS));
    file.write(reinterpret_cast<const char*>(&conf_cache::BOTRADIUS), sizeof(conf_cache::BOTRADIUS));
    file.write(reinterpret_cast<const char*>(&conf_cache::BOTSPEED), sizeof(conf_cache::BOTSPEED));
    file.write(reinterpret_cast<const char*>(&conf_cache::SPIKESPEED), sizeof(conf_cache::SPIKESPEED));
    file.write(reinterpret_cast<const char*>(&conf_cache::SPIKEMULT), sizeof(conf_cache::SPIKEMULT));
    file.write(reinterpret_cast<const char*>(&conf_cache::BABIES), sizeof(conf_cache::BABIES));
    file.write(reinterpret_cast<const char*>(&conf_cache::BOOSTSIZEMULT), sizeof(conf_cache::BOOSTSIZEMULT));
    file.write(reinterpret_cast<const char*>(&conf_cache::REPRATEH), sizeof(conf_cache::REPRATEH));
    file.write(reinterpret_cast<const char*>(&conf_cache::REPRATEC), sizeof(conf_cache::REPRATEC));
    
    // Vision and perception
    file.write(reinterpret_cast<const char*>(&conf_cache::DIST), sizeof(conf_cache::DIST));
    file.write(reinterpret_cast<const char*>(&conf_cache::METAMUTRATE1), sizeof(conf_cache::METAMUTRATE1));
    file.write(reinterpret_cast<const char*>(&conf_cache::METAMUTRATE2), sizeof(conf_cache::METAMUTRATE2));
    
    // Food system
    file.write(reinterpret_cast<const char*>(&conf_cache::FOODINTAKE), sizeof(conf_cache::FOODINTAKE));
    file.write(reinterpret_cast<const char*>(&conf_cache::FOODWASTE), sizeof(conf_cache::FOODWASTE));
    file.write(reinterpret_cast<const char*>(&conf_cache::FOODMAX), sizeof(conf_cache::FOODMAX));
    file.write(reinterpret_cast<const char*>(&conf_cache::FOODADDFREQ), sizeof(conf_cache::FOODADDFREQ));
    file.write(reinterpret_cast<const char*>(&conf_cache::FOODTRANSFER), sizeof(conf_cache::FOODTRANSFER));
    file.write(reinterpret_cast<const char*>(&conf_cache::FOOD_SHARING_DISTANCE), sizeof(conf_cache::FOOD_SHARING_DISTANCE));
    file.write(reinterpret_cast<const char*>(&conf_cache::FOOD_DISTRIBUTION_RADIUS), sizeof(conf_cache::FOOD_DISTRIBUTION_RADIUS));
    file.write(reinterpret_cast<const char*>(&conf_cache::REPMULT), sizeof(conf_cache::REPMULT));
    file.write(reinterpret_cast<const char*>(&conf_cache::PROP_INIT_FOOD_FILLED), sizeof(conf_cache::PROP_INIT_FOOD_FILLED));
    
    // Simulation settings
    file.write(reinterpret_cast<const char*>(&conf_cache::AUTOSAVE_FREQUENCY), sizeof(conf_cache::AUTOSAVE_FREQUENCY));
    file.write(reinterpret_cast<const char*>(&conf_cache::RANDOM_SPAWN_EPOCH_INTERVAL), sizeof(conf_cache::RANDOM_SPAWN_EPOCH_INTERVAL));
    file.write(reinterpret_cast<const char*>(&conf_cache::RANDOM_SPAWN_COUNT), sizeof(conf_cache::RANDOM_SPAWN_COUNT));
    file.write(reinterpret_cast<const char*>(&conf_cache::INITIAL_CLOSED_ENVIRONMENT), sizeof(conf_cache::INITIAL_CLOSED_ENVIRONMENT));
    file.write(reinterpret_cast<const char*>(&conf_cache::HERBIVORE_EXTINCTION_REPOPULATION_COUNT), sizeof(conf_cache::HERBIVORE_EXTINCTION_REPOPULATION_COUNT));
    file.write(reinterpret_cast<const char*>(&conf_cache::CARNIVORE_EXTINCTION_REPOPULATION_COUNT), sizeof(conf_cache::CARNIVORE_EXTINCTION_REPOPULATION_COUNT));
    
    // Neural network settings
    file.write(reinterpret_cast<const char*>(&conf_cache::INPUTSIZE), sizeof(conf_cache::INPUTSIZE));
    file.write(reinterpret_cast<const char*>(&conf_cache::OUTPUTSIZE), sizeof(conf_cache::OUTPUTSIZE));
    file.write(reinterpret_cast<const char*>(&conf_cache::NUMEYES), sizeof(conf_cache::NUMEYES));
    file.write(reinterpret_cast<const char*>(&conf_cache::BRAINSIZE), sizeof(conf_cache::BRAINSIZE));
    file.write(reinterpret_cast<const char*>(&conf_cache::CONNS), sizeof(conf_cache::CONNS));
    
    // Write world state
    file.write(reinterpret_cast<const char*>(&modcounter), sizeof(modcounter));
    file.write(reinterpret_cast<const char*>(&current_epoch), sizeof(current_epoch));
    file.write(reinterpret_cast<const char*>(&idcounter), sizeof(idcounter));
    file.write(reinterpret_cast<const char*>(&CLOSED), sizeof(CLOSED));
    
    // Write food grid
    file.write(reinterpret_cast<const char*>(&FW), sizeof(FW));
    file.write(reinterpret_cast<const char*>(&FH), sizeof(FH));
    file.write(reinterpret_cast<const char*>(&fx), sizeof(fx));
    file.write(reinterpret_cast<const char*>(&fy), sizeof(fy));
    
    // Write food data row by row
    for (int i = 0; i < FW; i++) {
        for (int j = 0; j < FH; j++) {
            file.write(reinterpret_cast<const char*>(&food[i][j]), sizeof(float));
        }
    }
    
    // Write population history
    size_t herbSize = numHerbivore.size();
    size_t carnSize = numCarnivore.size();
    file.write(reinterpret_cast<const char*>(&herbSize), sizeof(herbSize));
    file.write(reinterpret_cast<const char*>(&carnSize), sizeof(carnSize));
    file.write(reinterpret_cast<const char*>(numHerbivore.data()), herbSize * sizeof(int));
    file.write(reinterpret_cast<const char*>(numCarnivore.data()), carnSize * sizeof(int));
    
    // Write agents
    size_t numAgents = agents.size();
    file.write(reinterpret_cast<const char*>(&numAgents), sizeof(numAgents));
    for (const Agent& agent : agents) {
        agent.saveToStream(file);
    }
    
    // Save lineage tracking data
    if (STATSWINDOW) {
        STATSWINDOW->saveLineageData(file);
    }
    
    file.close();
    printf("Simulation state saved to %s\n", fullPath.c_str());
    return true;
}

bool World::loadFromFile(const std::string& filename)
{
    std::string fullPath = getSaveDirectory() + filename;
    std::ifstream file(fullPath, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Could not open file %s for reading\n", fullPath.c_str());
        return false;
    }
    
    // Read and verify file header
    char header[16];
    file.read(header, 15);
    header[15] = '\0';
    if (strcmp(header, "SCRIPTBOTS_SAVE") != 0) {
        printf("Error: Invalid save file format\n");
        file.close();
        return false;
    }
    
    // Read version number (if available)
    int version = 1; // Default to version 1 for backward compatibility
    std::streampos currentPos = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos endPos = file.tellg();
    file.seekg(currentPos);
    
    // Check if this is likely a version 2+ file by looking at the expected size
    // Version 1 files start with world state immediately after header
    // Version 2+ files have version number + configuration data after header
    std::streamoff remainingSize = endPos - currentPos;
    
    // If there's enough data for a version number and it's a reasonable value, read it
    if (remainingSize >= static_cast<std::streamoff>(sizeof(version))) {
        int testVersion;
        file.read(reinterpret_cast<char*>(&testVersion), sizeof(testVersion));
        
        // Only accept reasonable version numbers (1-10)
        if (testVersion >= 1 && testVersion <= 10) {
            version = testVersion;
        } else {
            // This was not a version number, rewind and treat as version 1
            file.seekg(currentPos);
            version = 1;
        }
    }
    
    // Handle configuration data based on version
    if (version >= 2) {
        // Version 2+ includes configuration data - restore from save file
        g_config.readFromSaveFile(file);
        printf("Loaded configuration from save file (version %d)\n", version);
        
        // Validate critical configuration values to prevent division by zero
        if (conf_cache::CZ <= 0) {
            printf("Error: Invalid CZ value (%d) in save file, using default\n", conf_cache::CZ);
            conf_cache::CZ = 10; // Default value
        }
        if (conf_cache::WIDTH <= 0) {
            printf("Error: Invalid WIDTH value (%d) in save file, using default\n", conf_cache::WIDTH);
            conf_cache::WIDTH = 800; // Default value
        }
        if (conf_cache::HEIGHT <= 0) {
            printf("Error: Invalid HEIGHT value (%d) in save file, using default\n", conf_cache::HEIGHT);
            conf_cache::HEIGHT = 600; // Default value
        }
    } else {
        // Version 1 (old format) - use current configuration
        printf("Loading old format save file (version %d) - using current configuration\n", version);
    }
    
    // Read world state
    file.read(reinterpret_cast<char*>(&modcounter), sizeof(modcounter));
    file.read(reinterpret_cast<char*>(&current_epoch), sizeof(current_epoch));
    file.read(reinterpret_cast<char*>(&idcounter), sizeof(idcounter));
    file.read(reinterpret_cast<char*>(&CLOSED), sizeof(CLOSED));
    
    // Read food grid
    file.read(reinterpret_cast<char*>(&FW), sizeof(FW));
    file.read(reinterpret_cast<char*>(&FH), sizeof(FH));
    file.read(reinterpret_cast<char*>(&fx), sizeof(fx));
    file.read(reinterpret_cast<char*>(&fy), sizeof(fy));
    
    // Validate food grid dimensions
    if (FW <= 0 || FH <= 0) {
        printf("Error: Invalid food grid dimensions (%d x %d) in save file, recalculating\n", FW, FH);
        FW = conf_cache::WIDTH / conf_cache::CZ;
        FH = conf_cache::HEIGHT / conf_cache::CZ;
    }
    
    // Resize food vector and read data row by row
    food.resize(FW, std::vector<float>(FH, 0));
    for (int i = 0; i < FW; i++) {
        for (int j = 0; j < FH; j++) {
            file.read(reinterpret_cast<char*>(&food[i][j]), sizeof(float));
        }
    }
    
    // Read population history
    size_t herbSize, carnSize;
    file.read(reinterpret_cast<char*>(&herbSize), sizeof(herbSize));
    file.read(reinterpret_cast<char*>(&carnSize), sizeof(carnSize));
    numHerbivore.resize(herbSize);
    numCarnivore.resize(carnSize);
    file.read(reinterpret_cast<char*>(numHerbivore.data()), herbSize * sizeof(int));
    file.read(reinterpret_cast<char*>(numCarnivore.data()), carnSize * sizeof(int));
    
    // Read agents
    size_t numAgents;
    file.read(reinterpret_cast<char*>(&numAgents), sizeof(numAgents));
    
    // Validate number of agents
    if (numAgents > 10000) { // Reasonable upper limit
        printf("Error: Invalid number of agents (%zu) in save file, limiting to 1000\n", numAgents);
        numAgents = 1000;
    }
    
    agents.clear();
    agents.reserve(numAgents);
    for (size_t i = 0; i < numAgents; ++i) {
        Agent agent;
        agent.loadFromStream(file);
        agents.push_back(agent);
    }
    
    // Load lineage tracking data (if available in version 3+)
    if (STATSWINDOW && version >= 3) {
        try {
            STATSWINDOW->loadLineageData(file);
        } catch (...) {
            // If lineage data loading fails, clear existing data and continue
            printf("Warning: Could not load lineage data from save file, clearing existing data\n");
            STATSWINDOW->loadLineageData(file); // This will clear the data
        }
    } else if (STATSWINDOW && version < 3) {
        // Clear existing lineage data for old save files
        STATSWINDOW->loadLineageData(file); // This will clear the data
    }
    
    file.close();
    printf("Simulation state loaded from %s\n", fullPath.c_str());
    return true;
}

float World::getTotalFood() const
{
    float total = 0.0f;
    for (int i = 0; i < FW; i++) {
        for (int j = 0; j < FH; j++) {
            total += food[i][j];
        }
    }
    return total;
}

float World::getFoodTilePercentage() const
{
    int tilesWithFood = 0;
    int totalTiles = FW * FH;
    
    for (int i = 0; i < FW; i++) {
        for (int j = 0; j < FH; j++) {
            if (food[i][j] > 0.0f) {
                tilesWithFood++;
            }
        }
    }
    
    return (totalTiles > 0) ? (100.0f * tilesWithFood / totalTiles) : 0.0f;
}

