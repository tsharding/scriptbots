#include "MLPBrain.h"
using namespace std;


MLPBox::MLPBox()
{

    w.resize(conf::CONNS(),0);
    id.resize(conf::CONNS(),0);
    type.resize(conf::CONNS(),0);

    //constructor
    for (int i=0;i<conf::CONNS();i++) {
        w[i]= randf(-3,3);
        if(randf(0,1)<0.5) w[i]=0; //make brains sparse
        
        id[i]= randi(0,conf::BRAINSIZE());
        if (randf(0,1)<0.2) id[i]= randi(0,conf::INPUTSIZE()); //20% of the brain AT LEAST should connect to input.
        
        type[i] = 0;
        if(randf(0,1)<0.05) type[i] = 1; //make 5% be change sensitive synapses
    }

    kp=randf(0.9,1.1);
    gw= randf(0,5);
    bias= randf(-2,2);

    out=0;
    oldout=0;
    target=0;
}

MLPBrain::MLPBrain()
{

    //constructor
    for (int i=0;i<conf::BRAINSIZE();i++) {
        MLPBox a; //make a random box and copy it over
        boxes.push_back(a);
        
        /*
        boxes[i].out= a.out;
        boxes[i].oldout = a.oldout;
        boxes[i].target= a.target;
        boxes[i].kp= a.kp;
        boxes[i].gw= a.gw;
        boxes[i].bias= a.bias;
        for (int j=0;j<conf::CONNS();j++) {
            boxes[i].w[j]= a.w[j];
            boxes[i].id[j]= a.id[j];
            boxes[i].type[j] = a.type[j];
            if (i<conf::BRAINSIZE()/2) {
                boxes[i].id[j]= randi(0,conf::INPUTSIZE());
            }
        }
        */
    }

    //do other initializations
    init();
}

MLPBrain::MLPBrain(const MLPBrain& other)
{
    boxes = other.boxes;
}

MLPBrain& MLPBrain::operator=(const MLPBrain& other)
{
    if( this != &other )
        boxes = other.boxes;
    return *this;
}


void MLPBrain::init()
{

}

void MLPBrain::tick(vector< float >& in, vector< float >& out)
{
    //do a single tick of the brain

    //take first few boxes and set their out to in[].
    for (int i=0;i<conf::INPUTSIZE();i++) {
        boxes[i].out= in[i];
    }

    //then do a dynamics tick and set all targets
    for (int i=conf::INPUTSIZE();i<conf::BRAINSIZE();i++) {
        MLPBox* abox= &boxes[i];
        
        float acc=0;
        for (int j=0;j<conf::CONNS();j++) {
            int idx=abox->id[j];
            int type = abox->type[j];
            
            // Bounds check to prevent crash
            if (idx < 0 || idx >= (int)boxes.size()) {
                continue; // Skip invalid connections
            }
            
            float val= boxes[idx].out;
            
            if(type==1){
                val-= boxes[idx].oldout;
                val*=10;
            }
            
            acc= acc + val*abox->w[j];
        }
        acc*= abox->gw;
        acc+= abox->bias;
        
        //put through sigmoid
        acc= 1.0/(1.0+exp(-acc));
        
        abox->target= acc;
    }
    
    //back up current out for each box
    for (int i=0;i<conf::BRAINSIZE();i++){
        boxes[i].oldout = boxes[i].out;
    }

    //make all boxes go a bit toward target
    for (int i=conf::INPUTSIZE();i<conf::BRAINSIZE();i++) {
        MLPBox* abox= &boxes[i];
        abox->out =abox->out + (abox->target-abox->out)*abox->kp;
    }

    //finally set out[] to the last few boxes output
    for (int i=0;i<conf::OUTPUTSIZE();i++) {
        int output_idx = conf::BRAINSIZE()-1-i;
        
        // Bounds check for output assignment
        if (output_idx < 0 || output_idx >= (int)boxes.size()) {
            out[i] = 0.0f; // Default to 0 for invalid indices
            continue;
        }
        
        out[i]= boxes[output_idx].out;
    }
}

void MLPBrain::mutate(float MR, float MR2)
{
    for (int j=0;j<conf::BRAINSIZE();j++) {

        if (randf(0,1)<MR) {
            boxes[j].bias+= randn(0, MR2);
//             a2.mutations.push_back("bias jiggled\n");
        }

        if (randf(0,1)<MR) {
            boxes[j].kp+= randn(0, MR2);
            if (boxes[j].kp<0.01) boxes[j].kp=0.01;
            if (boxes[j].kp>1) boxes[j].kp=1;
//             a2.mutations.push_back("kp jiggled\n");
        }
        
        if (randf(0,1)<MR) {
            boxes[j].gw+= randn(0, MR2);
            if (boxes[j].gw<0) boxes[j].gw=0;
//             a2.mutations.push_back("kp jiggled\n");
        }

        if (randf(0,1)<MR) {
            int rc= randi(0, conf::CONNS());
            boxes[j].w[rc]+= randn(0, MR2);
//          a2.mutations.push_back("weight jiggled\n");
        }
        
        if (randf(0,1)<MR) {
            int rc= randi(0, conf::CONNS());
            boxes[j].type[rc] = 1 - boxes[j].type[rc]; //flip type of synapse
//          a2.mutations.push_back("weight jiggled\n");
        }

        //more unlikely changes here
        if (randf(0,1)<MR) {
            int rc= randi(0, conf::CONNS());
            int ri= randi(0,conf::BRAINSIZE());
            boxes[j].id[rc]= ri;
//             a2.mutations.push_back("connectivity changed\n");
        }
    }
}

MLPBrain MLPBrain::crossover(const MLPBrain& other)
{
    //this could be made faster by returning a pointer
    //instead of returning by value
    MLPBrain newbrain(*this);
    
    for (int i=0;i<newbrain.boxes.size(); i++) {
        if(randf(0,1)<0.5){
            newbrain.boxes[i].bias= this->boxes[i].bias;
            newbrain.boxes[i].gw= this->boxes[i].gw;
            newbrain.boxes[i].kp= this->boxes[i].kp;
            for (int j=0;j<newbrain.boxes[i].id.size();j++) {
                newbrain.boxes[i].id[j] = this->boxes[i].id[j];
                newbrain.boxes[i].w[j] = this->boxes[i].w[j];
                newbrain.boxes[i].type[j] = this->boxes[i].type[j];
            }
        
        } else {
            newbrain.boxes[i].bias= other.boxes[i].bias;
            newbrain.boxes[i].gw= other.boxes[i].gw;
            newbrain.boxes[i].kp= other.boxes[i].kp;
            for (int j=0;j<newbrain.boxes[i].id.size();j++) {
                newbrain.boxes[i].id[j] = other.boxes[i].id[j];
                newbrain.boxes[i].w[j] = other.boxes[i].w[j];
                newbrain.boxes[i].type[j] = other.boxes[i].type[j];
            }
        }
    }
    return newbrain;
}

void MLPBrain::saveToStream(std::ofstream& file) const
{
    // Save number of boxes
    size_t numBoxes = boxes.size();
    file.write(reinterpret_cast<const char*>(&numBoxes), sizeof(numBoxes));
    
    // Save each box
    for (const MLPBox& box : boxes) {
        // Save weights
        size_t wSize = box.w.size();
        file.write(reinterpret_cast<const char*>(&wSize), sizeof(wSize));
        file.write(reinterpret_cast<const char*>(box.w.data()), wSize * sizeof(float));
        
        // Save IDs
        size_t idSize = box.id.size();
        file.write(reinterpret_cast<const char*>(&idSize), sizeof(idSize));
        file.write(reinterpret_cast<const char*>(box.id.data()), idSize * sizeof(int));
        
        // Save types
        size_t typeSize = box.type.size();
        file.write(reinterpret_cast<const char*>(&typeSize), sizeof(typeSize));
        file.write(reinterpret_cast<const char*>(box.type.data()), typeSize * sizeof(int));
        
        // Save other properties
        file.write(reinterpret_cast<const char*>(&box.kp), sizeof(box.kp));
        file.write(reinterpret_cast<const char*>(&box.gw), sizeof(box.gw));
        file.write(reinterpret_cast<const char*>(&box.bias), sizeof(box.bias));
        file.write(reinterpret_cast<const char*>(&box.target), sizeof(box.target));
        file.write(reinterpret_cast<const char*>(&box.out), sizeof(box.out));
        file.write(reinterpret_cast<const char*>(&box.oldout), sizeof(box.oldout));
    }
}

void MLPBrain::loadFromStream(std::ifstream& file)
{
    // Load number of boxes
    size_t numBoxes;
    file.read(reinterpret_cast<char*>(&numBoxes), sizeof(numBoxes));
    
    // Validate number of boxes
    if (numBoxes > 1000) { // Reasonable upper limit
        printf("Warning: Invalid number of brain boxes (%zu), limiting to 100\n", numBoxes);
        numBoxes = 100;
    }
    
    boxes.resize(numBoxes);
    
    // Load each box
    for (MLPBox& box : boxes) {
        // Load weights
        size_t wSize;
        file.read(reinterpret_cast<char*>(&wSize), sizeof(wSize));
        if (wSize > 1000) { // Reasonable upper limit
            printf("Warning: Invalid weight size (%zu), limiting to 100\n", wSize);
            wSize = 100;
        }
        box.w.resize(wSize);
        file.read(reinterpret_cast<char*>(box.w.data()), wSize * sizeof(float));
        
        // Load IDs
        size_t idSize;
        file.read(reinterpret_cast<char*>(&idSize), sizeof(idSize));
        if (idSize > 1000) { // Reasonable upper limit
            printf("Warning: Invalid ID size (%zu), limiting to 100\n", idSize);
            idSize = 100;
        }
        box.id.resize(idSize);
        file.read(reinterpret_cast<char*>(box.id.data()), idSize * sizeof(int));
        
        // Load types
        size_t typeSize;
        file.read(reinterpret_cast<char*>(&typeSize), sizeof(typeSize));
        if (typeSize > 1000) { // Reasonable upper limit
            printf("Warning: Invalid type size (%zu), limiting to 100\n", typeSize);
            typeSize = 100;
        }
        box.type.resize(typeSize);
        file.read(reinterpret_cast<char*>(box.type.data()), typeSize * sizeof(int));
        
        // Load other properties
        file.read(reinterpret_cast<char*>(&box.kp), sizeof(box.kp));
        file.read(reinterpret_cast<char*>(&box.gw), sizeof(box.gw));
        file.read(reinterpret_cast<char*>(&box.bias), sizeof(box.bias));
        file.read(reinterpret_cast<char*>(&box.target), sizeof(box.target));
        file.read(reinterpret_cast<char*>(&box.out), sizeof(box.out));
        file.read(reinterpret_cast<char*>(&box.oldout), sizeof(box.oldout));
    }
}

