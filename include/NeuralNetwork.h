#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>
#include "../lib/pesos_red_neuronal/pesos_red_neuronal.h"

class NeuralNetwork {
public:
    void predict(float input1, float input2, float duracion1, float duracion2, float outputs[4]);

private:
    float relu(float x);
    float sigmoid(float x);
};

#endif // NEURAL_NETWORK_H
