#include <Arduino.h>
#include "HCSR04.h"
#include "Motor.h"
#include "NeuralNetwork.h"
#include "pesos_red_neuronal.h"


const float learningRate = 0.01f; // Controla qué tan rápido se ajustan los pesos

// Valores esperados (Ejemplo: motores encendidos o apagados según una regla)
std::vector<std::vector<float>> targetOptions = {
    {1,0,1,0},  // Adelante
    {0,1,0,1},  // Atrás
    {1,0,0,1},  // Izquierda
    {0,1,1,0}   // Derecha
};
int findClosestMatch(const std::vector<float>& input, const std::vector<std::vector<float>>& targetOptions);
// -----------------------------------------------------------------------------
// 1. Configura la topología de la red.
//    - 4 entradas (dist. Izq, dist. Der, dur. Izq, dur. Der).
//    - 8 neuronas en la capa oculta (puedes variar según tu modelo).
//    - 4 salidas (una para cada motor).
//    - Usamos ReLU para la capa oculta y Sigmoid para la capa de salida.
// -----------------------------------------------------------------------------
NeuralNetwork redNeuronal({4, 8, 4},
                          ActivationFunction::RELU,
                          ActivationFunction::SIGMOID);

// -----------------------------------------------------------------------------
// 2. Definición de sensores y motores
// -----------------------------------------------------------------------------
HCSR04 sensorDerecho(5, 18);
HCSR04 sensorIzquierdo(19, 21);
Motor motores[] = { Motor(26), Motor(25), Motor(33), Motor(32) };

// -----------------------------------------------------------------------------
// 3. Variables de tiempo
// -----------------------------------------------------------------------------
unsigned long tiempoAnterior = 0;
const unsigned long intervaloMedicion = 100;
unsigned long tiempoInicioDerecha = 0;
unsigned long tiempoInicioIzquierda = 0;

int findClosestMatch(const std::vector<float>& input, const std::vector<std::vector<float>>& targetOptions) {
    if (targetOptions.empty() || input.size() != 4) {
        return -1; // Error
    }

    int closestIndex = -1;
    float minDistance = 99999.0f; // En Arduino, evitar std::numeric_limits<float>::max()

    for (size_t i = 0; i < targetOptions.size(); i++) {
        float distance = 0.0f;
        for (size_t j = 0; j < 4; j++) {
            float diff = input[j] - targetOptions[i][j];
            distance += diff * diff; // Distancia euclidiana
        }

        if (distance < minDistance) {
            minDistance = distance;
            closestIndex = i;
        }
    }

    return closestIndex;
}

void setup() {
    Serial.begin(115200);
    // Encabezados para la salida serial (opcional)
    Serial.println("distancia_izquierda,distancia_derecha,duracion_izquierda,duracion_derecha,IN1,IN2,IN3,IN4");

    // -------------------------------------------------------------------------
    // Cargar los pesos y sesgos desde el archivo 'pesos_red_neuronal.h'
    // -------------------------------------------------------------------------
    std::vector<std::vector<float>> weights = {
        std::vector<float>(PESOS_CAPA_0, PESOS_CAPA_0 + sizeof(PESOS_CAPA_0) / sizeof(float)),
        std::vector<float>(PESOS_CAPA_1, PESOS_CAPA_1 + sizeof(PESOS_CAPA_1) / sizeof(float))
    };

    std::vector<std::vector<float>> biases = {
        std::vector<float>(SESGOS_CAPA_0, SESGOS_CAPA_0 + sizeof(SESGOS_CAPA_0) / sizeof(float)),
        std::vector<float>(SESGOS_CAPA_1, SESGOS_CAPA_1 + sizeof(SESGOS_CAPA_1) / sizeof(float))
    };

    // Asignar los pesos y sesgos a la red neuronal
    redNeuronal.setWeights(weights, biases);
}

void loop() {
    unsigned long tiempoActual = millis();

    if (tiempoActual - tiempoAnterior >= intervaloMedicion) {
        tiempoAnterior = tiempoActual;

        // Leer sensores
        float distanciaDerecha = sensorDerecho.medirDistancia();
        float distanciaIzquierda = sensorIzquierdo.medirDistancia();

        if (distanciaDerecha > 0) tiempoInicioDerecha = tiempoActual;
        if (distanciaIzquierda > 0) tiempoInicioIzquierda = tiempoActual;

        float duracionDerecha = (tiempoActual - tiempoInicioDerecha) / 1000.0f;
        float duracionIzquierda = (tiempoActual - tiempoInicioIzquierda) / 1000.0f;

        // 1️ Entrada a la red neuronal
        std::vector<float> input = {distanciaIzquierda, distanciaDerecha, duracionIzquierda, duracionDerecha};
        
        // 2️ Forward pass
        std::vector<float> output = redNeuronal.forward(input);

        int closestMatch = findClosestMatch(output, targetOptions);
        Serial.println(closestMatch);
        std::vector<float> targetOutputs = targetOptions[closestMatch];
        // 3️ Calcular error con MSE
        float error = redNeuronal.calculateError(output, targetOutputs, ErrorFunction::MSE);

        std::vector<std::vector<float>> grad_weights, grad_biases;
        redNeuronal.computeGradients(input, targetOutputs, grad_weights, grad_biases);

        // 5️ Ajustar pesos con descenso de gradiente
        redNeuronal.updateWeights(grad_weights, grad_biases, learningRate);

        // 6️ Imprimir información de depuración
        // Serial.print(distanciaIzquierda); Serial.print(",");
        // Serial.print(distanciaDerecha); Serial.print(",");
        // Serial.print(duracionIzquierda); Serial.print(",");
        // Serial.print(duracionDerecha);
        for (int i = 0; i < 4; i++) {
            bool motorOn = (output[i] > 0.5f);
            motores[i].setEstado(motorOn);
            Serial.print(",");
            Serial.print(motorOn ? 1 : 0);
        }
        // Serial.print(", Error: ");
        // Serial.println(error);
    }
}


