#include <Arduino.h>
#include "HCSR04.h"
#include "Motor.h"
#include "NeuralNetwork.h"

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

void setup() {
    Serial.begin(115200);
    // Encabezados para la salida serial (opcional)
    Serial.println("distancia_izquierda,distancia_derecha,duracion_izquierda,duracion_derecha,IN1,IN2,IN3,IN4");
}

void loop() {
    unsigned long tiempoActual = millis();

    // Control de muestreo cada 'intervaloMedicion' ms
    if (tiempoActual - tiempoAnterior >= intervaloMedicion) {
        tiempoAnterior = tiempoActual;

        // Leer las distancias
        float distanciaDerecha = sensorDerecho.medirDistancia();
        float distanciaIzquierda = sensorIzquierdo.medirDistancia();

        // Actualizar el tiempo de inicio si hay nueva lectura válida (> 0)
        if (distanciaDerecha > 0) {
            tiempoInicioDerecha = tiempoActual;
        }
        if (distanciaIzquierda > 0) {
            tiempoInicioIzquierda = tiempoActual;
        }

        // Calcular duración desde el último cambio (en segundos)
        float duracionDerecha = (tiempoActual - tiempoInicioDerecha) / 1000.0f;
        float duracionIzquierda = (tiempoActual - tiempoInicioIzquierda) / 1000.0f;

        // ---------------------------------------------------------------------
        // 4. Preparar las entradas para la red neuronal
        //    En este ejemplo: 4 entradas = {distIzq, distDer, durIzq, durDer}
        // ---------------------------------------------------------------------
        std::vector<float> input = {
            distanciaIzquierda,
            distanciaDerecha,
            duracionIzquierda,
            duracionDerecha
        };

        // ---------------------------------------------------------------------
        // 5. Realizar la propagación hacia adelante
        //    forward() devuelve un std::vector<float> con la salida
        // ---------------------------------------------------------------------
        std::vector<float> output = redNeuronal.forward(input);

        // ---------------------------------------------------------------------
        // 6. Utilizar las salidas para controlar los motores
        //    Ej: Si la salida > 0.5, se enciende ese motor; si no, se apaga
        // ---------------------------------------------------------------------
        Serial.print(distanciaIzquierda);
        Serial.print(",");
        Serial.print(distanciaDerecha);
        Serial.print(",");
        Serial.print(duracionIzquierda);
        Serial.print(",");
        Serial.print(duracionDerecha);

        for (int i = 0; i < 4; i++) {
            bool motorOn = (output[i] > 0.5f);
            motores[i].setEstado(motorOn);

            Serial.print(",");
            Serial.print(motorOn ? 1 : 0);
        }

        Serial.println();
    }
}
