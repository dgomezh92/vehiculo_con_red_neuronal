#include <Arduino.h>
#include "HCSR04.h"
#include "Motor.h"
#include "NeuralNetwork.h"

// Definición de sensores y motores
HCSR04 sensorDerecho(5, 18);
HCSR04 sensorIzquierdo(19, 21);
NeuralNetwork redNeuronal;
Motor motores[] = {Motor(26), Motor(25), Motor(33), Motor(32)};

// Variables para temporización
unsigned long tiempoAnterior = 0;
const unsigned long intervaloMedicion = 100;

// Tiempos de activación inicial de sensores
unsigned long tiempoInicioDerecha = 0;
unsigned long tiempoInicioIzquierda = 0;

void setup() {
    Serial.begin(115200);
    // Encabezados para la salida serial
    Serial.println("distancia_izquierda,distancia_derecha,duracion_izquierda,duracion_derecha,IN1,IN2,IN3,IN4");
}

void loop() {
    unsigned long tiempoActual = millis();

    if (tiempoActual - tiempoAnterior >= intervaloMedicion) {
        tiempoAnterior = tiempoActual;

        // Variables de salida
        float outputs[4];
        float distanciaDerecha = sensorDerecho.medirDistancia();
        float distanciaIzquierda = sensorIzquierdo.medirDistancia();

        // Actualizar tiempo de inicio si cambia la distancia
        if (distanciaDerecha > 0) tiempoInicioDerecha = tiempoActual;
        if (distanciaIzquierda > 0) tiempoInicioIzquierda = tiempoActual;

        // Calcular duración desde el último cambio (en segundos)
        float duracionDerecha = (tiempoActual - tiempoInicioDerecha) / 1000.0;
        float duracionIzquierda = (tiempoActual - tiempoInicioIzquierda) / 1000.0;

        // Realizar la predicción de la red neuronal
        redNeuronal.predict(distanciaIzquierda, distanciaDerecha, duracionIzquierda, duracionDerecha, outputs);

        // Imprimir las distancias, duraciones y salidas
        Serial.print(distanciaIzquierda);
        Serial.print(",");
        Serial.print(distanciaDerecha);
        Serial.print(",");
        Serial.print(duracionIzquierda);
        Serial.print(",");
        Serial.print(duracionDerecha);

        for (int i = 0; i < 4; i++) {
            Serial.print(",");
            Serial.print(outputs[i] > 0.5 ? 1 : 0); // Activar como 1 o 0
            motores[i].setEstado(outputs[i] > 0.5);
        }

        Serial.println(); // Nueva línea para el siguiente conjunto de datos
    }
}
