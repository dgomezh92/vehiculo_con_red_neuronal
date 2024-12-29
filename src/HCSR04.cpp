#include "HCSR04.h"

HCSR04::HCSR04(int trigPin, int echoPin) : trigPin(trigPin), echoPin(echoPin) {
    pinMode(trigPin, OUTPUT);
    pinMode(echoPin, INPUT);
}

float HCSR04::medirDistancia() {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    long duracion = pulseIn(echoPin, HIGH, 30000);
    if (duracion == 0) return -1;

    float distancia = duracion * 0.034 / 2;
    return (distancia >= 2 && distancia <= 400) ? distancia : -1;
}
