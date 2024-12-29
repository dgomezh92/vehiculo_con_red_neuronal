#include "Motor.h"

Motor::Motor(int pin) : pin(pin) {
    pinMode(pin, OUTPUT);
}

void Motor::setEstado(bool estado) {
    digitalWrite(pin, estado ? HIGH : LOW);
}
