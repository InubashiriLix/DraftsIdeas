#include "pid.hpp"

#include <algorithm>

PID_Controller::PID_Controller(const PID_param &params)
    : param__(params), target__(0.0), last_error__(0.00), integral__(0.00) {
    last_time__ = std::chrono::steady_clock::now();
}

void PID_Controller::setTarget(double target) { target__ = target; }

void PID_Controller::reset() {
    target__ = 0.00;
    integral__ = 0.00;
    last_time__ = std::chrono::steady_clock::now();
}

double PID_Controller::update(double measurement, double dt) {
    auto now = std::chrono::steady_clock::now();
    if (dt < 0.0) {
        std::chrono::duration<double> elapsed = now - last_time__;
        dt = elapsed.count();
    }
    last_time__ = now;

    double error = target__ - measurement;

    // Deadzone
    if (std::abs(error) < param__.deadzone) {
        error = 0.0;
    }

    // Integral
    integral__ += error * dt;
    integral__ = std::clamp(integral__, param__.i_max.first, param__.i_max.second);

    double derivative = dt > 0.00 ? (error - last_error__) / dt : 0.00;
    last_error__ = error;

    // Derivative
    double output = (param__.Kp * error) + (param__.Ki * integral__) + (param__.Kd * derivative);
    output = std::clamp(output, param__.o_max.first, param__.o_max.second);

    return output;
}
