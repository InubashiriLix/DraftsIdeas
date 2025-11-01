#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

#include "alarm_thread.hpp"

int main() {
    std::cout << "Simple Alarm Clock" << std::endl;
    std::cout << "Enter alarm time in seconds: ";

    int seconds;
    std::cin >> seconds;

    if (seconds <= 0) {
        std::cerr << "Invalid time! Must be positive." << std::endl;
        return 1;
    }

    // Print 3 empty lines for the alarm display area
    std::cout << std::endl << std::endl << std::endl;

    // Create alarm with 1000ms refresh interval
    AlarmThread alarm(1000, seconds);

    if (!alarm.start()) {
        std::cerr << "Failed to start alarm!" << std::endl;
        return 1;
    }

    // Wait for user to press Enter to stop early (optional)
    std::cout << "\nPress Enter to stop alarm early..." << std::endl;
    std::cin.ignore();
    std::cin.get();

    if (alarm.stop()) {
        std::cout << "Alarm stopped." << std::endl;
    }

    return 0;
}
