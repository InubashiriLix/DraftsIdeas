#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

class AlarmThread {
   public:
    AlarmThread(int interval_ms, int set_time_s)
        : _interval_ms(std::move(interval_ms)), _set_time_s(set_time_s){};
    ~AlarmThread() = default;
    [[nodiscard("the return value of start must be captured")]] bool start() {
        if (std::atomic_exchange(&_running, true)) {
            return false;
        }

        _th = std::thread(&AlarmThread::worker, this);
        return true;
    }
    void worker() {
        auto start_time = std::chrono::steady_clock::now();

        while (_running.load(std::memory_order_relaxed)) {
            auto loop_start = std::chrono::steady_clock::now();

            // Calculate elapsed time since alarm started
            auto elapsed_total =
                std::chrono::duration_cast<std::chrono::seconds>(loop_start - start_time);
            int remaining_seconds = _set_time_s - elapsed_total.count();

            // Clear previous lines (clear 3 lines)
            std::cout << "\033[3A\033[J";

            if (remaining_seconds > 0) {
                // Show remaining time
                int hours = remaining_seconds / 3600;
                int minutes = (remaining_seconds % 3600) / 60;
                int seconds = remaining_seconds % 60;

                int hours_set = _set_time_s / 3600;
                int minutes_set = (_set_time_s % 3600) / 60;
                int seconds_set = _set_time_s % 60;

                std::cout << "Set time: " << hours_set << "h " << minutes_set << "m " << seconds_set
                          << "s" << std::endl;
                std::cout << "Time left: " << hours << "h " << minutes << "m " << seconds << "s"
                          << std::endl;
                std::cout << "Status: Running..." << std::endl;
            } else {
                // Time's up! Show alarm with highlight
                int hours_set = _set_time_s / 3600;
                int minutes_set = (_set_time_s % 3600) / 60;
                int seconds_set = _set_time_s % 60;

                std::cout << "Set time: " << hours_set << "h " << minutes_set << "m " << seconds_set
                          << "s" << std::endl;
                std::cout << "\033[1;31m\033[5m⏰ ALARM! TIME'S UP! ⏰\033[0m" << std::endl;
                std::cout << "\033[1;33mStatus: FINISHED!\033[0m" << std::endl;

                // Exit after alarm
                break;
            }

            auto loop_end = std::chrono::steady_clock::now();
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);
            auto sleep_time = std::chrono::milliseconds(_interval_ms) - elapsed;
            if (sleep_time > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(sleep_time);
            }
        }
    }

    [[nodiscard("the value of stop must be captured")]] bool stop() {
        if (std::atomic_exchange(&_running, false) == false) {
            return false;
        }

        // join the thread
        if (_th.joinable()) {
            _th.join();
        }

        return true;
    }

   private:
    int _interval_ms;
    int _set_time_s;

    std::atomic<bool> _running{false};

    std::thread _th;
};
