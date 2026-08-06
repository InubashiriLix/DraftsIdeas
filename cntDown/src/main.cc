#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "toml.hpp"

struct Colors
{
    constexpr static const char* red    = "\033[31m"; // red
    constexpr static const char* yellow = "\033[33m"; // yellow
    constexpr static const char* green  = "\033[32m"; // green
    constexpr static const char* gray   = "\033[90m"; // gray
    constexpr static const char* reset  = "\033[0m";  // reset
};

std::string format_countdown(std::chrono::system_clock::duration diff) {
    bool overdue = diff < std::chrono::system_clock::duration::zero();

    if (overdue) {
        diff = -diff;
    }

    auto total_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(diff).count();

    auto days = total_seconds / 86400;
    total_seconds %= 86400;

    auto hours = total_seconds / 3600;
    total_seconds %= 3600;

    auto minutes = total_seconds / 60;
    auto seconds = total_seconds % 60;

    std::ostringstream out;

    if (overdue) {
        out << "overdue by ";
    }

    if (days > 0) {
        out << days << "d ";
    }

    out << std::setfill('0')
        << std::setw(2) << hours << ":"
        << std::setw(2) << minutes << ":"
        << std::setw(2) << seconds;

    if (!overdue) {
        out << " left";
    }

    return out.str();
}

const char* countdown_color(std::chrono::system_clock::duration diff) {
    using namespace std::chrono;

    if (diff < system_clock::duration::zero()) {
        return "\033[90m"; // gray: already passed
    }

    if (diff <= hours(24)) {
        return "\033[31m"; // red: within 1 day
    }

    if (diff <= hours(72)) {
        return "\033[33m"; // yellow: within 3 days
    }

    return "\033[32m"; // green: more than 3 days
}

std::chrono::system_clock::time_point to_time_point(const toml::date_time& dt) {
    std::tm tm {};
    tm.tm_year  = static_cast<int>(dt.date.year) - 1900;
    tm.tm_mon   = static_cast<int>(dt.date.month) - 1;
    tm.tm_mday  = static_cast<int>(dt.date.day);
    tm.tm_hour  = static_cast<int>(dt.time.hour);
    tm.tm_min   = static_cast<int>(dt.time.minute);
    tm.tm_sec   = static_cast<int>(dt.time.second);
    tm.tm_isdst = -1;

    std::time_t t = std::mktime(&tm);

    return std::chrono::system_clock::from_time_t(t) + std::chrono::nanoseconds(dt.time.nanosecond);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <timetable.toml>" << std::endl;
        return 1;
    }

    auto now = std::chrono::system_clock::now();

    const char* fileAbsPath = argv[1];

    try {
        // use the fileAbsPath to read the file
        auto config = toml::parse_file(fileAbsPath);

        auto exams = config["exam"].as_array();
        if (!exams) {
            std::cerr << "Error: 'exam' is not an array in the TOML file." << std::endl;
            return 1;
        }

        for (const auto& item : *exams) {
            const auto* exam = item.as_table();
            if (!exam) {
                std::cerr << "Error: Each item in 'exam' array should be a table." << std::endl;
                continue;
            }

            std::string course    = (*exam)["course"].value_or("");
            auto        time      = (*exam)["time"].value<toml::date_time>();
            auto        exam_time = to_time_point(*time);
            auto        left      = exam_time - now;

            const char* reset = "\033[0m";

            std::cout << countdown_color(left)
                      << course << " | "
                      << *time << " | "
                      << format_countdown(left)
                      << reset << '\n';
        }
    }
    catch (const toml::parse_error& err) {
        std::cerr << "TOML parse error:\n"
                  << err << '\n';
        return 1;
    }

    return 0;
}
