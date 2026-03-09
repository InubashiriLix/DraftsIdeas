#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#if __has_include(<format>)
#include <format>
#define CDPROJ_HAS_STD_FORMAT 1
#else
#define CDPROJ_HAS_STD_FORMAT 0
#endif

class Proj {
   public:
    using Path = std::filesystem::path;
    Proj(Path projPath);

    std::string toString() const {
#if CDPROJ_HAS_STD_FORMAT
        return std::format("{} {} {}", m_projPath.string(), m_note, static_cast<long long>(m_date));
#else
        std::ostringstream oss;
        oss << m_projPath.string();
        if (!m_note.empty()) {
            oss << ' ' << m_note;
        }
        oss << ' ' << static_cast<long long>(m_date);
        return oss.str();
#endif
    }

    void updateState() {}

   private:
    const Path m_projPath;
    std::time_t m_date = std::time(nullptr);
    std::string m_note = "";
};

// namespace std::filesystem
