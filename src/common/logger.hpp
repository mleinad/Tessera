#ifndef logger_hpp
#define logger_hpp

#include <iostream>
#include <format>
#include <chrono>
#include <utility>
#include <ctime>

template <typename... Args>
void log(std::format_string<Args...> fmt, Args&&... args) {
    // Get the current system time
    auto now = std::chrono::system_clock::now();
    
    // Format the time as a readable string (e.g., 2026-11-11 22:00:00)
    auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", now);
    
    // Format the user's message
    std::string user_msg = std::format(fmt, std::forward<Args>(args)...);
    
    // Output with the timestamp prefix
    std::clog << std::format("[{}] {}\n", time_str, user_msg);
}

#endif /* logger_h */
