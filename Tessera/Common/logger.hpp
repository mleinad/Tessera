#ifndef logger_hpp
#define logger_hpp

#include <iostream>
#include <format>

template <typename... Args>
void log(std::format_string<Args...> fmt, Args&&... args) {
    std::clog << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

#endif /* logger_h */
