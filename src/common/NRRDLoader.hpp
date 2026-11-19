#ifndef NRRDLoader_hpp
#define NRRDLoader_hpp

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "common/logger.hpp"

namespace loader {

// inline: defined in a header, so including it from several .cpp files must not duplicate the symbol
inline std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t");
    const size_t b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Reads a 3D uint8 NRRD whose sizes must all equal `side`, into `dst` (side^3 bytes).
// Only the simple case is supported: raw encoding, data in the same file.
inline bool LoadNrrdU8(const char* path, size_t side, unsigned char* dst) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        log("NRRD: cannot open {}", path);
        return false;
    }

    std::string line;
    if (!std::getline(file, line) || line.rfind("NRRD000", 0) != 0) {
        log("NRRD: {} is not a NRRD file", path);
        return false;
    }

    std::string type;
    std::string encoding;
    
    int         dimension = 0;
    long        sizes[3]  = {0, 0, 0};
    bool        detached  = false;

    // Header: "key: value" lines until the first empty line; raw data follows it
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            break;
        }
        if (line[0] == '#') {
            continue; // comment
        }
        const size_t colon = line.find(": ");
        if (colon == std::string::npos) {
            continue; // "key:=value" pairs etc. are ignored
        }

        const std::string key   = line.substr(0, colon);
        const std::string value = Trim(line.substr(colon + 2));
        if (key == "type") {
            type = value;
        } else if (key == "dimension") {
            dimension = std::atoi(value.c_str());
        } else if (key == "encoding") {
            encoding = value;
        } else if (key == "data file" || key == "datafile") {
            detached = true;
        } else if (key == "sizes") {
            std::istringstream ss(value);
            ss >> sizes[0] >> sizes[1] >> sizes[2]; // fastest axis first: x y z
        }
        // "endian" doesn't matter for 1-byte data; other fields (space, kinds, ...) are ignored
    }

    const bool is_u8 = type == "uint8" || type == "uchar" || type == "unsigned char" || type == "uint8_t";
    
    if (!is_u8 || dimension != 3 || encoding != "raw" || detached) {
        log("NRRD: {} must be uint8, 3D, raw, with attached data "
            "(got type='{}' dimension={} encoding='{}'{})",
            path,
            type,
            dimension,
            encoding,
            detached ? " detached" : "");
        return false;
    }
    for (long s : sizes) {
        if (s != (long)side) {
            log("NRRD: {} has sizes {} {} {}, expected {} {} {}",
                path,
                sizes[0],
                sizes[1],
                sizes[2],
                side,
                side,
                side);
            return false;
        }
    }

    const size_t bytes = side * side * side;
    file.read(reinterpret_cast<char*>(dst), (std::streamsize)bytes);
    if ((size_t)file.gcount() != bytes) {
        log("NRRD: {} is truncated ({} of {} bytes)", path, file.gcount(), bytes);
        return false;
    }

    log("NRRD: loaded {} ({} x {} x {}, {}, {}, {} bytes)",
        path,
        side,
        side,
        side,
        type,
        encoding,
        bytes);
    return true;
}

} // namespace Loader

#endif
