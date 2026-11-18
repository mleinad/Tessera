#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace NRRDLoader {

std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t");
    const size_t b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
    
    
    // Reads a 3D uint8 NRRD whose sizes must all equal `side`, into `dst` (side^3 bytes).
    // Only the simple case is supported: raw encoding, data in the same file.
    bool LoadNrrdU8(const char* path, size_t side, unsigned char* dst) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::fprintf(stderr, "NRRD: cannot open %s\n", path);
            return false;
        }
        
        std::string line;
        if (!std::getline(file, line) || line.rfind("NRRD000", 0) != 0) {
            std::fprintf(stderr, "NRRD: %s is not a NRRD file\n", path);
            return false;
        }
        
        std::string type, encoding;
        int  dimension = 0;
        long sizes[3]  = {0, 0, 0};
        bool detached  = false;
        
        // Header: "key: value" lines until the first empty line; raw data follows it
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) break;
            if (line[0] == '#') continue;                    // comment
            const size_t colon = line.find(": ");
            if (colon == std::string::npos) continue;        // "key:=value" pairs etc. are ignored
            
            const std::string key   = line.substr(0, colon);
            const std::string value = Trim(line.substr(colon + 2));
            if (key == "type")                              type = value;
            else if (key == "dimension")                    dimension = std::atoi(value.c_str());
            else if (key == "encoding")                     encoding = value;
            else if (key == "data file" || key == "datafile") detached = true;
            else if (key == "sizes") {
                std::istringstream ss(value);
                ss >> sizes[0] >> sizes[1] >> sizes[2];      // fastest axis first: x y z
            }
            // "endian" doesn't matter for 1-byte data; other fields (space, kinds, ...) are ignored
        }
        
        const bool is_u8 = type == "uint8" || type == "uchar" || type == "unsigned char" ||
        type == "uint8_t";
        if (!is_u8 || dimension != 3 || encoding != "raw" || detached) {
            std::fprintf(stderr, "NRRD: %s must be uint8, 3D, raw, with attached data "
                         "(got type='%s' dimension=%d encoding='%s'%s)\n",
                         path, type.c_str(), dimension, encoding.c_str(), detached ? " detached" : "");
            return false;
        }
        for (long s : sizes) {
            if (s != (long)side) {
                std::fprintf(stderr, "NRRD: %s has sizes %ld %ld %ld, expected %zu %zu %zu\n",
                             path, sizes[0], sizes[1], sizes[2], side, side, side);
                return false;
            }
        }
        
        const size_t bytes = side * side * side;
        file.read(reinterpret_cast<char*>(dst), (std::streamsize)bytes);
        if ((size_t)file.gcount() != bytes) {
            std::fprintf(stderr, "NRRD: %s is truncated (%zu of %zu bytes)\n",
                         path, (size_t)file.gcount(), bytes);
            return false;
        }
        return true;
    }
    
}  // namespace

}

