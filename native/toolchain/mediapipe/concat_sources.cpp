#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Concatenate bytes, never text, in the same sorted order as XNNPACK's generator.
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream list(argv[1]);
    if (!list) return 3;
    std::vector<std::string> paths;
    std::string path;
    while (std::getline(list, path)) {
        if (!path.empty() && path.back() == '\r') path.pop_back();
        if (!path.empty()) paths.push_back(path);
    }
    std::sort(paths.begin(), paths.end());
    std::ofstream output(argv[2], std::ios::binary);
    if (!output) return 4;
    char buffer[65536];
    for (const auto& name : paths) {
        std::ifstream input(name, std::ios::binary);
        if (!input) { std::cerr << "Cannot read " << name << '\n'; return 5; }
        while (input.read(buffer, sizeof buffer) || input.gcount()) {
            output.write(buffer, input.gcount());
        }
        if (!input.eof() || !output) return 6;
    }
    output.close();
    return output ? 0 : 7;
}
