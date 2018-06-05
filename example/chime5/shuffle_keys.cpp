#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <assert.h>

int main() {
    // read scp file
    std::ifstream ifs("egs.scp");
    assert(ifs.is_open());
    std::string line;
    std::vector<std::string> keys;
    bool key = true;
    while (ifs) {
        ifs >> line;
        if (key) {
            keys.push_back(line);
        }
        key = !key;
    }

    // shuffle keys
    int seed = 0;
    std::mt19937 engine(seed);
    std::shuffle(keys.begin(), keys.end(), engine);
    for (auto&& k : keys) {
        std::cout << k << std::endl;
    }
}
