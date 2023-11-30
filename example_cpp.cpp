#include <jpeggpu/jpeggpu.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>

#define CHECK_JPEGGPU(call)                                                                        \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            std::cerr << "jpeggpu error \"" << jpeggpu_get_status_string(stat)                     \
                      << "\" at: " __FILE__ << ":" << __LINE__ << "\n";                            \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: example_cpp <jpeg_file>\n";
        return EXIT_FAILURE;
    }

    std::ifstream file(argv[1]);
    file.seekg(0, std::ios_base::end);
    const size_t file_length = file.tellg();
    file.seekg(0);

    std::vector<uint8_t> file_data(file_length);
    static_assert(sizeof(char) == sizeof(uint8_t));
    file.read(reinterpret_cast<char*>(file_data.data()), file_length);

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(
        jpeggpu_decoder_parse_header(decoder, &img_info, file_data.data(), file_data.size()));

    jpeggpu_img img;
    CHECK_JPEGGPU(jpeggpu_decoder_decode(decoder, &img));

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));
}