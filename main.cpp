#include <jpeggpu/jpeggpu.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>

#define CHECK_JPEGGPU(call)                                                    \
  do {                                                                         \
    jpeggpu::status stat = call;                                               \
    if (stat != jpeggpu::status::success) {                                    \
      std::cerr << "jpeggpu error \"" << jpeggpu::get_status_string(stat)      \
                << "\" at: " __FILE__ << ":" << __LINE__ << "\n";              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: jpeggputool <jpeg_file>\n";
    return EXIT_FAILURE;
  }

  std::ifstream file(argv[1]);
  file.seekg(0, std::ios_base::end);
  const size_t file_length = file.tellg();
  file.seekg(0);

  std::vector<uint8_t> file_data(file_length);
  static_assert(sizeof(char) == 1);
  file.read(reinterpret_cast<char *>(file_data.data()), file_length);

  jpeggpu::decoder_t decoder;
  CHECK_JPEGGPU(jpeggpu::decoder_startup(&decoder));

  CHECK_JPEGGPU(jpeggpu::parse_header(decoder, file_data));

  CHECK_JPEGGPU(jpeggpu::decoder_cleanup(decoder));
}