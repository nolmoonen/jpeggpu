#ifndef JPEGGPU_JPEGGPU_HPP_
#define JPEGGPU_JPEGGPU_HPP_

#include <stdint.h>
#include <vector>

namespace jpeggpu {

struct decoder;
typedef decoder *decoder_t;

enum class status { success, error_marker_eof, error_marker_no_ff, error, not_supported };

const char *get_status_string(status stat);

status decoder_startup(decoder_t *decoder);

/// \brief Caller should already allocate file data in pinned memory (`cudaMallocHost`).
status parse_header(jpeggpu::decoder_t decoder,
                    const std::vector<uint8_t> &file_data);

status decoder_cleanup(decoder_t decoder);

} // namespace jpeggpu

#endif // JPEGGPU_JPEGGPU_HPP_