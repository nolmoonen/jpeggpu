// Copyright (c) 2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string_view>
#include <vector>

enum view_type {
    VIEW_TYPE_NOT_SCAN   = 0x01,
    VIEW_TYPE_DC_INITIAL = 0x02,
    VIEW_TYPE_DC_REFINE  = 0x04,
    VIEW_TYPE_AC_INITIAL = 0x08,
    VIEW_TYPE_AC_REFINE  = 0x10
};

struct stream_view {
    stream_view(std::basic_string_view<uint8_t> view, view_type type) : view(view), type(type) {}
    std::basic_string_view<uint8_t> view;
    view_type type;
};

bool has_remaining(const uint8_t* data, const uint8_t* data_end, size_t n)
{
    assert(data <= data_end);
    return static_cast<size_t>(data_end - data) >= n;
}

uint8_t read_u8(const uint8_t*& data) { return *(data++); }
uint16_t read_u16(const uint8_t*& data)
{
    const uint16_t high = read_u8(data);
    return (high << 8) | read_u8(data);
}

int skip_segment(const uint8_t*& data, const uint8_t* data_end)
{
    if (!has_remaining(data, data_end, 2)) {
        std::cerr << "incomplete bitstream\n";
        return EXIT_FAILURE;
    }

    const uint16_t length = read_u16(data);
    if (length < 2 || !has_remaining(data, data_end, length - 2)) {
        std::cerr << "incomplete bitstream\n";
        return EXIT_FAILURE;
    }

    data += length - 2;

    return EXIT_SUCCESS;
}

int parse(const std::vector<uint8_t>& file_data, std::vector<stream_view>& views)
{
    views.clear();
    const uint8_t* data           = file_data.data();
    const uint8_t* const data_end = data + file_data.size();

    const uint8_t marker_soi = 0xd8;
    if (!has_remaining(data, data_end, 2) || data[0] != 0xff || data[1] != marker_soi) {
        std::cerr << "not a JPEG file\n";
        return EXIT_FAILURE;
    }
    views.emplace_back(std::basic_string_view<uint8_t>(data, 2), view_type::VIEW_TYPE_NOT_SCAN);
    data += 2;

    while (has_remaining(data, data_end, 2)) {
        const uint8_t* const segment_begin = data;

        if (data[0] != 0xff) {
            std::cerr << "expected 0xff\n";
            return EXIT_FAILURE;
        }
        const uint8_t marker = data[1];
        data += 2;

        const uint8_t marker_eoi = 0xd9;
        if (marker == marker_eoi) {
            // include garbage after eoi
            views.emplace_back(
                std::basic_string_view<uint8_t>(segment_begin, data_end - segment_begin),
                view_type::VIEW_TYPE_NOT_SCAN);
            return EXIT_SUCCESS;
        }

        const uint8_t marker_sof0  = 0xc0;
        const uint8_t marker_sof1  = 0xc1;
        const uint8_t marker_sof3  = 0xc3;
        const uint8_t marker_sof5  = 0xc5;
        const uint8_t marker_sof11 = 0xcb;
        const uint8_t marker_sof13 = 0xcd;
        const uint8_t marker_sof15 = 0xcf;
        if (marker == marker_sof0 || marker == marker_sof1 || marker == marker_sof3 ||
            (marker_sof5 <= marker && marker <= marker_sof11) ||
            (marker_sof13 <= marker && marker <= marker_sof15)) {
            std::cerr << "not progressive\n";
            return EXIT_FAILURE;
        }

        const uint8_t marker_dht        = 0xc4;
        const uint8_t marker_dac        = 0xcc;
        const uint8_t marker_dqt        = 0xdb;
        const uint8_t marker_app15      = 0xef;
        const uint8_t marker_com        = 0xfe;
        const bool is_skippable_non_sof = marker == marker_dht || marker == marker_dac ||
                                          (marker_dqt <= marker && marker <= marker_app15) ||
                                          marker == marker_com;
        const uint8_t marker_sof2 = 0xc2;
        if (is_skippable_non_sof || marker == marker_sof2) {
            if (skip_segment(data, data_end) != EXIT_SUCCESS) {
                return EXIT_FAILURE;
            }
            views.emplace_back(
                std::basic_string_view<uint8_t>(segment_begin, data - segment_begin),
                view_type::VIEW_TYPE_NOT_SCAN);
            continue;
        }

        const uint8_t marker_sos = 0xda;
        if (marker == marker_sos) {
            if (!has_remaining(data, data_end, 2)) {
                std::cerr << "incomplete bitstream\n";
                return EXIT_FAILURE;
            }

            const uint16_t length = read_u16(data);
            if (length < 3 || !has_remaining(data, data_end, length - 2)) {
                std::cerr << "incomplete bitstream\n";
                return EXIT_FAILURE;
            }

            const uint8_t num_components = read_u8(data);
            if (length - 3 != 2 * num_components + 3) {
                std::cerr << "invalid JPEG\n";
                return EXIT_FAILURE;
            }

            for (int c = 0; c < num_components; ++c) {
                [[maybe_unused]] const uint8_t cs   = read_u8(data);
                [[maybe_unused]] const uint8_t tdta = read_u8(data);
            }

            const uint8_t ss                  = read_u8(data);
            const uint8_t se                  = read_u8(data);
            const uint8_t aa                  = read_u8(data);
            const uint8_t ah                  = aa >> 4;
            [[maybe_unused]] const uint8_t al = aa & 0xf;

            const bool is_dc   = ss == 0 && se == 0;
            const bool is_init = ah == 0;
            const view_type type =
                is_dc
                    ? (is_init ? view_type::VIEW_TYPE_DC_INITIAL : view_type::VIEW_TYPE_DC_REFINE)
                    : (is_init ? view_type::VIEW_TYPE_AC_INITIAL : view_type::VIEW_TYPE_AC_REFINE);

            do {
                const uint8_t* ret = reinterpret_cast<const uint8_t*>(
                    std::memchr(reinterpret_cast<const void*>(data), 0xff, data_end - data));
                if (ret == nullptr) {
                    std::cerr << "invalid JPEG\n";
                    return EXIT_FAILURE;
                }

                data                 = ret + 1; // skip to after 0xff
                const uint8_t marker = read_u8(data);

                // `data` now points to after marker
                if (marker == 0) continue;

                const uint8_t marker_rst0 = 0xd0;
                const uint8_t marker_rst7 = 0xd7;
                const bool is_rst         = marker_rst0 <= marker && marker <= marker_rst7;
                if (is_rst) {
                    continue;
                } else {
                    // Not a restart marker, so end of scan. Rewind 0xff and marker byte
                    data -= 2;
                }
                break;
            } while (data < data_end);
            views.emplace_back(
                std::basic_string_view<uint8_t>(segment_begin, data - segment_begin), type);
            continue;
        }

        std::cerr << "unrecognized marker\n";
        return EXIT_FAILURE;
    }

    std::cerr << "incomplete bitstream\n";
    return EXIT_FAILURE;
}

int write_file(
    const std::filesystem::path& file_path,
    const std::string& postfix,
    const std::vector<stream_view>& views,
    int view_type_flag)
{
    const std::filesystem::path path =
        file_path.parent_path() / (file_path.stem() += postfix + ".jpg");
    std::ofstream file_stream(path);
    if (!file_stream) {
        std::cerr << "failed to open \"" << path << "\"\n";
        return EXIT_FAILURE;
    }

    for (const stream_view& view : views) {
        if (view.type & view_type_flag) {
            file_stream.write(reinterpret_cast<const char*>(view.view.data()), view.view.size());
        }
    }
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "usage: tool_split_progressive <file_path>\n";
        return EXIT_FAILURE;
    }

    const std::filesystem::path file_path(argv[1]);
    std::ifstream file_stream(file_path);
    if (!file_stream) {
        std::cerr << "failed to open \"" << file_path << "\"\n";
        return EXIT_FAILURE;
    }
    file_stream.seekg(0, std::ios_base::end);
    const std::streampos file_size = file_stream.tellg();
    file_stream.seekg(0);
    std::vector<uint8_t> file_data(file_size);
    file_stream.read(reinterpret_cast<char*>(file_data.data()), file_size);
    file_stream.close();

    std::vector<stream_view> views;
    if (parse(file_data, views) != EXIT_SUCCESS) return EXIT_FAILURE;

    // sanity check that all views together cover all bytes
    size_t offset = 0;
    for (const stream_view& view : views) {
        assert(
            view.view.data() >= file_data.data() + offset &&
            offset + view.view.size() <= file_data.size());
        offset += view.view.size();
    }
    assert(offset == file_data.size());

    const int flag_dc_init = VIEW_TYPE_NOT_SCAN | VIEW_TYPE_DC_INITIAL;
    if (write_file(file_path, "_dc_init", views, flag_dc_init) != EXIT_SUCCESS) return EXIT_FAILURE;
    const int flag_dc_full = VIEW_TYPE_NOT_SCAN | VIEW_TYPE_DC_INITIAL | VIEW_TYPE_DC_REFINE;
    if (write_file(file_path, "_dc_full", views, flag_dc_full) != EXIT_SUCCESS) return EXIT_FAILURE;
    const int flag_ac_init = VIEW_TYPE_NOT_SCAN | VIEW_TYPE_AC_INITIAL;
    if (write_file(file_path, "_ac_init", views, flag_ac_init) != EXIT_SUCCESS) return EXIT_FAILURE;
    const int flag_ac_full = VIEW_TYPE_NOT_SCAN | VIEW_TYPE_AC_INITIAL | VIEW_TYPE_AC_REFINE;
    if (write_file(file_path, "_ac_full", views, flag_ac_full) != EXIT_SUCCESS) return EXIT_FAILURE;
}
