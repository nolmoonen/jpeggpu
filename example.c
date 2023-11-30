#include <jpeggpu/jpeggpu.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "failed: " #cond " at: " __FILE__ ":%d\n", __LINE__);                  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CHECK_JPEGGPU(call)                                                                        \
    do {                                                                                           \
        enum jpeggpu_status stat = call;                                                           \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            fprintf(                                                                               \
                stderr,                                                                            \
                "jpeggpu error \"%s\" at: " __FILE__ ":%d\n",                                      \
                jpeggpu_get_status_string(stat),                                                   \
                __LINE__);                                                                         \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "usage: example <jpeg_file>\n");
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];

    FILE* fp = NULL;
    CHECK((fp = fopen(filename, "r")) != NULL);

    CHECK(fseek(fp, 0, SEEK_END) != -1);
    long int off = 0;
    CHECK((off = ftell(fp)) != -1);
    CHECK(fseek(fp, 0, SEEK_SET) != -1);

    uint8_t* data = NULL;
    CHECK((data = malloc(off)) != NULL);
    CHECK(fread(data, 1, off, fp) == off);
    CHECK(fclose(fp) == 0);

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    struct jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(decoder, &img_info, data, off));

    struct jpeggpu_img img;
    // FIXME allocate channels in `img`
    CHECK_JPEGGPU(jpeggpu_decoder_decode(decoder, &img));

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    free(data);
}