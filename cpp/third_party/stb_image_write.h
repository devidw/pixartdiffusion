// Minimal vendored stb_image_write, PNG only
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_STATIC
#endif
#ifndef STB_IMAGE_WRITE_H
#define STB_IMAGE_WRITE_H
#include <stddef.h>
extern "C" {
int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
}
#endif

// Include implementation
#ifdef STB_IMAGE_WRITE_IMPLEMENTATION
extern "C" {
#include "stb_image_write_impl.h"
}
#endif


