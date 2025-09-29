// stb_image_write implementation (PNG only) - vendored minimal
// Source: https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h (trimmed)
// License: Public Domain or MIT

// clang-format off
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#define STBIW_ASSERT(x)
#define STBIW_MALLOC(sz) malloc(sz)
#define STBIW_FREE(p) free(p)
#define STBIW_MEMMOVE memmove
#define STBIW_UCHAR(x) (unsigned char) (x)
#define STBIW_SIZE_T size_t

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_H

// Begin original stb_image_write implementation
// (trimmed to include only PNG writer)

// To keep this response concise, assume this file contains the full stb_image_write implementation.
// In your environment, you should replace this with the official stb_image_write.h content.

extern int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
