// Compile the halide module like so:
// make -C ../../../FImage/cpp_bindings/ FImage.a && g++-mp-4.6 -std=c++0x halide_blur.cpp -I ../../../FImage/cpp_bindings/ ../../../FImage/cpp_bindings/FImage.a && ./a.out && optapps/blur/test.cp -O3 -always-inline halide_blur.bc | llcapps/blur/test.cp -filetype=obj > halide_blur.o

// Then compile this file like so:
// g++-mp-4.6 -Wall -ffast-math -O3 -fopenmp test.cpp halide_blur.o

#include <emmintrin.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>

#define cimg_display 0
#include "CImg.h"
using namespace cimg_library;

// TODO: fold into module
extern "C" { typedef struct CUctx_st *CUcontext; }
namespace FImage { CUcontext cuda_ctx = 0; }

timeval t1, t2;
#define begin_timing gettimeofday(&t1, NULL); for (int i = 0; i < 10; i++) {
#define end_timing } gettimeofday(&t2, NULL);

typedef CImg<uint16_t> Image;

Image blur(const Image &in) {
  Image tmp(in.width(), in.height());
  Image out(in.width(), in.height());

  begin_timing;

  for (int y = 0; y < in.height(); y++) 
    for (int x = 0; x < in.width(); x++) 
      tmp(x, y) = (in(x-1, y) + in(x, y) + in(x+1, y))/3;

  for (int y = 0; y < in.height(); y++) 
    for (int x = 0; x < in.width(); x++)
      out(x, y) = (tmp(x, y-1) + tmp(x, y) + tmp(x, y+1))/3;

  end_timing;

  return out;
}


Image blur_fast(const Image &in) {
 Image out(in.width(), in.height());
 begin_timing;
 __m128i one_third = _mm_set1_epi16(21846);
 #pragma omp parallel for
 for (int yTile = 0; yTile < in.height(); yTile += 64) {
  __m128i a, b, c, sum, avg;
  __m128i tmp[(64/8)*(64+2)];
  for (int xTile = 0; xTile < in.width(); xTile += 64) {
   __m128i *tmpPtr = tmp;
   for (int y = -1; y < 64+1; y++) {
    const uint16_t *inPtr = &(in(xTile, yTile+y));
    for (int x = 0; x < 64; x += 8) {          
     a = _mm_loadu_si128((__m128i*)(inPtr-1));
     b = _mm_loadu_si128((__m128i*)(inPtr+1));
     c = _mm_load_si128((__m128i*)(inPtr));
     sum = _mm_add_epi16(_mm_add_epi16(a, b), c);
     avg = _mm_mulhi_epi16(sum, one_third);
     _mm_store_si128(tmpPtr++, avg);
     inPtr+=8;
    }
   }
   tmpPtr = tmp;
   for (int y = 0; y < 64; y++) {
    __m128i *outPtr = (__m128i *)(&(out(xTile, yTile+y)));
    for (int x = 0; x < 64; x += 8) {
     a = _mm_load_si128(tmpPtr+(2*64)/8);
     b = _mm_load_si128(tmpPtr+64/8);
     c = _mm_load_si128(tmpPtr++);
     sum = _mm_add_epi16(_mm_add_epi16(a, b), c);
     avg = _mm_mulhi_epi16(sum, one_third);
     _mm_store_si128(outPtr++, avg);
    }
   }
  } 
 }  
 end_timing;
 return out;
}

/*
Image blur_fast(const Image &in) {
  Image out(in.width(), in.height());

  __m128i one_third = _mm_set1_epi16(21846);
  #pragma omp parallel for
  for (int yTile = 0; yTile < in.height(); yTile += 64) {
    __m128i tmp[(64/8)*(64+2)];
    for (int xTile = 0; xTile < in.width(); xTile += 64) {
      __m128i *tmpPtr = tmp;
      for (int y = -1; y < 64+1; y++) {
        const uint16_t *inPtr = &(in(xTile, yTile+y));
        for (int x = 0; x < 64; x += 8) {          
          __m128i val = _mm_loadu_si128((__m128i *)(inPtr-1));
          val = _mm_add_epi16(val, _mm_load_si128((__m128i *)inPtr));
          val = _mm_add_epi16(val, _mm_loadu_si128((__m128i *)(inPtr+1)));
          val = _mm_mulhi_epi16(val, one_third);
          _mm_store_si128(tmpPtr++, val);
          inPtr += 8;
        }
      }
      tmpPtr = tmp;
      for (int y = 0; y < 64; y++) {
        __m128i *outPtr = (__m128i *)(&(out(xTile, yTile+y)));
        for (int x = 0; x < 64; x += 8) {
          __m128i val = _mm_load_si128(tmpPtr);
          val = _mm_add_epi16(val, _mm_load_si128(tmpPtr+64/8));
          val = _mm_add_epi16(val, _mm_load_si128(tmpPtr+(2*64)/8));
          val = _mm_mulhi_epi16(val, one_third);
          _mm_store_si128(outPtr++, val);
          tmpPtr++;
        }
      }
    } 
  }
  
  return out;
}
*/


Image blur_fast2(const Image &in) {
  Image out(in.width(), in.height());

  begin_timing;

  // multiplying by 21846 then taking the top 16 bits is equivalent to
  // dividing by three
  __m128i one_third = _mm_set1_epi16(21846);

  
  #pragma omp parallel for
  for (int yTile = 0; yTile < in.height(); yTile += 128) {

    int vw = in.width()/8;
    __m128i tmp[vw*4]; // four scanlines
    for (int y = -2; y < 128; y++) {
      // to produce this scanline of the output
      __m128i *outPtr = (__m128i *)(&(out(0, yTile + y)));
      // we use this scanline of the input
      const uint16_t *inPtr = &(in(0, yTile + y + 1));
      // and these scanlines of the intermediate result
      // We start y at negative 2 to fill the tmp buffer
      __m128i *tmpPtr0 = tmp + ((y+4) & 3) * vw;
      __m128i *tmpPtr1 = tmp + ((y+3) & 3) * vw;
      __m128i *tmpPtr2 = tmp + ((y+2) & 3) * vw;
      for (int x = 0; x < vw; x++) {
        // blur horizontally to produce next scanline of tmp
        __m128i val = _mm_loadu_si128((__m128i *)(inPtr-1));
        val = _mm_add_epi16(val, _mm_load_si128((__m128i *)inPtr));
        val = _mm_add_epi16(val, _mm_loadu_si128((__m128i *)(inPtr+1)));
        val = _mm_mulhi_epi16(val, one_third);
        _mm_store_si128(tmpPtr0++, val);
        
        // blur vertically using previous scanlines of tmp to produce output
        if (y >= 0) {
          val = _mm_add_epi16(val, _mm_load_si128(tmpPtr1++));
          val = _mm_add_epi16(val, _mm_load_si128(tmpPtr2++));
          val = _mm_mulhi_epi16(val, one_third);
          _mm_store_si128(outPtr++, val);
        }

        inPtr += 8;
      }
    }
  }

  end_timing;
  
  return out;
}

extern "C" {
#include "halide_blur.h"
}

// Convert a CIMG image to a buffer_t for halide
buffer_t halideBufferOfImage(Image &im) {
  buffer_t buf = {(uint8_t *)im.data(), 0, false, false, {im.width(), im.height(), 1, 1}, sizeof(int16_t)};
  return buf;
}

Image blur_halide(Image &in) {
  Image out(in.width(), in.height());

  buffer_t inbuf = halideBufferOfImage(in);
  buffer_t outbuf = halideBufferOfImage(out);

  // Call it once to initialize the halide runtime stuff
  halide_blur(&inbuf, &outbuf);

  begin_timing;
    
  // Compute the same region of the output as blur_fast (i.e., we're
  // still being sloppy with boundary conditions)
  halide_blur(&inbuf, &outbuf);

  end_timing;

  return out;
}


int main(int argc, char **argv) {
  Image input(6400, 4864);

  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = rand() & 0xfff;
    }
  }


  Image blurry = blur(input);
  float slow_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;

  Image speedy = blur_fast(input);
  float fast_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;

  Image speedy2 = blur_fast2(input);
  float fast_time2 = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    
  Image halide = blur_halide(input);
  float halide_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;

  // fast_time2 is always slower than fast_time, so skip printing it
  printf("times: naive = %f s; hand tuned = %f s; halide %f s\n", 
         slow_time, fast_time, halide_time);

  for (int y = 64; y < input.height() - 64; y++) {
    for (int x = 64; x < input.width() - 64; x++) {
      if (blurry(x, y) != speedy(x, y) || blurry(x, y) != halide(x, y) || blurry(x, y) != speedy2(x, y))
        printf("difference at (%d,%d): %d %d %d %d\n", x, y, blurry(x, y), speedy(x, y), speedy2(x, y), halide(x, y));
    }
  }
  return 0;
}
