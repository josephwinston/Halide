#include <Halide.h>
using namespace Halide;

int main(int argc, char **argv) {

#if USE_VERSION_IN_PAPER
   printf("Executing using scheduler found in paper\n");
#else
   printf("Executing using a different scheduler\n");
#endif /* USE_VERSION_IN_PAPER */

  UniformImage input(UInt(16), 2);
  Func blur_x("blur_x"), blur_y("blur_y");
  Var x("x"), y("y"), xo("blockidx"), yo("blockidy"), xi("threadidx"), yi("threadidy");

  // The algorithm
  blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y))/3;
  blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1))/3;
  
  // How to schedule it
#if USE_VERSION_IN_PAPER
  blur_y.tile(x, y, xi, yi, 256, 32);
  blur_y.vectorize(xi, 8);
  blur_y.parallel(y);
  blur_x.chunk(x);
  blur_x.vectorize(x, 8);
#else /* USE_VERSION_IN_PAPER is false -> Using different schedule */
  blur_y.split(x, xo, xi, 32).split(y, yo, yi, 16).transpose(xo, yi);
  //blur_y.tile()
  blur_y.parallel(yo).parallel(yi).parallel(xo).parallel(xi);
  
#if 0
  // chunking
  blur_x.chunk(xo);
  blur_x.split(x, xi, x, 1).split(y, yi, y, 1);
  blur_x.parallel(yo).parallel(yi).parallel(xo).parallel(xi);
#else /* 0 */
  // inline
#endif /* 0 */
#endif /* USE_VERSION_IN_PAPER */

  blur_y.compileToFile("halide_blur");
  return 0;
}
