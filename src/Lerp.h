#ifndef HALIDE_LERP_H
#define HALIDE_LERP_H

#include "IR.h"

/** \file
 * Defines methods for converting a lerp intrinsic into Halide IR.
 */

namespace Halide {
namespace Internal {

/** Build Halide IR that computes a lerp. Use by codegen targets that
 * don't have a native lerp. */
Expr EXPORT lower_lerp(Expr zero_val, Expr one_val, Expr weight);

}
}

#endif
