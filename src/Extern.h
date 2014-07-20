#ifndef HALIDE_EXTERN_H
#define HALIDE_EXTERN_H

#include "Debug.h"

/** \file
 *
 * Convenience macros that lift functions that take C types into
 * functions that take and return exprs, and call the original
 * function at runtime under the hood. See test/c_function.cpp for
 * example usage.
 */

#define _halide_check_arg_type(t, name, e, n)                     \
    _halide_user_assert(e.type() == t) << "Type mismatch for argument " << n << " to extern function " << #name << ". Type expected is " << t << " but the argument " << e << " has type " << e.type() << ".\n";

#define HalideExtern_1(rt, name, t1)                                    \
    Halide::Expr name(Halide::Expr a1) {                                \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, vec(a1), Halide::Internal::Call::Extern); \
    }

#define HalideExtern_2(rt, name, t1, t2)                                \
    Halide::Expr name(Halide::Expr a1, Halide::Expr a2) {               \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);                        \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);                        \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, vec(a1, a2), Halide::Internal::Call::Extern); \
    }

#define HalideExtern_3(rt, name, t1, t2, t3)                            \
    Halide::Expr name(Halide::Expr a1, Halide::Expr a2, Halide::Expr a3) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);                                \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);                                \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);                                \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, vec(a1, a2, a3), Halide::Internal::Call::Extern); \
    }

#define HalideExtern_4(rt, name, t1, t2, t3, t4)                        \
    Halide::Expr name(Halide::Expr a1, Halide::Expr a2, Halide::Expr a3, Halide::Expr a4) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);                                \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);                                \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);                                \
        _halide_check_arg_type(Halide::type_of<t4>(), name, a4, 4);                                \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, vec(a1, a2, a3, a4), Halide::Internal::Call::Extern); \
  }

#define HalideExtern_5(rt, name, t1, t2, t3, t4, t5)                       \
    Halide::Expr name(Halide::Expr a1, Halide::Expr a2, Halide::Expr a3, Halide::Expr a4, Halide::Expr a5) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);                                \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);                                \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);                                \
        _halide_check_arg_type(Halide::type_of<t4>(), name, a4, 4);                                \
        _halide_check_arg_type(Halide::type_of<t5>(), name, a5, 5);                                \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, vec(a1, a2, a3, a4, a5), Halide::Internal::Call::Extern); \
  }

#endif
