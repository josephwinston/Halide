#ifndef HALIDE_TARGET_H
#define HALIDE_TARGET_H

/** \file
 * Defines the structure that describes a Halide target.
 */

#include <stdint.h>
#include <string>
#include "Util.h"

namespace llvm {
class Module;
class LLVMContext;
}

namespace Halide {

/** A struct representing a target machine and os to generate code for. */
struct Target {
    /** The operating system used by the target. Determines which
     * system calls to generate. */
    enum OS {OSUnknown = 0, Linux, Windows, OSX, Android, IOS, NaCl} os;

    /** The architecture used by the target. Determines the
     * instruction set to use. For the PNaCl target, the "instruction
     * set" is actually llvm bitcode. */
    enum Arch {ArchUnknown = 0, X86, ARM, PNaCl} arch;

    /** The bit-width of the target machine. Must be 0 for unknown, or 32 or 64. */
    int bits;

    /** Optional features a target can have. */
    enum Features {JIT       = 1 << 0,  ///< Generate code that will run immediately inside the calling process.
                   SSE41     = 1 << 1,  ///< Use SSE 4.1 and earlier instructions. Only relevant on x86.
                   AVX       = 1 << 2,  ///< Use AVX 1 instructions. Only relevant on x86.
                   AVX2      = 1 << 3,  ///< Use AVX 2 instructions. Only relevant on x86.
                   CUDA      = 1 << 4,  ///< Enable the CUDA runtime. Defaults to compute capability 2.0 (Fermi)
                   OpenCL    = 1 << 5,  ///< Enable the OpenCL runtime.
                   OpenGL    = 1 << 6,  ///< Enable the OpenGL runtime.
                   GPUDebug  = 1 << 7,  ///< Increase the level of checking and the verbosity of the gpu runtimes.
                   NoAsserts = 1 << 8,  ///< Disable all runtime checks, for slightly tighter code.
                   NoBoundsQuery = 1 << 9, ///< Disable the bounds querying functionality.
                   ARMv7s    = 1 << 10,  ///< Generate code for ARMv7s. Only relevant for 32-bit ARM.
                   CLDoubles = 1 << 11, ///< Enable double support on OpenCL targets
                   FMA       = 1 << 12, ///< Enable x86 FMA instruction
                   FMA4      = 1 << 13, ///< Enable x86 (AMD) FMA4 instruction set
                   F16C      = 1 << 14, ///< Enable x86 16-bit float support
                   CUDACapability30    = 1 << 15, ///< Enable CUDA compute capability 3.0 (Kepler)
                   CUDACapability32    = 1 << 16, ///< Enable CUDA compute capability 3.2 (Tegra K1)
                   CUDACapability35    = 1 << 17, ///< Enable CUDA compute capability 3.5 (Kepler)
                   CUDACapability50    = 1 << 18  ///< Enable CUDA compute capability 5.0 (Maxwell)
    };

    /** A bitmask that stores the active features. */
    uint64_t features;

    Target() : os(OSUnknown), arch(ArchUnknown), bits(0), features(0) {}
    Target(OS o, Arch a, int b, uint64_t f) : os(o), arch(a), bits(b), features(f) {}

    /** Is OpenCL or CUDA enabled in this target? I.e. is
     * Func::gpu_tile and similar going to work? We do not include
     * OpenGL, because it is not capable of gpgpu, and is not
     * scheduled via Func::gpu_tile. */
    bool has_gpu_feature() const {
        return (features & (CUDA|OpenCL)) != 0;
    }

    bool operator==(const Target &other) const {
      return os == other.os &&
          arch == other.arch &&
          bits == other.bits &&
          features == other.features;
    }

    bool operator!=(const Target &other) const {
      return !(*this == other);
    }

    /** Convert the Target into a string form that can be reconstituted
     * by merge_string(), which will always be of the form
     *
     *   arch-bits-os-feature1-feature2...featureN.
     *
     * Note that is guaranteed that t2.from_string(t1.to_string()) == t1
     * ,but not that from_string(s).to_string() == s (since there can be
     * multiple strings that parse to the same Target)...
     * *unless* t1 contains 'unknown' fields (in which case you'll get a string
     * that can't be parsed, which is intentional).
     */
    EXPORT std::string to_string() const;

    /**
     * Parse the contents of 'target' and merge into 'this',
     * replacing only the parts that are specified. (e.g., if 'target' specifies
     * only an arch, only the arch field of 'this' will be changed, leaving
     * the other fields untouched). Any features specified in 'target'
     * are added to 'this', whether or not originally present.
     *
     * If the string contains unknown tokens, or multiple tokens of the
     * same category (e.g. multiple arch values), return false
     * (possibly leaving 'this' munged). (Multiple feature specifications
     * will not cause a failure.)
     *
     * If 'target' contains "host" as the first token, it replaces the entire
     * contents of 'this' with get_host_target(), then proceeds to parse the
     * remaining tokens (allowing for things like "host-opencl" to mean
     * "host configuration, but with opencl added").
     *
     * Note that unlike parse_from_string(), this will never print to cerr or
     * assert in the event of a parse failure. Note also that an empty target
     * string is essentially a no-op, leaving 'this' unaffected.
     */
    EXPORT bool merge_string(const std::string &target);

    /**
     * Like merge_string(), but reset the contents of 'this' first.
     */
    EXPORT bool from_string(const std::string &target) {
	*this = Target();
	return merge_string(target);
    }
};

/** Return the target corresponding to the host machine. */
EXPORT Target get_host_target();

/** Return the target that Halide will use. If HL_TARGET is set it
 * uses that. Otherwise calls \ref get_host_target */
EXPORT Target get_target_from_environment();

/** Return the target that Halide will use for jit-compilation. If
 * HL_JIT_TARGET is set it uses that. Otherwise calls \ref
 * get_host_target. Throws an error if the architecture, bit width,
 * and OS of the target do not match the host target, so this is only
 * useful for controlling the feature set. */
EXPORT Target get_jit_target_from_environment();

/** Given a string of the form used in HL_TARGET (e.g. "x86-64-avx"),
 * return the Target it specifies. Note that this always starts with
 * the result of get_host_target(), replacing only the parts found in the
 * target string, so if you omit (say) an OS specification, the host OS
 * will be used instead. An empty string is exactly equivalent to get_host_target().
 */
EXPORT Target parse_target_string(const std::string &target);

namespace Internal {

/** Create an llvm module containing the support code for a given target. */
llvm::Module *get_initial_module_for_target(Target, llvm::LLVMContext *);

/** Create an llvm module containing the support code for ptx device. */
llvm::Module *get_initial_module_for_ptx_device(Target, llvm::LLVMContext *c);

}

}


#endif
