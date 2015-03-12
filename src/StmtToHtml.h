#ifndef HALIDE_STMT_TO_HTML
#define HALIDE_STMT_TO_HTML

/** \file
 * Defines a function to dump an HTML-formatted stmt to a file.
 */

#include "IR.h"

namespace Halide {
namespace Internal {

/**
 * Dump an HTML-formatted print of a Stmt to filename.
 */
EXPORT void print_to_html(std::string filename, Stmt s);

}}

#endif
