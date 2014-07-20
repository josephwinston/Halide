#include "RDom.h"
#include "Util.h"
#include "IROperator.h"
#include "IRPrinter.h"

namespace Halide {

using std::string;
using std::vector;

RVar::operator Expr() const {
    if (!min().defined() || !extent().defined()) {
        user_error << "Use of undefined RDom dimension: " <<
            (name().empty() ? "<unknown>" : name()) << "\n";
    }
    return Internal::Variable::make(Int(32), name(), domain());
}

Internal::ReductionDomain build_domain(string name0, Expr min0, Expr extent0,
                                       string name1, Expr min1, Expr extent1,
                                       string name2, Expr min2, Expr extent2,
                                       string name3, Expr min3, Expr extent3) {
    vector<Internal::ReductionVariable> d;
    if (min0.defined()) {
        Internal::ReductionVariable v = {name0, min0, extent0};
        d.push_back(v);
    }
    if (min1.defined()) {
        Internal::ReductionVariable v = {name1, min1, extent1};
        d.push_back(v);
    }
    if (min2.defined()) {
        Internal::ReductionVariable v = {name2, min2, extent2};
        d.push_back(v);
    }
    if (min3.defined()) {
        Internal::ReductionVariable v = {name3, min3, extent3};
        d.push_back(v);
    }

    Internal::ReductionDomain dom(d);

    return dom;
}

RDom::RDom(Internal::ReductionDomain d) : dom(d) {
    const std::vector<Internal::ReductionVariable> &vars = dom.domain();
    if (vars.size() > 0) {
        x = RVar(vars[0].var, vars[0].min, vars[0].extent, d);
    }
    if (vars.size() > 1) {
        y = RVar(vars[1].var, vars[1].min, vars[1].extent, d);
    }
    if (vars.size() > 2) {
        z = RVar(vars[2].var, vars[2].min, vars[2].extent, d);
    }
    if (vars.size() > 3) {
        w = RVar(vars[3].var, vars[3].min, vars[3].extent, d);
    }
}

// We suffix all RVars with $r to prevent unintentional name matches with pure vars called x, y, z, w.
RDom::RDom(Expr min, Expr extent, string name) {
    min = cast<int>(min);
    extent = cast<int>(extent);
    if (name == "") name = Internal::make_entity_name(this, "Halide::RDom", 'r');
    dom = build_domain(name + ".x$r", min, extent,
                       "", Expr(), Expr(),
                       "", Expr(), Expr(),
                       "", Expr(), Expr());
    x = RVar(name + ".x$r", min, extent, dom);
    y = RVar(name + ".y");
    z = RVar(name + ".z");
    w = RVar(name + ".w");
}

RDom::RDom(Expr min0, Expr extent0, Expr min1, Expr extent1, string name) {
    min0 = cast<int>(min0);
    extent0 = cast<int>(extent0);
    min1 = cast<int>(min1);
    extent1 = cast<int>(extent1);
    if (name == "") name = Internal::make_entity_name(this, "Halide::RDom", 'r');
    dom = build_domain(name + ".x$r", min0, extent0,
                       name + ".y$r", min1, extent1,
                       "", Expr(), Expr(),
                       "", Expr(), Expr());
    x = RVar(name + ".x$r", min0, extent0, dom);
    y = RVar(name + ".y$r", min1, extent1, dom);
    z = RVar(name + ".z");
    w = RVar(name + ".w");
}

RDom::RDom(Expr min0, Expr extent0, Expr min1, Expr extent1, Expr min2, Expr extent2, string name) {
    min0 = cast<int>(min0);
    extent0 = cast<int>(extent0);
    min1 = cast<int>(min1);
    extent1 = cast<int>(extent1);
    min2 = cast<int>(min2);
    extent2 = cast<int>(extent2);
    if (name == "") name = Internal::make_entity_name(this, "Halide::RDom", 'r');
    dom = build_domain(name + ".x$r", min0, extent0,
                       name + ".y$r", min1, extent1,
                       name + ".z$r", min2, extent2,
                       "", Expr(), Expr());
    x = RVar(name + ".x$r", min0, extent0, dom);
    y = RVar(name + ".y$r", min1, extent1, dom);
    z = RVar(name + ".z$r", min2, extent2, dom);
    w = RVar(name + ".w");
}

RDom::RDom(Expr min0, Expr extent0, Expr min1, Expr extent1, Expr min2, Expr extent2, Expr min3, Expr extent3, string name) {
    min0 = cast<int>(min0);
    extent0 = cast<int>(extent0);
    min1 = cast<int>(min1);
    extent1 = cast<int>(extent1);
    min2 = cast<int>(min2);
    extent2 = cast<int>(extent2);
    min3 = cast<int>(min3);
    extent3 = cast<int>(extent3);
    if (name == "") name = Internal::make_entity_name(this, "Halide::RDom", 'r');
    dom = build_domain(name + ".x$r", min0, extent0,
                       name + ".y$r", min1, extent1,
                       name + ".z$r", min2, extent2,
                       name + ".w$r", min3, extent3);
    x = RVar(name + ".x$r", min0, extent0, dom);
    y = RVar(name + ".y$r", min1, extent1, dom);
    z = RVar(name + ".z$r", min2, extent2, dom);
    w = RVar(name + ".w$r", min3, extent3, dom);
}

RDom::RDom(Buffer b) {
    Expr min[4], extent[4];
    for (int i = 0; i < 4; i++) {
        if (b.dimensions() > i) {
            min[i] = b.min(i);
            extent[i] = b.extent(i);
        }
    }
    string names[] = {b.name() + ".x$r", b.name() + ".y$r", b.name() + ".z$r", b.name() + ".w$r"};
    dom = build_domain(names[0], min[0], extent[0],
                       names[1], min[1], extent[1],
                       names[2], min[2], extent[2],
                       names[3], min[3], extent[3]);
    RVar *vars[] = {&x, &y, &z, &w};
    for (int i = 0; i < 4; i++) {
        if (b.dimensions() > i) {
            *(vars[i]) = RVar(names[i], min[i], extent[i], dom);
        }
    }
}

RDom::RDom(ImageParam p) {
    Expr min[4], extent[4];
    for (int i = 0; i < 4; i++) {
        if (p.dimensions() > i) {
            min[i] = 0;
            extent[i] = p.extent(i);
        }
    }
    string names[] = {p.name() + ".x$r", p.name() + ".y$r", p.name() + ".z$r", p.name() + ".w$r"};
    dom = build_domain(names[0], min[0], extent[0],
                       names[1], min[1], extent[1],
                       names[2], min[2], extent[2],
                       names[3], min[3], extent[3]);
    RVar *vars[] = {&x, &y, &z, &w};
    for (int i = 0; i < 4; i++) {
        if (p.dimensions() > i) {
            *(vars[i]) = RVar(names[i], min[i], extent[i], dom);
        }
    }
}


int RDom::dimensions() const {
    return (int)dom.domain().size();
}

RVar RDom::operator[](int i) {
    if (i == 0) return x;
    if (i == 1) return y;
    if (i == 2) return z;
    if (i == 3) return w;
    user_error << "Reduction domain index out of bounds: " << i << "\n";
    return x; // Keep the compiler happy
}

RDom::operator Expr() const {
    if (dimensions() != 1) {
        user_error << "Error: Can't treat this multidimensional RDom as an Expr:\n"
                   << (*this) << "\n"
                   << "Only single-dimensional RDoms can be cast to Expr.\n";
    }
    return Expr(x);
}

RDom::operator RVar() const {
    if (dimensions() != 1) {
        user_error << "Error: Can't treat this multidimensional RDom as an RVar:\n"
                   << (*this) << "\n"
                   << "Only single-dimensional RDoms can be cast to RVar.\n";
    }
    return x;
}

/** Emit an RVar in a human-readable form */
std::ostream &operator<<(std::ostream &stream, RVar v) {
    stream << v.name() << "(" << v.min() << ", " << v.extent() << ")";
    return stream;
}

/** Emit an RDom in a human-readable form. */
std::ostream &operator<<(std::ostream &stream, RDom dom) {
    stream << "RDom(\n";
    for (int i = 0; i < dom.dimensions(); i++) {
        stream << "  " << dom[i] << "\n";
    }
    stream << ")\n";
    return stream;
}

}
