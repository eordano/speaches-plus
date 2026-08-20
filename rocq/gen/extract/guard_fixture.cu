// Ground truth for the enforcement-guard reader in launch_geometry.py.
//
// This file is NOT part of the corpus (it lives outside sources.json's
// corpus_root and is never extracted into GenLaunch.v).  It exists so that the
// answer "no guard" can be distinguished from "the reader is broken": every
// host function below is one shape, its expected verdict is in
// extract/selftest.py keyed by function name, and BOTH directions are asserted
// -- a case that must be recognised and a case that must not be.
//
// Every function puts the same identifier on gridDim.y, so the only variable
// between cases is the guard.

#include <cuda_runtime.h>

__global__ void gk(int x) {}

static const int kLimit = 65535;

#define LAUNCH(s, m)                                                          \
    do {                                                                      \
        dim3 g(1, (m));                                                       \
        gk<<<g, 32, 0, (cudaStream_t)(s)>>>((m));                             \
    } while (0)

extern "C" int case_early_return_gt(void* s, int m) {
    if (m > 65535) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_early_return_ge(void* s, int m) {
    if (m >= 65536) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_constant_on_the_left(void* s, int m) {
    if (65535 < m) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_named_constant(void* s, int m) {
    if (m > kLimit) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_or_chain(void* s, int m) {
    if (m <= 0 || m > 65535) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_early_return_in_a_block(void* s, int m, int* err) {
    if (m > 65535) {
        *err = 1;
        return -1;
    }
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_guarded_then_branch(void* s, int m) {
    if (m <= 65535) {
        LAUNCH(s, m);
        return 0;
    }
    return -1;
}

extern "C" int case_guarded_then_branch_and_chain(void* s, int m) {
    if (m > 0 && m < 65536) {
        LAUNCH(s, m);
        return 0;
    }
    return -1;
}

extern "C" int case_guarded_else_branch(void* s, int m) {
    if (m > 65535) {
        return -1;
    } else {
        LAUNCH(s, m);
    }
    return 0;
}

extern "C" int case_constant_on_the_left_then_branch(void* s, int m) {
    if (65535 >= m) {
        LAUNCH(s, m);
    }
    return 0;
}

extern "C" int case_nested_dominating_blocks(void* s, int m, int go) {
    if (m > 65535) return -1;
    if (go) {
        LAUNCH(s, m);
    }
    return 0;
}

// --- everything below must NOT be recognised -------------------------------

extern "C" int case_and_chain_early_return(void* s, int m) {
    if (m > 0 && m > 65535) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_guard_after_the_launch(void* s, int m) {
    LAUNCH(s, m);
    if (m > 65535) return -1;
    return 0;
}

extern "C" int case_guard_on_another_identifier(void* s, int m, int n) {
    if (n > 65535) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_guard_that_does_not_return(void* s, int m, int* err) {
    if (m > 65535) {
        *err = 1;
    }
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_guard_with_an_else(void* s, int m, int* err) {
    if (m > 65535) {
        return -1;
    } else {
        *err = 0;
    }
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_guarded_then_reassigned(void* s, int m) {
    if (m > 65535) return -1;
    m = m * 4;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_address_of_the_guarded_value(void* s, int m) {
    if (m > 65535) return -1;
    int* p = &m;
    *p = 262144;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_guard_inside_a_conditional_block(void* s, int m, int go) {
    if (go) {
        if (m > 65535) return -1;
    }
    LAUNCH(s, m);
    return 0;
}

// A guard BEFORE a switch dominates everything after it: the switch's case
// labels are reachable only through the switch statement, so no path skips the
// guard. This used to be refused because any switch anywhere disqualified the
// whole function.
extern "C" int case_guard_before_a_switch(void* s, int m, int sel) {
    if (m > 65535) return -1;
    switch (sel) {
        case 0:
            break;
        default:
            break;
    }
    LAUNCH(s, m);
    return 0;
}

// ... but a guard INSIDE the switch body does NOT dominate a launch in another
// case: control enters at a case label and skips everything above it. This is
// the case SWITCH_CUT exists for, and it must stay refused.
extern "C" int case_guard_inside_a_switch_case(void* s, int m, int sel) {
    switch (sel) {
        case 0:
            if (m > 65535) return -1;
            break;
        default:
            LAUNCH(s, m);
            break;
    }
    return 0;
}

extern "C" int case_function_contains_a_label(void* s, int m) {
    if (m > 65535) return -1;
    goto go;
go:
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_guard_is_not_a_constant(void* s, int m, int n) {
    if (m > n) return -1;
    LAUNCH(s, m);
    return 0;
}

extern "C" int case_lower_bound_only(void* s, int m) {
    if (m < 4) return -1;
    LAUNCH(s, m);
    return 0;
}
