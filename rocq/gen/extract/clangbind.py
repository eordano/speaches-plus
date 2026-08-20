#!/usr/bin/env python3
"""ctypes bindings for libclang, enough to walk a CUDA translation unit.

No third-party python dependencies (gen.py's standing rule): the `clang`
python package is not installed anywhere on this box, so the C API is bound
by hand.  Only the ~40 entry points the extractors actually use are declared;
an undeclared entry point is an AttributeError at import, never a silent
default-int return.

The one subtlety that is load-bearing for every caller: `in_file()` compares
the EXPANSION location's filename, not clang_Location_isFromMainFile.  That
predicate is false for macro-expanded cursors (their FileID is the macro
expansion buffer), and using it drops every launch that lives inside a
macro body -- measured at 93 of 275 launches (34%) on this corpus.
"""

import ctypes
import os
import sys

DEFAULT_LIBCLANG = "libclang.so"


class CXString(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("private_flags", ctypes.c_uint)]


class CXCursor(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("xdata", ctypes.c_int),
        ("data", ctypes.c_void_p * 3),
    ]


class CXType(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("data", ctypes.c_void_p * 2)]


class CXSourceLocation(ctypes.Structure):
    _fields_ = [("ptr_data", ctypes.c_void_p * 2), ("int_data", ctypes.c_uint)]


class CXSourceRange(ctypes.Structure):
    _fields_ = [
        ("ptr_data", ctypes.c_void_p * 2),
        ("begin_int_data", ctypes.c_uint),
        ("end_int_data", ctypes.c_uint),
    ]


VISITOR = ctypes.CFUNCTYPE(ctypes.c_int, CXCursor, CXCursor, ctypes.c_void_p)

CHILD_RECURSE = 2
CHILD_CONTINUE = 1
CHILD_BREAK = 0

_SIG = [
    ("clang_getCString", ctypes.c_char_p, [CXString]),
    ("clang_disposeString", None, [CXString]),
    ("clang_createIndex", ctypes.c_void_p, [ctypes.c_int, ctypes.c_int]),
    ("clang_disposeIndex", None, [ctypes.c_void_p]),
    (
        "clang_parseTranslationUnit",
        ctypes.c_void_p,
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ],
    ),
    ("clang_disposeTranslationUnit", None, [ctypes.c_void_p]),
    ("clang_getTranslationUnitCursor", CXCursor, [ctypes.c_void_p]),
    ("clang_getCursorKindSpelling", CXString, [ctypes.c_int]),
    ("clang_getCursorSpelling", CXString, [CXCursor]),
    ("clang_getCursorLocation", CXSourceLocation, [CXCursor]),
    ("clang_getCursorExtent", CXSourceRange, [CXCursor]),
    ("clang_getRangeStart", CXSourceLocation, [CXSourceRange]),
    ("clang_getRangeEnd", CXSourceLocation, [CXSourceRange]),
    (
        "clang_getExpansionLocation",
        None,
        [
            CXSourceLocation,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ],
    ),
    (
        "clang_getSpellingLocation",
        None,
        [
            CXSourceLocation,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ],
    ),
    ("clang_getFileName", CXString, [ctypes.c_void_p]),
    ("clang_getCursorReferenced", CXCursor, [CXCursor]),
    ("clang_Cursor_isNull", ctypes.c_int, [CXCursor]),
    ("clang_getCursorType", CXType, [CXCursor]),
    ("clang_getTypeSpelling", CXString, [CXType]),
    ("clang_getNumDiagnostics", ctypes.c_uint, [ctypes.c_void_p]),
    ("clang_getDiagnostic", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_uint]),
    ("clang_getDiagnosticSeverity", ctypes.c_uint, [ctypes.c_void_p]),
    ("clang_formatDiagnostic", CXString, [ctypes.c_void_p, ctypes.c_uint]),
    ("clang_disposeDiagnostic", None, [ctypes.c_void_p]),
    ("clang_visitChildren", ctypes.c_uint, [CXCursor, VISITOR, ctypes.c_void_p]),
    ("clang_Cursor_Evaluate", ctypes.c_void_p, [CXCursor]),
    ("clang_EvalResult_getKind", ctypes.c_int, [ctypes.c_void_p]),
    ("clang_EvalResult_getAsLongLong", ctypes.c_longlong, [ctypes.c_void_p]),
    ("clang_EvalResult_dispose", None, [ctypes.c_void_p]),
    ("clang_getCursorBinaryOperatorKind", ctypes.c_int, [CXCursor]),
    ("clang_getBinaryOperatorKindSpelling", CXString, [ctypes.c_int]),
    ("clang_getCursorUnaryOperatorKind", ctypes.c_int, [CXCursor]),
    ("clang_getUnaryOperatorKindSpelling", CXString, [ctypes.c_int]),
    ("clang_getNumOverloadedDecls", ctypes.c_uint, [CXCursor]),
    ("clang_getOverloadedDecl", CXCursor, [CXCursor, ctypes.c_uint]),
]

EVAL_INT = 1


class Clang:
    def __init__(self, libpath=None):
        self.libpath = libpath or os.environ.get("NV_LIBCLANG", DEFAULT_LIBCLANG)
        if not os.path.exists(self.libpath):
            raise RuntimeError(
                "libclang not found at %s; set NV_LIBCLANG" % self.libpath
            )
        self.lib = ctypes.cdll.LoadLibrary(self.libpath)
        for name, restype, argtypes in _SIG:
            fn = getattr(self.lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
        self.index = self.lib.clang_createIndex(0, 0)

    def s(self, cxstr):
        raw = self.lib.clang_getCString(cxstr)
        out = raw.decode(errors="replace") if raw else ""
        self.lib.clang_disposeString(cxstr)
        return out

    def kind(self, cur):
        return self.s(self.lib.clang_getCursorKindSpelling(cur.kind))

    def spelling(self, cur):
        return self.s(self.lib.clang_getCursorSpelling(cur))

    def type_spelling(self, cur):
        return self.s(self.lib.clang_getTypeSpelling(self.lib.clang_getCursorType(cur)))

    def is_null(self, cur):
        return bool(self.lib.clang_Cursor_isNull(cur))

    def referenced(self, cur):
        return self.lib.clang_getCursorReferenced(cur)

    def children(self, cur):
        out = []

        def visit(cc, _pp, _dd):
            out.append(CXCursor(cc.kind, cc.xdata, cc.data))
            return CHILD_CONTINUE

        cb = VISITOR(visit)
        self.lib.clang_visitChildren(cur, cb, None)
        return out

    def walk(self, cur, fn):
        """Depth-first over every descendant; fn returns a CHILD_* code."""
        keep = []

        def visit(cc, pp, _dd):
            c = CXCursor(cc.kind, cc.xdata, cc.data)
            p = CXCursor(pp.kind, pp.xdata, pp.data)
            keep.append(c)
            return fn(c, p)

        cb = VISITOR(visit)
        self.lib.clang_visitChildren(cur, cb, None)

    def _loc(self, which, loc):
        f = ctypes.c_void_p()
        ln = ctypes.c_uint()
        col = ctypes.c_uint()
        off = ctypes.c_uint()
        which(loc, ctypes.byref(f), ctypes.byref(ln), ctypes.byref(col), ctypes.byref(off))
        name = self.s(self.lib.clang_getFileName(f)) if f.value else ""
        return name, ln.value, col.value, off.value

    def expansion_loc(self, cur):
        return self._loc(
            self.lib.clang_getExpansionLocation, self.lib.clang_getCursorLocation(cur)
        )

    def spelling_loc(self, cur):
        return self._loc(
            self.lib.clang_getSpellingLocation, self.lib.clang_getCursorLocation(cur)
        )

    def spelling_extent(self, cur):
        rng = self.lib.clang_getCursorExtent(cur)
        a = self._loc(self.lib.clang_getSpellingLocation, self.lib.clang_getRangeStart(rng))
        b = self._loc(self.lib.clang_getSpellingLocation, self.lib.clang_getRangeEnd(rng))
        return a, b

    def expansion_extent(self, cur):
        rng = self.lib.clang_getCursorExtent(cur)
        a = self._loc(self.lib.clang_getExpansionLocation, self.lib.clang_getRangeStart(rng))
        b = self._loc(self.lib.clang_getExpansionLocation, self.lib.clang_getRangeEnd(rng))
        return a, b

    def eval_int(self, cur):
        """Constant-fold cur to a python int, or None."""
        res = self.lib.clang_Cursor_Evaluate(cur)
        if not res:
            return None
        try:
            if self.lib.clang_EvalResult_getKind(res) != EVAL_INT:
                return None
            return int(self.lib.clang_EvalResult_getAsLongLong(res))
        finally:
            self.lib.clang_EvalResult_dispose(res)

    def binop(self, cur):
        k = self.lib.clang_getCursorBinaryOperatorKind(cur)
        return self.s(self.lib.clang_getBinaryOperatorKindSpelling(k))

    def unop(self, cur):
        k = self.lib.clang_getCursorUnaryOperatorKind(cur)
        return self.s(self.lib.clang_getUnaryOperatorKindSpelling(k))

    def overloads(self, cur):
        n = self.lib.clang_getNumOverloadedDecls(cur)
        return [self.lib.clang_getOverloadedDecl(cur, i) for i in range(n)]

    def parse(self, path, args):
        argv = (ctypes.c_char_p * len(args))(*[a.encode() for a in args])
        tu = self.lib.clang_parseTranslationUnit(
            self.index, path.encode(), argv, len(args), None, 0, 0
        )
        return tu

    def errors(self, tu):
        out = []
        for i in range(self.lib.clang_getNumDiagnostics(tu)):
            d = self.lib.clang_getDiagnostic(tu, i)
            if self.lib.clang_getDiagnosticSeverity(d) >= 3:
                out.append(self.s(self.lib.clang_formatDiagnostic(d, 1)))
            self.lib.clang_disposeDiagnostic(d)
        return out

    def dispose(self, tu):
        self.lib.clang_disposeTranslationUnit(tu)
