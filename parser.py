"""
DirectX .x and Bugsnax .xcache parsers
============================================
Produces a tree of XNode objects that the importer walks, regardless of
whether the source file is text, binary, MS-ZIP compressed (tzip/bzip), or
the SEMS .xcache binary format used by the Horsepower engine (Bugsnax).
"""

import math
import os
import re
import struct
import zlib
from typing import List, Optional

TOK_WORD   = "WORD"
TOK_STR    = "STR"
TOK_NUM    = "NUM"
TOK_LBRACE = "{"
TOK_RBRACE = "}"
TOK_SEMI   = ";"
TOK_COMMA  = ","
TOK_EOF    = "EOF"

_RE_TOKEN = re.compile(
    r'\"([^\"]*)\"|'
    r'(//[^\n]*)|'
    r'([{};,])|'
    r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)|'
    r'([A-Za-z_][A-Za-z0-9_.:]*)',
)

def _tokenize(text):
    tokens = []
    for m in _RE_TOKEN.finditer(text):
        s, comment, punc, num, word = m.groups()
        if comment:
            continue
        if s is not None:
            tokens.append((TOK_STR, s))
        elif punc:
            tokens.append((punc, punc))
        elif num is not None:
            tokens.append((TOK_NUM, num))
        elif word is not None:
            tokens.append((TOK_WORD, word))
    tokens.append((TOK_EOF, None))
    return tokens

class XNode:
    __slots__ = ("kind", "name", "children", "values", "meta")

    def __init__(self, kind, name=""):
        self.kind     = kind
        self.name     = name
        self.children = []
        self.values   = []
        # Optional metadata dict for parser→importer side-channel
        # info that doesn't belong in the .x format itself (e.g. the
        # original mesh name when a multi-material Mesh has been split
        # into per-material sub-meshes, for round-trip export merge).
        # Stays None unless explicitly populated; keeps memory cost
        # negligible for the typical case.
        self.meta: 'Optional[dict]' = None

    def child(self, *kinds):
        for c in self.children:
            if c.kind in kinds:
                return c
        return None

    def children_of(self, kind):
        return [c for c in self.children if c.kind == kind]

    def nums(self):
        return [float(v) for t, v in self.values if t == TOK_NUM]

    def ints(self):
        return [int(float(v)) for t, v in self.values if t == TOK_NUM]

    def strings(self):
        return [v for t, v in self.values if t == TOK_STR]

    def __repr__(self):
        return (f"<XNode {self.kind!r} name={self.name!r} "
                f"vals={len(self.values)} children={len(self.children)}>")

class _TextParser:
    def __init__(self, tokens):
        self._tok = tokens
        self._pos = 0

    def peek(self):
        return self._tok[self._pos]

    def consume(self):
        t = self._tok[self._pos]
        self._pos += 1
        return t

    def maybe(self, ttype):
        if self._tok[self._pos][0] == ttype:
            return self.consume()
        return None

    def parse_file(self):
        root = XNode("ROOT", "")
        while self.peek()[0] != TOK_EOF:
            n = self._parse_block()
            if n:
                root.children.append(n)
        return root

    def _parse_block(self):
        t = self.peek()
        if t[0] == TOK_LBRACE:
            self.consume()
            ref = XNode("REF")
            while self.peek()[0] != TOK_RBRACE and self.peek()[0] != TOK_EOF:
                ref.values.append(self.consume())
            self.maybe(TOK_RBRACE)
            return ref
        if t[0] in (TOK_SEMI, TOK_COMMA):
            self.consume()
            return None
        if t[0] == TOK_WORD:
            kind = self.consume()[1]
            name = ""
            if self.peek()[0] == TOK_WORD:
                name = self.consume()[1]
            if self.peek()[0] != TOK_LBRACE:
                return None
            self.consume()
            node = XNode(kind, name)
            self._fill_block(node)
            self.maybe(TOK_RBRACE)
            return node
        self.consume()
        return None

    def _fill_block(self, node):
        while True:
            t = self.peek()
            if t[0] in (TOK_RBRACE, TOK_EOF):
                return
            if t[0] == TOK_WORD:
                p1 = self._tok[self._pos + 1][0] if self._pos + 1 < len(self._tok) else TOK_EOF
                if p1 == TOK_LBRACE or p1 == TOK_WORD:
                    child = self._parse_block()
                    if child:
                        node.children.append(child)
                    continue
                node.values.append(self.consume())
                continue
            if t[0] == TOK_LBRACE:
                child = self._parse_block()
                if child:
                    node.children.append(child)
                continue
            if t[0] in (TOK_NUM, TOK_STR, TOK_SEMI, TOK_COMMA):
                node.values.append(self.consume())
                continue
            self.consume()

_BIN_TOK_NAME      = 0x01
_BIN_TOK_STRING    = 0x02
_BIN_TOK_INTEGER   = 0x03
_BIN_TOK_GUID      = 0x05
_BIN_TOK_INT_LIST  = 0x06
_BIN_TOK_FLT_LIST  = 0x07
_BIN_TOK_OBRACE    = 0x0a
_BIN_TOK_CBRACE    = 0x0b
_BIN_TOK_COMMA     = 0x13
_BIN_TOK_SEMICOLON = 0x14
_BIN_TOK_TEMPLATE  = 0x1f

_BIN_KEYWORD = {
    0x28: "WORD", 0x29: "DWORD", 0x2a: "FLOAT", 0x2b: "DOUBLE",
    0x2c: "CHAR", 0x2d: "UCHAR", 0x2e: "SWORD", 0x2f: "SDWORD",
    0x30: "void", 0x31: "string", 0x32: "unicode", 0x33: "cstring",
    0x34: "array",
}

class _BinaryParser:
    """Parse a DirectX binary token stream into XNode trees."""

    def __init__(self, buf: bytes, float_size: int):
        self._buf         = buf
        self._p           = 0
        self._end         = len(buf)
        self._float_fmt   = 'f' if float_size == 32 else 'd'
        self._float_bytes = float_size // 8
        self._num_count   = 0
        self._peeked      = None

    def _u16(self):
        v, = struct.unpack_from('H', self._buf, self._p); self._p += 2; return v

    def _u32(self):
        v, = struct.unpack_from('I', self._buf, self._p); self._p += 4; return v

    def _raw_int(self):
        return self._u32()

    def _raw_float(self):
        v, = struct.unpack_from(self._float_fmt, self._buf, self._p)
        self._p += self._float_bytes
        return v

    def _next_token(self):
        if self._p + 2 > self._end:
            return None, None
        tok = self._u16()
        if tok == _BIN_TOK_NAME:
            n = self._u32()
            s = self._buf[self._p:self._p + n].decode("latin-1"); self._p += n
            return tok, s
        if tok == _BIN_TOK_STRING:
            n = self._u32()
            s = self._buf[self._p:self._p + n].decode("latin-1"); self._p += n + 2
            return tok, s
        if tok == _BIN_TOK_INTEGER:
            return tok, self._u32()
        if tok == _BIN_TOK_GUID:
            self._p += 16; return tok, None
        if tok == _BIN_TOK_INT_LIST:
            count = self._u32(); self._num_count += count; return tok, count
        if tok == _BIN_TOK_FLT_LIST:
            count = self._u32(); self._num_count += count; return tok, count
        if tok in (_BIN_TOK_OBRACE, _BIN_TOK_CBRACE,
                   _BIN_TOK_COMMA, _BIN_TOK_SEMICOLON, _BIN_TOK_TEMPLATE):
            return tok, None
        if tok in _BIN_KEYWORD:
            return tok, _BIN_KEYWORD[tok]
        return tok, None

    def _peek(self):
        if self._peeked is None:
            p0, nc0 = self._p, self._num_count
            self._peeked = (self._next_token(), self._p, self._num_count)
            self._p, self._num_count = p0, nc0
        return self._peeked[0]

    def _eat_peeked(self):
        tok, new_p, new_nc = self._peeked
        self._peeked = None
        self._p, self._num_count = new_p, new_nc
        return tok

    def _get(self):
        if self._peeked:
            return self._eat_peeked()
        return self._next_token()

    def read_int(self) -> int:
        if self._num_count > 0:
            self._num_count -= 1
            return self._raw_int()
        while self._p + 2 <= self._end:
            p0 = self._p; tok = self._u16()
            if tok == _BIN_TOK_INT_LIST:
                cnt = self._u32(); self._num_count = cnt - 1
                return self._raw_int()
            if tok == _BIN_TOK_INTEGER:
                return self._u32()
            if tok in (_BIN_TOK_SEMICOLON, _BIN_TOK_COMMA):
                continue
            self._p = p0; break
        return 0

    def read_float(self) -> float:
        if self._num_count > 0:
            self._num_count -= 1
            return self._raw_float()
        while self._p + 2 <= self._end:
            p0 = self._p; tok = self._u16()
            if tok == _BIN_TOK_FLT_LIST:
                cnt = self._u32(); self._num_count = cnt - 1
                return self._raw_float()
            if tok == _BIN_TOK_INT_LIST:
                cnt = self._u32(); self._num_count = cnt - 1
                return float(self._raw_int())
            if tok in (_BIN_TOK_SEMICOLON, _BIN_TOK_COMMA):
                continue
            self._p = p0; break
        return 0.0

    def read_string(self) -> str:
        tok, val = self._get()
        return val if tok == _BIN_TOK_STRING else ""

    def skip_sep(self):
        while self._p + 2 <= self._end:
            p0 = self._p; tok = self._u16()
            if tok in (_BIN_TOK_SEMICOLON, _BIN_TOK_COMMA):
                continue
            self._p = p0; break

    def read_name_and_brace(self) -> str:
        tok, val = self._peek()
        name = ""
        if tok == _BIN_TOK_NAME:
            self._eat_peeked(); name = val
        tok2, _ = self._peek()
        if tok2 == _BIN_TOK_OBRACE:
            self._eat_peeked()
        return name

    def parse_file(self) -> XNode:
        root = XNode("ROOT", "")
        while self._p < self._end:
            tok, val = self._peek()
            if tok is None:
                break
            if tok != _BIN_TOK_NAME:
                self._eat_peeked(); continue
            node = self._dispatch(val)
            if node:
                root.children.append(node)
        return root

    def _dispatch(self, name_str: str) -> XNode | None:
        lv = (name_str or "").lower()
        self._eat_peeked()
        dispatch = {
            "animtickspersecond": self._p_anim_ticks,
            "frame":              self._p_frame,
            "mesh":               self._p_mesh,
            "animationset":       self._p_anim_set,
            "material":           self._p_material,
            "template":           self._p_template,
        }
        fn = dispatch.get(lv)
        if fn:
            return fn()
        return self._p_generic(name_str or "Unknown")

    def _p_generic(self, kind: str) -> XNode:
        name = self.read_name_and_brace()
        node = XNode(kind, name)
        depth = 1
        while self._p < self._end:
            tok, val = self._get()
            if tok is None:
                break
            if tok == _BIN_TOK_OBRACE:
                depth += 1
            elif tok == _BIN_TOK_CBRACE:
                depth -= 1
                if depth == 0:
                    break
            elif tok == _BIN_TOK_FLT_LIST:
                for _ in range(val):
                    node.values.append((TOK_NUM, repr(self._raw_float())))
                    self._num_count -= 1
            elif tok == _BIN_TOK_INT_LIST:
                for _ in range(val):
                    node.values.append((TOK_NUM, repr(self._raw_int())))
                    self._num_count -= 1
            elif tok == _BIN_TOK_STRING:
                node.values.append((TOK_STR, val))
            elif tok == _BIN_TOK_NAME:
                node.values.append((TOK_WORD, val))
        return node

    def _p_anim_ticks(self) -> XNode:
        node = XNode("AnimTicksPerSecond", "")
        self.read_name_and_brace()
        node.values.append((TOK_NUM, str(self.read_int())))
        self.skip_sep()
        self._get()
        return node

    def _p_template(self) -> None:
        depth = 0
        while self._p < self._end:
            tok, _ = self._get()
            if tok is None:
                break
            if tok == _BIN_TOK_OBRACE:
                depth += 1
            elif tok == _BIN_TOK_CBRACE:
                depth -= 1
                if depth <= 0:
                    return
        return None

    def _p_material(self) -> XNode:
        mat_name = self.read_name_and_brace()
        node = XNode("Material", mat_name)
        for _ in range(4):
            node.values.append((TOK_NUM, repr(self.read_float())))
        node.values.append((TOK_NUM, repr(self.read_float())))
        for _ in range(3):
            node.values.append((TOK_NUM, repr(self.read_float())))
        for _ in range(3):
            node.values.append((TOK_NUM, repr(self.read_float())))
        self.skip_sep()
        while self._p < self._end:
            tok, val = self._peek()
            if tok == _BIN_TOK_CBRACE:
                self._eat_peeked(); break
            if tok is None:
                break
            if tok == _BIN_TOK_NAME:
                self._eat_peeked()
                lv = (val or "").lower()
                child = self._p_tex_filename() if lv == "texturefilename" else self._p_generic(val or "Unknown")
                node.children.append(child)
            else:
                self._eat_peeked()
        return node

    def _p_tex_filename(self) -> XNode:
        node = XNode("TextureFileName", self.read_name_and_brace())
        node.values.append((TOK_STR, self.read_string()))
        self.skip_sep()
        self._get()
        return node

    def _p_frame(self) -> XNode:
        frame_name = self.read_name_and_brace()
        node = XNode("Frame", frame_name)
        while self._p < self._end:
            tok, val = self._peek()
            if tok == _BIN_TOK_CBRACE:
                self._eat_peeked(); break
            if tok is None:
                break
            if tok != _BIN_TOK_NAME:
                self._eat_peeked(); continue
            self._eat_peeked()
            lv = (val or "").lower()
            if lv == "frametransformmatrix":
                child = self._p_ftm()
            elif lv == "frame":
                child = self._p_frame()
            elif lv == "mesh":
                child = self._p_mesh()
            else:
                child = self._p_generic(val or "Unknown")
            node.children.append(child)
        return node

    def _p_ftm(self) -> XNode:
        self.read_name_and_brace()
        node = XNode("FrameTransformMatrix", "")
        for _ in range(16):
            node.values.append((TOK_NUM, repr(self.read_float())))
        self.skip_sep()
        self._get()
        return node

    def _p_mesh(self) -> XNode:
        mesh_name = self.read_name_and_brace()
        node = XNode("Mesh", mesh_name)

        n_verts = self.read_int()
        node.values.append((TOK_NUM, str(n_verts)))
        for _ in range(n_verts):
            for _ in range(3):
                node.values.append((TOK_NUM, repr(self.read_float())))
            self.skip_sep()

        n_faces = self.read_int()
        node.values.append((TOK_NUM, str(n_faces)))
        for _ in range(n_faces):
            cc = self.read_int()
            node.values.append((TOK_NUM, str(cc)))
            for _ in range(cc):
                node.values.append((TOK_NUM, str(self.read_int())))
            self.skip_sep()

        while self._p < self._end:
            tok, val = self._peek()
            if tok == _BIN_TOK_CBRACE:
                self._eat_peeked(); break
            if tok is None:
                break
            if tok != _BIN_TOK_NAME:
                self._eat_peeked(); continue
            self._eat_peeked()
            lv = (val or "").lower()
            if lv == "meshnormals":
                child = self._p_normals(n_faces)
            elif lv == "meshtexturecoords":
                child = self._p_uvs()
            elif lv == "meshmateriallist":
                child = self._p_matlist()
            elif lv == "xskinmeshheader":
                child = self._p_skinheader()
            elif lv == "skinweights":
                child = self._p_skinweights()
            else:
                child = self._p_generic(val or "Unknown")
            node.children.append(child)
        return node

    def _p_normals(self, n_faces: int) -> XNode:
        self.read_name_and_brace()
        node = XNode("MeshNormals", "")
        n = self.read_int()
        node.values.append((TOK_NUM, str(n)))
        for _ in range(n):
            for _ in range(3):
                node.values.append((TOK_NUM, repr(self.read_float())))
            self.skip_sep()
        node.values.append((TOK_NUM, str(n_faces)))
        for _ in range(n_faces):
            cc = self.read_int()
            node.values.append((TOK_NUM, str(cc)))
            for _ in range(cc):
                node.values.append((TOK_NUM, str(self.read_int())))
            self.skip_sep()
        self._get()
        return node

    def _p_uvs(self) -> XNode:
        self.read_name_and_brace()
        node = XNode("MeshTextureCoords", "")
        n = self.read_int()
        node.values.append((TOK_NUM, str(n)))
        for _ in range(n):
            node.values.append((TOK_NUM, repr(self.read_float())))
            node.values.append((TOK_NUM, repr(self.read_float())))
            self.skip_sep()
        self._get()
        return node

    def _p_matlist(self) -> XNode:
        self.read_name_and_brace()
        node = XNode("MeshMaterialList", "")
        n_mats = self.read_int()
        n_idx  = self.read_int()
        node.values.append((TOK_NUM, str(n_mats)))
        node.values.append((TOK_NUM, str(n_idx)))
        for _ in range(n_idx):
            node.values.append((TOK_NUM, str(self.read_int())))
        self.skip_sep()
        while self._p < self._end:
            tok, val = self._peek()
            if tok == _BIN_TOK_CBRACE:
                self._eat_peeked(); break
            if tok is None:
                break
            if tok == _BIN_TOK_OBRACE:
                self._eat_peeked()
                ref = XNode("REF")
                tok2, v2 = self._get()
                if tok2 == _BIN_TOK_NAME:
                    ref.values.append((TOK_WORD, v2))
                self._get()
                node.children.append(ref)
            elif tok == _BIN_TOK_NAME and (val or "").lower() == "material":
                self._eat_peeked()
                node.children.append(self._p_material())
            else:
                self._eat_peeked()
        return node

    def _p_skinheader(self) -> XNode:
        self.read_name_and_brace()
        node = XNode("XSkinMeshHeader", "")
        for _ in range(3):
            node.values.append((TOK_NUM, str(self.read_int())))
        self.skip_sep()
        self._get()
        return node

    def _p_skinweights(self) -> XNode:
        self.read_name_and_brace()
        node = XNode("SkinWeights", "")
        node.values.append((TOK_STR, self.read_string()))
        n = self.read_int()
        node.values.append((TOK_NUM, str(n)))
        for _ in range(n):
            node.values.append((TOK_NUM, str(self.read_int())))
        for _ in range(n):
            node.values.append((TOK_NUM, repr(self.read_float())))
        for _ in range(16):
            node.values.append((TOK_NUM, repr(self.read_float())))
        self.skip_sep()
        self._get()
        return node

    def _p_anim_set(self) -> XNode:
        set_name = self.read_name_and_brace()
        node = XNode("AnimationSet", set_name)
        while self._p < self._end:
            tok, val = self._peek()
            if tok == _BIN_TOK_CBRACE:
                self._eat_peeked(); break
            if tok is None:
                break
            if tok == _BIN_TOK_NAME and (val or "").lower() == "animation":
                self._eat_peeked()
                node.children.append(self._p_animation())
            else:
                self._eat_peeked()
        return node

    def _p_animation(self) -> XNode:
        self.read_name_and_brace()
        node = XNode("Animation", "")
        while self._p < self._end:
            tok, val = self._peek()
            if tok == _BIN_TOK_CBRACE:
                self._eat_peeked(); break
            if tok is None:
                break
            if tok == _BIN_TOK_OBRACE:
                self._eat_peeked()
                ref = XNode("REF")
                tok2, v2 = self._get()
                if tok2 == _BIN_TOK_NAME:
                    ref.values.append((TOK_WORD, v2))
                self._get()
                node.children.append(ref)
            elif tok == _BIN_TOK_NAME and (val or "").lower() == "animationkey":
                self._eat_peeked()
                node.children.append(self._p_anim_key())
            else:
                self._eat_peeked()
        return node

    def _p_anim_key(self) -> XNode:
        self.read_name_and_brace()
        node = XNode("AnimationKey", "")
        key_type  = self.read_int()
        key_count = self.read_int()
        node.values.append((TOK_NUM, str(key_type)))
        node.values.append((TOK_NUM, str(key_count)))

        # Per spec, each TimedFloatKeys carries `tick`, then `nValues`,
        # then nValues floats. Trust the stream's nValues over the
        # hardcoded lookup — unusual or version-specific encodings
        # might disagree with the conventional values per key_type.
        _n_vals = {0: 4, 1: 3, 2: 3, 3: 16, 4: 16}
        default_n = _n_vals.get(key_type, 4)

        for _ in range(key_count):
            tick = self.read_int()
            nv   = self.read_int()
            node.values.append((TOK_NUM, str(tick)))
            node.values.append((TOK_NUM, str(nv)))
            # Use stream's nv when sane, fall back to lookup when nv
            # is out of plausible range (corrupt stream).
            n_to_read = nv if 1 <= nv <= 64 else default_n
            for _ in range(n_to_read):
                node.values.append((TOK_NUM, repr(self.read_float())))
            self.skip_sep()

        self.skip_sep()
        self._get()
        return node

_MSZIP_MAGIC = 0x4B43

def _mszip_decompress(buf: bytes, start: int) -> bytes:
    chunks, p, end = [], start, len(buf)
    while p + 4 <= end:
        chunk_size = struct.unpack_from('H', buf, p)[0]; p += 2
        magic      = struct.unpack_from('H', buf, p)[0]; p += 2
        if magic != _MSZIP_MAGIC:
            raise ValueError(f"X: bad MSZIP magic 0x{magic:04x} at offset {p-2}")
        raw = buf[p:p + chunk_size]; p += chunk_size
        try:
            chunks.append(zlib.decompress(raw, wbits=-15))
        except zlib.error as exc:
            raise ValueError(f"X: zlib error in MSZIP chunk: {exc}") from exc
    return b"".join(chunks)



# =============================================================================
# Bugsnax .xcache (SEMS) binary format
# =============================================================================

def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from('<I', data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from('<f', data, offset)[0]


def _read_matrix(data: bytes, offset: int):
    """Read a 4×4 float32 matrix at *offset* (64 bytes). Returns a flat list of 16 floats."""
    return list(struct.unpack_from('<16f', data, offset))


def _find_vertex_block(data: bytes, start: int, end: int):
    """Scan *data[start:end]* for a plausible vertex-count uint32 followed by
    a contiguous run of valid 64-byte vertex records.

    Real Bugsnax verts have a specific 16-float layout per 64-byte record:
        [0:3]   position xyz
        [3:6]   normal xyz (unit length)
        [6]     usually NaN (vertex color sentinel)
        [7:9]   UV
        [9:16]  zeros (padding)

    Empirically validated against 499 xcache files in the dev dataset:
    real meshes ALWAYS pass all four of these probes on the first 5
    verts in the block, and fake "63-vert" candidates (regularly
    appearing in Queen/Arms/RPaw/Shot* xcaches) ALWAYS fail at least
    one of them. The strict 5/5/5/5 signature is what discriminates.

    Returns (vertex_count, vert_data_offset) or raises ValueError.
    """
    end = min(end, len(data) - 16)
    STRIDE = 64  # bytes per vertex (16 floats)

    def vert_signals(off):
        """Returns (pos_ok, nonzero, unit_normal, valid_uv, color_opaque).

        Validates that bytes at `off` look like a Bugsnax S3DVertexTangents
        record (60 bytes of data + 4 bytes padding = 64 bytes):
            [0:12]   Position (3 floats)
            [12:24]  Normal (3 floats, unit length)
            [24:28]  Color (u32 ARGB, always 0xFFFFFFFF = opaque white)
            [28:36]  TCoords / UV (2 floats)
            [36:48]  Tangent (3 floats, often zero — stored but unused)
            [48:60]  Binormal (3 floats, often zero)
            [60:64]  Padding (4 bytes zero)
        """
        if off + 64 > len(data):
            return False, False, False, False, False
        fx = _f32(data, off)
        fy = _f32(data, off + 4)
        fz = _f32(data, off + 8)
        # Position checks
        if math.isnan(fx) or math.isnan(fy) or math.isnan(fz):
            return False, False, False, False, False
        if abs(fx) > 10000 or abs(fy) > 10000 or abs(fz) > 10000:
            return False, False, False, False, False
        # Denormalized-float check (catches misaligned reads)
        if (0.0 < abs(fx) < 1e-15 or 0.0 < abs(fy) < 1e-15
                or 0.0 < abs(fz) < 1e-15):
            return False, False, False, False, False
        pos_ok = True
        nonzero = (abs(fx) + abs(fy) + abs(fz)) >= 1e-4
        # Normal at floats [3:6] must be unit length
        nx = _f32(data, off + 12)
        ny = _f32(data, off + 16)
        nz = _f32(data, off + 20)
        if (math.isnan(nx) or math.isnan(ny) or math.isnan(nz)
                or any(abs(v) > 1.5 for v in (nx, ny, nz))):
            unit_normal = False
        else:
            nmag = math.sqrt(nx*nx + ny*ny + nz*nz)
            unit_normal = 0.95 < nmag < 1.05
        # Color at u32 [24:28] in S3DVertexTangents. Bugsnax verts
        # are always OPAQUE (alpha byte = 0xFF in the high nibble of
        # the ARGB u32), but RGB can be anything from 0xFFFFFFFF
        # (default white) to 0xFFCCCCCC (PineappleLeaves/PineappleRings
        # have a baked gray tint) or other authored values. Only
        # check the alpha byte — that's a strong-enough signal
        # combined with the other four checks (random data has a
        # 1/256 chance of alpha=0xFF, but multiplied by passing the
        # position/normal/UV checks the false-positive rate is
        # effectively zero).
        color = struct.unpack_from('<I', data, off + 24)[0]
        color_opaque = ((color >> 24) & 0xFF) == 0xFF
        # UV at floats [7:9]: most files use 0..1 or -1..1, but some
        # outliers exceed that; tolerate up to ±10 (rejects garbage
        # but accepts authored-UV outliers).
        uu = _f32(data, off + 28)
        vv = _f32(data, off + 32)
        if math.isnan(uu) or math.isnan(vv):
            valid_uv = False
        else:
            valid_uv = abs(uu) < 10.0 and abs(vv) < 10.0
        return pos_ok, nonzero, unit_normal, valid_uv, color_opaque

    PROBE_N = 5      # how many vertices to look at for confidence

    for scan in range(start, end):
        candidate = struct.unpack_from('<I', data, scan)[0]
        # Bugsnax characters are typically 1k-50k verts; props (flags,
        # signs, etc.) can be as small as ~20. Below 20 the false-positive
        # rate gets too high — small integers in non-vertex byte regions
        # would match.
        if not (20 < candidate < 1_000_000):
            continue
        v0_off = scan + 4
        # Reject candidates whose claimed vertex count doesn't physically fit
        if v0_off + candidate * STRIDE + 8 > len(data):
            continue

        # Require ALL FIVE signals on ALL FIVE probed verts. This is
        # the strict signature that discriminates real meshes from
        # the recurring 63-entry fake structure that appears in many
        # xcaches (Queen has 11 fakes, Arms/Paws/Shot* have 1 each).
        # The opaque-alpha check at byte 27 of each vert is the
        # last-line filter — random data has a 1/256 chance of
        # matching, but combined with the other four signals the
        # overall false-positive rate is effectively zero.
        n_probe = min(PROBE_N, candidate)
        all_strict = True
        for vi in range(n_probe):
            pos_ok, nz, un, uv, co = vert_signals(v0_off + vi * STRIDE)
            if not (pos_ok and nz and un and uv and co):
                all_strict = False
                break
        if not all_strict:
            continue
        return candidate, v0_off
    raise ValueError("Could not locate vertex block")


# Bone / skeleton parsing

def _parse_bones(data: bytes, bone_count: int, start: int):
    """Walk *bone_count* bone entries starting at *start*."""
    bones = []
    offset = start

    for i in range(bone_count):
        if offset + 8 > len(data):
            break

        parent_idx = _u32(data, offset)
        name_len   = _u32(data, offset + 4)

        if name_len < 1 or name_len > 256:
            break  # corrupt or non-bone data

        name_bytes = data[offset + 8: offset + 8 + name_len]
        if not all(0x20 <= b < 0x7f for b in name_bytes):
            break
        name = name_bytes.decode('ascii')

        # The byte at (offset + 8 + name_len) is the FIRST byte of the FTM
        ftm_offset = offset + 8 + name_len
        if ftm_offset + 64 > len(data):
            break

        ftm = _read_matrix(data, ftm_offset)

        # The 64 bytes immediately after the FTM hold the bone's WORLD-SPACE
        bind_off = ftm_offset + 64
        bind_pose = [0.0] * 16
        if bind_off + 64 <= len(data):
            mat3x4 = struct.unpack_from('<12f', data, bind_off)
            trans4 = struct.unpack_from('<4f',  data, bind_off + 48)
            # Rows 0..2 from the 3x4 block, row 3 from the translation block.
            bind_pose[:12] = mat3x4
            bind_pose[12:] = trans4
        else:
            # Fallback: identity
            bind_pose[0] = bind_pose[5] = bind_pose[10] = bind_pose[15] = 1.0

        # The TRUE skinning offset matrix (= SkinWeights "matrixOffset"
        # in .x format) sits at ftm_offset + 296. The three 64-byte blocks
        # after the FTM are duplicates of the chained-FTM walk and the FTM
        # itself; the offset matrix is bit-identical to .x exports of
        # Orange/Spaghetti/Potato/Taco.
        skin_offset_at = ftm_offset + 296
        skin_offset = [0.0] * 16
        if skin_offset_at + 64 <= len(data):
            skin_offset = list(struct.unpack_from('<16f', data, skin_offset_at))
        else:
            # Fallback: identity (zero-matrix would be singular and
            # break skinning entirely)
            skin_offset[0] = skin_offset[5] = skin_offset[10] = skin_offset[15] = 1.0

        bones.append({
            'name':        name,
            'parent':      parent_idx,
            'ftm':         ftm,
            'bind_pose':   bind_pose,        # 4×4 world-space bind pose (DX row-major)
            'skin_offset': skin_offset,      # 4×4 skinning offset = inv(bone-bind in mesh space)
            'data_start':  ftm_offset + 64,
        })

        # Advance to next bone: we don't know the anim-data size, so we scan
        # forward for the next valid bone-header pattern.
        if i < bone_count - 1:
            next_offset = _find_next_bone_header(data, ftm_offset + 64)
            if next_offset is None:
                # Fall through — we'll try to parse mesh sections from here
                offset = ftm_offset + 64
                break
            offset = next_offset
        else:
            offset = ftm_offset + 64

    return bones, offset


def _find_next_bone_header(data: bytes, search_from: int) -> Optional[int]:
    """Scan forward from *search_from* for the next valid bone header.

    Uses byte-stride (not 4-byte stride) because bone headers in .xcache
    files are not guaranteed to be 4-aligned — the preceding bone's
    animation+skin region can end on any byte boundary.

    Search window: 5MB. Animation-heavy files (ChandloAnimation,
    BefficaAnimation) have ~250KB of animation curves per bone, so the
    previous 200KB cap stopped at bone 0. Real bone headers are
    distinctive enough (small parent_idx + valid name_len + ASCII name
    starting with a letter) that the validity checks filter random
    byte hits inside a wider scan.
    """
    end = min(search_from + 5_000_000, len(data) - 16)
    for off in range(search_from, end):
        parent = _u32(data, off)
        nlen   = _u32(data, off + 4)
        if parent > 60 or nlen < 3 or nlen > 80:
            continue
        nb = data[off + 8: off + 8 + nlen]
        if len(nb) != nlen:
            continue
        if not all(0x20 <= b < 0x7f for b in nb):
            continue
        name = nb.decode('ascii')
        if not (name[0].isupper() or name[0].isalpha()):
            continue
        return off
    return None


# Animation extraction

def _find_rot20_section(data: bytes, search_start: int, search_end: int):
    """Scan for a stride-20 block of unit quaternion keyframes."""
    for start in range(search_start, min(search_end - 100, search_start + 200_000), 4):
        # Fast pre-check: read 5 candidate records
        ok = True
        prev_frame = -1.0
        for i in range(5):
            off = start + i * 20
            if off + 20 > len(data):
                ok = False; break
            f = struct.unpack_from('<5f', data, off)
            frame = f[0]
            qlen  = math.sqrt(f[1]*f[1] + f[2]*f[2] + f[3]*f[3] + f[4]*f[4])
            if not (0 < frame < 300 and abs(qlen - 1.0) < 0.01):
                ok = False; break
            if frame <= prev_frame:
                ok = False; break
            prev_frame = frame
        if not ok:
            continue

        # Full scan
        entries = {}
        cur = start
        prev_frame = -1
        while cur + 20 <= len(data):
            f = struct.unpack_from('<5f', data, cur)
            if not all(math.isfinite(v) for v in f):
                break
            qlen = math.sqrt(f[1]*f[1] + f[2]*f[2] + f[3]*f[3] + f[4]*f[4])
            if abs(qlen - 1.0) > 0.05:
                break
            frame = f[0]
            if not (0 < frame < 1_000_000) or frame <= prev_frame:
                break
            prev_frame = frame
            entries[int(frame)] = (f[1], f[2], f[3], f[4])
            cur += 20
        if len(entries) > 10:
            return entries, cur

    return {}, search_start


def _find_skin_block(data: bytes, search_start: int, search_end: int):
    """Scan forward from *search_start* for a plausible skin-block header.

    Returns the offset of (count: u32, pad: u16) if found, else None.
    Used for "fixup" bones (e.g. Olive's RootFix) that have no rotation
    section between their FTM data and skin block — for those,
    _find_rot20_section returns search_start unchanged, leaving us
    pointed at the FTM data rather than the skin block. This function
    scans byte-by-byte for the next location where (count, pad, first
    20 entries) look like valid skin weights.

    Validates first 20 entries against the format:
        vi:      u32  (0..100k, chunk-relative)
        weight:  f32  (in [0, 1.001], no denormals)
        trailer: u16  (chunk index, 0..255)

    Real entries have trailer == pad (the first entry's trailer).
    """
    end = min(search_end - 6, search_start + 100_000)
    for off in range(search_start, end):
        count = struct.unpack_from('<I', data, off)[0]
        if count < 2 or count > 20_000:
            continue
        pad = struct.unpack_from('<H', data, off + 4)[0]
        if pad > 255:
            continue
        block_end = off + 6 + count * 10
        if block_end > search_end:
            continue
        # Validate first entries: each (vi:u32, w:f32, t:u16) must be sane,
        # weights must be real (≥ 1e-5 or exactly 0), and trailer must match
        # the pad for at least the first entry.
        sample_n = min(count, 20)
        ok = True
        n_real_weights = 0
        first_trailer = None
        for j in range(sample_n):
            eo = off + 6 + j * 10
            vi = struct.unpack_from('<I', data, eo)[0]
            w  = struct.unpack_from('<f', data, eo + 4)[0]
            t  = struct.unpack_from('<H', data, eo + 8)[0]
            # Sentinel — end of block, accept as termination
            if vi == 0x80000000:
                break
            if not math.isfinite(w):
                ok = False; break
            if not (0.0 <= w <= 1.001):
                ok = False; break
            if 0 < w < 1e-5:
                ok = False; break
            if w >= 1e-5:
                n_real_weights += 1
            if vi >= 100_000:
                ok = False; break
            if t > 255:
                ok = False; break
            if first_trailer is None:
                first_trailer = t
                # First-trailer should equal pad for real blocks
                if t != pad:
                    ok = False; break
        if ok and n_real_weights >= max(1, sample_n // 2):
            return off
    return None


def _read_skin_weights(data: bytes, offset: int, bone_end: int):
    """Parse the skin-weight block that immediately follows the rotation section.

    Format (10 bytes per entry):
        vi:      u32  (vertex index, CHUNK-RELATIVE within the entry's chunk)
        weight:  f32  (blend weight in [0, 1])
        trailer: u16  (chunk index this entry's vi belongs to)

    The header is (count: u32, pad: u16). `pad` is the bone's primary chunk
    (the first trailer encountered, informational). A single bone's skin
    block may contain entries for MULTIPLE chunks, distinguished by their
    trailer. For Queen.xcache's ROOT, the block has 6336 entries: 3584
    in chunk 0 (trailer=0), 1936 in chunk 1 (trailer=1), and 816 in chunk 6
    (trailer=6), each with its own chunk-relative vi range.

    Returns (influences, pad, resets) where:
      • influences is a list of (vi, weight, chunk_idx) — the caller is
        responsible for shifting vi to absolute by adding the chunk's
        base offset (chunk_idx looked up in offsets_per_mesh).
      • pad is the first chunk index encountered (informational).
      • resets is a list of indices where a "natural" vi reset happens
        WITHIN A SINGLE CHUNK (e.g. Honey's root spans multiple meshes
        within chunk 0). Chunk-boundary transitions (trailer change) are
        not recorded as resets — they're handled by the chunk shift.
    """
    if offset + 6 > bone_end:
        return [], 0, []
    count = struct.unpack_from('<I', data, offset)[0]
    if count == 0 or count > 50_000:
        return [], 0, []
    pad = struct.unpack_from('<H', data, offset + 4)[0]
    if pad > 255:
        return [], 0, []
    entries_start = offset + 6
    if entries_start + count * 10 > bone_end + 20:
        return [], 0, []
    influences = []
    resets = []
    prev_vi = -1
    SENTINEL_VI = 0x80000000  # marker for end-of-real-entries
    # Chunk interpretation rule (the "lookback" rule):
    # ----------------------------------------------------
    # Each skin entry has a trailer u16 at byte offset 8. The trailer
    # of entry N is the chunk index that entry N+1 belongs to — NOT
    # the current entry's chunk. Entry N's chunk is given by the
    # PREVIOUS entry's trailer (or the bone's `pad` u16 for entry [0]).
    #
    # This was verified against Journal.xcache where the trailer
    # values change at chunk boundaries (e.g. bone Journal_r_TopPage_
    # 01_03SHJnt has 18 entries with trailer=2 followed by entry
    # vi=29, trailer=0). With the lookback rule, vi=29 correctly
    # belongs to chunk 2 (it's the LAST chunk-2 entry, and its
    # trailer=0 announces that the NEXT entry starts chunk 0). The
    # .x ground truth confirms 20 chunk-2 entries for that bone,
    # which matches the lookback count exactly.
    #
    # The previous interpretation (each entry's trailer = its own
    # chunk) happened to work for files like Beffica and most of
    # Queen because trailers stayed CONSTANT across long runs of
    # weights — only entries at the chunk boundary differ between
    # the two interpretations.
    current_chunk = pad
    for i in range(count):
        off = entries_start + i * 10
        if off + 10 > len(data):
            break
        vi = struct.unpack_from('<I', data, off)[0]
        # End-of-data sentinel: vi == 0x80000000 marks end of real entries.
        if vi == SENTINEL_VI:
            break
        trailer = struct.unpack_from('<H', data, off + 8)[0]
        # Validate trailer is a plausible chunk index (small int).
        # Anything > 255 is parse garbage.
        if trailer > 255:
            break
        w = struct.unpack_from('<f', data, off + 4)[0]
        if not math.isfinite(w) or w < 0 or w > 1.0 + 1e-4:
            break
        # vi as u32 — must be a reasonable chunk-relative index.
        if vi >= 100_000:
            break
        # Detect WITHIN-CHUNK reset: vi drops by 100+ from previous vi
        # while the chunk index stays the same. This signals
        # continuation onto another mesh WITHIN THE SAME CHUNK
        # (Honey's root, Beffica's UpperJaw).
        if (prev_vi >= 100 and vi < prev_vi - 100
                and influences
                and influences[-1][2] == current_chunk):
            resets.append(len(influences))
        # Record THIS entry with `current_chunk` (looked up from
        # previous entry's trailer or `pad` for entry 0). The
        # trailer field of this entry tells us what chunk the NEXT
        # entry will be in.
        influences.append((vi, w, current_chunk))
        prev_vi = vi
        current_chunk = trailer
    return influences, pad, resets


def _extract_anim(data: bytes, bone: dict, next_bone_start: int):
    """Extract position, scale, and rotation animation keyframes plus skin
    weights from a single bone's data section in a .xcache file. Returns
    a dict with keys 'pos', 'scale', 'rot', 'skin', 'skin_pad', 'skin_resets'."""
    anim_start = bone['data_start']
    anim_end   = next_bone_start if next_bone_start else len(data)

    pos_keys   = {}
    scale_keys = {}

    # --- Find stride-16 section ---
    #
    # Position records are 4 floats (16 bytes) of the form
    # (x, y, tick, z) where tick is a monotonically increasing integer.
    # We scan for the first 4 consecutive records whose tick column is
    # an integer ≥ 1 and increases by exactly 1 each step, and whose
    # x/y/z values are finite and reasonably bounded.
    #
    # Bounds (deliberately generous to tolerate future content):
    #   |x|,|y|,|z| ≤ 10000  — typical Bugsnax models fit within ~50
    #                          units, but environment props and future
    #                          larger models could legitimately use
    #                          higher values; anything beyond 10k is
    #                          almost certainly uninit memory
    #   1 ≤ tick ≤ 100000    — animations don't have 100k frames
    #   |Δtick - 1| ≤ 0.5    — consecutive ticks differ by 1 exactly,
    #                          but allow rounding noise
    POS_BOUND   = 10000.0
    TICK_MAX    = 100_000
    TICK_TOL    = 0.5
    LOOKAHEAD   = 4         # records to validate before accepting start

    stride16_start = None
    for off in range(anim_start, anim_end - 16, 1):
        try:
            f0, f1, f2, f3 = struct.unpack_from('<4f', data, off)
        except struct.error:
            break
        if any(math.isnan(v) or abs(v) > POS_BOUND for v in (f0, f1, f3)):
            continue
        if math.isnan(f2) or math.isinf(f2):
            continue
        if f2 >= 1.0 and f2 == math.floor(f2) and f2 <= TICK_MAX:
            valid = True
            for step in range(1, LOOKAHEAD):
                noff = off + step * 16
                if noff + 16 > anim_end:
                    valid = False; break
                try:
                    g0, g1, g2, g3 = struct.unpack_from('<4f', data, noff)
                except struct.error:
                    valid = False; break
                if any(math.isnan(v) or abs(v) > POS_BOUND for v in (g0, g1, g3)):
                    valid = False; break
                if math.isnan(g2):
                    valid = False; break
                if abs(g2 - (f2 + step)) > TICK_TOL:
                    valid = False; break
            if valid:
                stride16_start = off
                break

    if stride16_start is None:
        # No animation data for this bone, but a skin block might
        # still exist further along (e.g. Olive's RootFix fixup
        # bone has 1136 skin weights but no rotation/pos/scale).
        skin, skin_pad, skin_resets = [], 0, []
        scan_from = _find_skin_block(data, anim_start, anim_end)
        if scan_from is not None:
            skin, skin_pad, skin_resets = _read_skin_weights(data, scan_from, anim_end)
        return {'pos': pos_keys, 'scale': scale_keys, 'rot': {},
                'skin': skin, 'skin_pad': skin_pad, 'skin_resets': skin_resets}

    # --- Parse stride-16 POSITION records ---

    pos_records = []
    off = stride16_start
    while off + 16 <= anim_end:
        try:
            f0, f1, f2, f3 = struct.unpack_from('<4f', data, off)
        except struct.error:
            break
        if any(math.isnan(v) or abs(v) > POS_BOUND for v in (f0, f1, f3)):
            break
        if math.isnan(f2) or math.isinf(f2):
            break
        if f2 < 0 or f2 != math.floor(f2):
            break
        pos_records.append((int(f2), f0, f1, f3))
        off += 16

    stride16_pos_end = off  # first byte after position block (the transition entry)

    # Reconstruct (X, Y, Z) using lag-1 lookahead.
    for i in range(len(pos_records) - 1):
        tick, _, _, v3_cur = pos_records[i]
        _,    v0_nxt, v1_nxt, _ = pos_records[i + 1]
        pos_keys[tick] = (v3_cur, v0_nxt, v1_nxt)

    # Last entry has no lookahead — use its own v0/v1 as approximation
    if len(pos_records) >= 2:
        tick_last, v0_l, v1_l, v3_l = pos_records[-1]
        pos_keys[tick_last] = (v3_l, v0_l, v1_l)

    # --- Parse stride-16 SCALE records (separate block after position) ---

    stride16_scale_end = stride16_pos_end + 16  # past transition
    off = stride16_scale_end
    scale_records = []
    while off + 16 <= anim_end:
        try:
            f0, f1, f2, f3 = struct.unpack_from('<4f', data, off)
        except struct.error:
            break
        if any(math.isnan(v) or abs(v) > POS_BOUND for v in (f0, f1, f2, f3)):
            break
        if f3 < 1 or f3 != math.floor(f3) or f3 > 10000:
            break
        scale_records.append((int(f3), f0, f1, f2))
        off += 16

    # If we found a real scale block, populate scale_keys and update end pointer.
    if scale_records:
        for tick, sx, sy, sz in scale_records:
            if tick >= 2:
                scale_keys[tick - 1] = (sx, sy, sz)
        # Replicate the last value forward to fill the missing final tick
        if scale_keys:
            last_tick = max(scale_keys)
            scale_keys[last_tick + 1] = scale_keys[last_tick]
        stride16_end = off
    else:
        # No scale block found — rotation block starts right after the
        # transition entry that the position parser stopped at.
        stride16_end = stride16_pos_end

    # --- Stride-20 rotation section (follows stride-16 blocks) ---
    rot_keys, rot_end = _find_rot20_section(data, stride16_end, anim_end)

    # Negate all quaternion components to convert xcache convention → .x world
    rot_keys = {tick: (-qx, -qy, -qz, -qw)
                for tick, (qx, qy, qz, qw) in rot_keys.items()}

    # --- Skin weight section (follows rotation section) ---
    # For mesh-frame bones like Olive's "OliveRig_RootFix" there is
    # no rotation section, so _find_rot20_section returns rot_end ==
    # stride16_end — pointing at FTM/matrix data rather than the
    # skin block. The skin block still exists further along the
    # bone's data, so scan for it by signature.
    skin, skin_pad, skin_resets = _read_skin_weights(data, rot_end, anim_end)
    if not skin:
        scan_from = _find_skin_block(data, rot_end, anim_end)
        if scan_from is not None:
            skin, skin_pad, skin_resets = _read_skin_weights(data, scan_from, anim_end)

    return {'pos': pos_keys, 'scale': scale_keys, 'rot': rot_keys,
            'skin': skin, 'skin_pad': skin_pad, 'skin_resets': skin_resets}


# Mesh parsing

def _parse_mesh_section(data: bytes, search_start: int):
    """Locate and parse all mesh sections from *search_start* onwards."""
    meshes = []

    # Match any printable name (3-80 chars) followed by \x00 — the null is the
    # LSByte of the first float in the following FTM matrix.  Mesh entries can
    # have any name (e.g. CeleryGeo, skinnedGrumpus, CrinkleFryder), so we
    # don't constrain the name shape here beyond a length floor.  False
    # positives are filtered by the FTM-validity and vertex-block-scan
    # checks downstream.
    #
    # Length floor of 6 chars: real Bugsnax mesh names are always at least
    # 6 chars (ChiliGeo, OliveGeo1, WatermelonGeo3, skinnedGrumpus,
    # ShishkebabGeoShape, ...). 3-char alphanumeric sequences ("SI6",
    # "SI7", "Z48", "vw8", "dds", "CQk") are statistically guaranteed to
    # appear inside random binary data — and they were causing the
    # parser to lock onto a garbage-passing vertex block (e.g. for
    # ChandloLUpperArm.xcache, "SI6" at 0x5ffe found the real mesh
    # data at 0x675c, but ended up naming it "SI6" instead of the
    # actual "skinnedGrumpus" name that immediately preceded the data).
    pattern = re.compile(rb'[A-Za-z][A-Za-z0-9_]{5,79}\x00')
    seen_offsets = set()
    # Also dedupe by the vert block's starting offset: the regex can
    # match the same name byte-pattern at multiple positions (e.g.
    # "SI6" appearing inside "SkinnedSI6Foo" and standalone), each
    # of which independently passes the FTM check and points to the
    # SAME vertex block downstream. Without this, Beffica's primary
    # 4068-vert block was being added 5 times.
    seen_vert_blocks = set()
    for m in pattern.finditer(data):
        geo_off  = m.start()
        name_end = m.end() - 1

        if geo_off in seen_offsets:
            continue

        name_len = name_end - geo_off
        if name_len < 3 or name_len > 80:
            continue

        name = m.group()[:-1].decode('ascii', errors='replace')

        # Skip names that match the joint-bone naming convention — those
        # entries have animation data following them, not a vertex block.
        if name.endswith('SHJnt'):
            continue

        ftm_off = name_end
        if ftm_off + 64 > len(data):
            continue

        transform = _read_matrix(data, ftm_off)

        if not any(abs(transform[i*4+i] - 1.0) < 0.1 for i in range(3)):
            continue

        try:
            vert_count, vert_data_off = _find_vertex_block(
                data, ftm_off + 64, ftm_off + 64 + 4096)
        except ValueError:
            continue

        STRIDE = 16  # floats per vertex (64 bytes)
        verts, normals, uvs = [], [], []
        for vi in range(vert_count):
            off = vert_data_off + vi * STRIDE * 4
            if off + STRIDE * 4 > len(data):
                break
            f = struct.unpack_from('<16f', data, off)
            verts.append((f[0], f[1], f[2]))
            normals.append((f[3], f[4], f[5]))
            uvs.append((f[7], f[8]))

        if len(verts) < vert_count:
            continue  # truncated

        # Sanity check: reject obviously bogus "meshes" where the
        # purported vert block is actually some other binary
        # structure (matrix dumps, animation curves, bone-index
        # lookup tables encoded as floats, etc).
        # Real meshes have all-finite positions in a reasonable
        # range; fixup frames (e.g. Olive's "OliveRig_RootFix")
        # have a regex-matchable name but no real geometry — the
        # bytes scanned as "verts" are mostly identity-matrix
        # patterns or garbage that misleads _find_vertex_block.
        bad = 0
        n_zero = 0
        n_int_x = 0    # X is a small integer 0..1000 (looks like an index, not a coord)
        n_denorm = 0   # Any coord in (0, 1e-30) — denormalized garbage
        sx_min = sx_max = verts[0][0]
        sy_min = sy_max = verts[0][1]
        sz_min = sz_max = verts[0][2]
        for x, y, z in verts:
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                bad += 1
                continue
            if abs(x) > 1e6 or abs(y) > 1e6 or abs(z) > 1e6:
                bad += 1
                continue
            # Denormalized-float detection. Real mesh coordinates are
            # never in (0, 1e-30) — typical Bugsnax coords are in
            # 0.001..100 range. When the parser locks onto a fake
            # mesh whose "vertex" bytes are actually animation/skin
            # block data (e.g. Watermelon's xcache produces a phantom
            # 32-vert "SI6" mesh at offset 0x566c8 with values like
            # 2.28e-41, 4.59e-41), those values reinterpret as
            # subnormal floats — they pass the finite/huge checks but
            # are clearly not geometry. A real vert with magnitude
            # below 1e-30 just doesn't exist in any character model.
            if (0.0 < abs(x) < 1e-30 or 0.0 < abs(y) < 1e-30
                    or 0.0 < abs(z) < 1e-30):
                n_denorm += 1
                continue
            if x == 0.0 and y == 0.0 and z == 0.0:
                n_zero += 1
            # Integer X in the range [0, 1000] is highly suspicious —
            # real mesh verts have continuous fractional coordinates.
            # Bone-index lookup tables (Shishkabob's "SI8" frame) get
            # mis-read as floats and show X = 1.0, 5.0, 9.0, 13.0, ...
            # which all pass `x == int(x)`.
            if 0 <= x <= 1000 and x == int(x):
                n_int_x += 1
            if x < sx_min: sx_min = x
            if x > sx_max: sx_max = x
            if y < sy_min: sy_min = y
            if y > sy_max: sy_max = y
            if z < sz_min: sz_min = z
            if z > sz_max: sz_max = z
        # If more than 5% of "verts" are non-finite or absurdly
        # large, this isn't a real mesh — skip it without marking
        # seen_offsets, so the real mesh (later in the file) can
        # still be picked up.
        if bad > max(8, vert_count // 20):
            continue
        # Denormalized coords are an unambiguous "not a real mesh"
        # signal. Even one is suspicious; allow 1 (the threshold-1
        # case is dominated by the bad/zero/int-x checks below).
        # A real mesh has zero denormals.
        if n_denorm > max(1, vert_count // 20):
            continue
        # All-zero vert ratio: real character meshes have
        # essentially zero verts at the exact origin. Fixup-frame
        # fake "vert blocks" are dominated by zeros (the matrix
        # entries that aren't on the diagonal). Reject if >5% of
        # verts are at (0, 0, 0).
        if n_zero > max(4, vert_count // 20):
            continue
        # Integer-X ratio: lookup-table data mis-read as verts has
        # many integer X values. Real character meshes have <5% (a
        # stylised cube might have a couple of corners at X=1.0).
        # Reject if >20% of verts have integer X in [0, 1000].
        if n_int_x > max(8, vert_count // 5):
            continue
        # Tight bbox check: a real character mesh has spatial
        # extent in all three axes. Fixup-frame fake "vert blocks"
        # are typically clamped to the unit cube with most entries
        # at integer coordinates. Require at least one axis to
        # span > 0.1 units.
        spans = (sx_max - sx_min, sy_max - sy_min, sz_max - sz_min)
        if max(spans) < 0.1:
            continue

        # Dedupe: if a different name has already produced a mesh
        # at this same vert_data_off, skip — the prior candidate
        # was the canonical one.
        if vert_data_off in seen_vert_blocks:
            continue
        seen_vert_blocks.add(vert_data_off)
        seen_offsets.add(geo_off)

        # Index buffers
        after_verts = vert_data_off + vert_count * STRIDE * 4
        buf1_start  = after_verts + 8   # skip 2 zero u32 separators

        # Detect the buf1/buf2 boundary.

        buf2_start_scan = None
        scan            = buf1_start
        prev_max_       = -1
        consec_         = 0

        while scan + 6 <= len(data):
            a, b, c = struct.unpack_from('<3H', data, scan)
            if max(a, b, c) >= vert_count:
                break
            expected_max = prev_max_ + 3
            if (max(a, b, c) == expected_max
                    and min(a, b, c) == prev_max_ + 1
                    and sorted([a, b, c]) == [prev_max_ + 1,
                                              prev_max_ + 2,
                                              prev_max_ + 3]):
                consec_ += 1
                if consec_ >= 3:
                    buf2_start_scan = scan - (consec_ - 1) * 6
                    break
            else:
                consec_ = 0
            prev_max_ = max(prev_max_, max(a, b, c))
            scan += 6

        if buf2_start_scan is None:
            buf2_start_scan = scan

        # Read buf1 faces (body/leg geometry — shared vertices)
        faces = []
        scan = buf1_start
        while scan < buf2_start_scan and scan + 6 <= len(data):
            a, b, c = struct.unpack_from('<3H', data, scan)
            if max(a, b, c) >= vert_count:
                break
            faces.append((a, b, c))
            scan += 6

        # Read buf2 faces (eye/accessory geometry — sequential)
        scan = buf2_start_scan
        while scan + 6 <= len(data):
            a, b, c = struct.unpack_from('<3H', data, scan)
            if max(a, b, c) >= vert_count:
                break
            faces.append((a, b, c))
            scan += 6

        # Texture paths: scan the material/flags block between the mesh FTM and
        # the vert data for .dds entries. The xcache stores each texture as
        # a structured entry — `01` presence byte + u32 name_length + name +
        # null terminator. We match the presence-byte structure first, then
        # fall back to a lenient .dds scan for files that don't have the
        # full prefix structure (e.g. older test data).
        tex_paths = []
        # Structured matcher: 0x01 + 4-byte LE length + N printable bytes + 0x00
        # where N == the length value. Authoritative for round-tripped exports.
        struct_pat = re.compile(rb'\x01(.{4})([\x20-\x7e]+\.dds)\x00', re.DOTALL)
        # The region between end-of-FTM and start-of-vertex-data
        mat_block_start = ftm_off + 64   # = data_start for this mesh
        mat_block_end   = vert_data_off  # vertex data begins here
        block = data[mat_block_start:mat_block_end]
        for tm in struct_pat.finditer(block):
            decl_len = struct.unpack('<I', tm.group(1))[0]
            path_bytes = tm.group(2)
            # Reject false positives: the declared length must match the
            # actual ASCII path length (without trailing null).
            if decl_len != len(path_bytes):
                continue
            path = path_bytes.decode('ascii', errors='replace')
            if path not in tex_paths:
                tex_paths.append(path)

        meshes.append({
            'name':      name,
            'transform': transform,
            'verts':     verts,
            'normals':   normals,
            'uvs':       uvs,
            'faces':     faces,
            'tex_paths': tex_paths,
            'data_end':  scan,   # offset where this mesh's face buffer ended
        })

    # Pass 2: pick up UNNAMED continuation meshes following the named
    # ones. Honey.xcache stores its model in two parts (named body +
    # unnamed goop overlay); Queen.xcache chains multiple chunks with
    # 12-byte separators (uint32=2, then 8 bytes of padding). The
    # face-index reader stops on the face-triple (0, 0, 0) inside the
    # separator, so we probe several alignments around data_end to
    # find the next mesh's bbox header.
    while meshes:
        last_end = meshes[-1].get('data_end')
        if last_end is None or last_end + 88 > len(data):
            break

        # Try multiple alignments. 0 is the legacy behaviour (no
        # separator). +6 covers the case where the parser stopped
        # on the (0,0,0) inside a 12-byte separator. +2/+4 catch
        # variants where the face reader stopped one byte short.
        hdr = None
        chosen_off = None
        for skip in (0, 2, 4, 6, 12):
            probe_off = last_end + skip
            if probe_off + 88 > len(data):
                continue
            try:
                probe = struct.unpack_from('<22f', data, probe_off)
            except struct.error:
                continue
            if not all(math.isfinite(v) for v in probe[:22]):
                continue
            # Identity matrix diag at indices 6, 11, 16, 21
            if not (abs(probe[6] - 1.0) < 0.1 and
                    abs(probe[11] - 1.0) < 0.1 and
                    abs(probe[16] - 1.0) < 0.1 and
                    abs(probe[21] - 1.0) < 0.1):
                continue
            # Off-diagonal must be near-zero — rejects coincidental
            # 1.0 values inside vertex data or animation keys.
            if any(abs(probe[6 + r*4 + c]) > 0.1
                   for r in range(4) for c in range(4) if r != c):
                continue
            # bbox must be finite, non-degenerate (some non-zero
            # extent), and within sane bounds.
            bbox = probe[:6]
            if not all(abs(v) < 1e6 for v in bbox):
                continue
            if not any(abs(v) > 0.01 for v in bbox):
                continue
            hdr = probe
            chosen_off = probe_off
            break

        if hdr is None:
            break
        # Find the vert block in the region after the header.
        try:
            vert_count, vert_data_off = _find_vertex_block(
                data, chosen_off + 88, chosen_off + 88 + 8192)
        except ValueError:
            break

        STRIDE = 16
        verts, normals, uvs = [], [], []
        for vi in range(vert_count):
            off = vert_data_off + vi * STRIDE * 4
            if off + STRIDE * 4 > len(data):
                break
            f = struct.unpack_from('<16f', data, off)
            verts.append((f[0], f[1], f[2]))
            normals.append((f[3], f[4], f[5]))
            uvs.append((f[7], f[8]))
        if len(verts) < vert_count:
            break

        # Face buffer: simple sequential read (no buf1/buf2 split for the
        # continuation mesh; faces just stop when index >= vert_count).
        after_verts = vert_data_off + vert_count * STRIDE * 4
        buf1_start  = after_verts + 8
        faces = []
        scan = buf1_start
        while scan + 6 <= len(data):
            a, b, c = struct.unpack_from('<3H', data, scan)
            if max(a, b, c) >= vert_count:
                break
            if a == b == c == 0 and len(faces) > 0:
                # trailing zero padding past real faces
                break
            faces.append((a, b, c))
            scan += 6

        # Texture paths inside the header region (between bbox+matrix and vert data).
        # Same structured matcher as pass 1 — accepts any .dds path, not just
        # Content/-prefixed ones, to support round-tripped xcache files.
        tex_paths = []
        struct_pat = re.compile(rb'\x01(.{4})([\x20-\x7e]+\.dds)\x00', re.DOTALL)
        for tm in struct_pat.finditer(data[chosen_off + 88:vert_data_off]):
            decl_len = struct.unpack('<I', tm.group(1))[0]
            path_bytes = tm.group(2)
            if decl_len != len(path_bytes):
                continue
            path = path_bytes.decode('ascii', errors='replace')
            if path not in tex_paths:
                tex_paths.append(path)

        # Synthesize a unique name for this continuation mesh based
        # on the FIRST (named) mesh, not the previous chunk — Queen
        # has 11 chunks and would otherwise produce names like
        # QueenGeo_Part2_Part3_Part4_..._Part11.
        base_name = meshes[0]['name'] if meshes else 'Mesh'
        anon_name = f"{base_name}_Part{len(meshes)+1}"
        # Build a 4x4 transform matrix from the header floats.
        transform = list(hdr[6:22])

        meshes.append({
            'name':      anon_name,
            'transform': transform,
            'verts':     verts,
            'normals':   normals,
            'uvs':       uvs,
            'faces':     faces,
            'tex_paths': tex_paths,
            'data_end':  scan,
        })

    # Strip the bookkeeping field before returning.
    for m in meshes:
        m.pop('data_end', None)

    # ---- Name-independent direct-scan fallback ----
    #
    # If the regex-driven scan above found nothing, fall back to a
    # direct byte-by-byte scan for vertex-block signatures. This is
    # the only way to find meshes in files that have no >=6-char
    # ASCII name candidate anywhere — e.g. IceCream.xcache (only
    # name in the mesh region is the 3-char bone "Neo") and ~440
    # other files in the dev xcache dataset (88% of the corpus).
    #
    # Empirically validated against 499 dev xcaches: the strict
    # 5/5/5/5 signature in _find_vertex_block (5 probed verts all
    # passing position-finite, position-nonzero, unit-length normal
    # at [3:6], in-range UV at [7:9]) gives zero false positives,
    # so we can scan the entire post-bones region without name
    # anchoring.
    if not meshes:
        # Start scanning from `search_start` (passed in by caller,
        # typically the offset right after the bone block).
        scan = max(0, search_start)
        STRIDE = 16  # floats per vertex (64 bytes)
        while scan < len(data) - 64 * 5 - 4:
            try:
                vert_count, vert_data_off = _find_vertex_block(
                    data, scan, scan + 1)
            except ValueError:
                scan += 1
                continue

            # Successfully located a vert block. Extract verts, normals,
            # UVs the same way as the regex path.
            verts, normals, uvs = [], [], []
            for vi in range(vert_count):
                off = vert_data_off + vi * STRIDE * 4
                if off + STRIDE * 4 > len(data):
                    break
                f = struct.unpack_from('<16f', data, off)
                verts.append((f[0], f[1], f[2]))
                normals.append((f[3], f[4], f[5]))
                uvs.append((f[7], f[8]))

            if len(verts) < vert_count:
                # Truncated: skip past this block and keep searching.
                scan = vert_data_off + 4
                continue

            # Face buffer extraction (mirrors regex-path logic).
            after_verts = vert_data_off + vert_count * STRIDE * 4
            buf1_start  = after_verts + 8

            buf2_start_scan = None
            face_scan = buf1_start
            prev_max_ = -1
            consec_   = 0
            while face_scan + 6 <= len(data):
                a, b, c = struct.unpack_from('<3H', data, face_scan)
                if max(a, b, c) >= vert_count:
                    break
                expected_max = prev_max_ + 3
                if (max(a, b, c) == expected_max
                        and min(a, b, c) == prev_max_ + 1
                        and sorted([a, b, c]) == [prev_max_ + 1,
                                                  prev_max_ + 2,
                                                  prev_max_ + 3]):
                    consec_ += 1
                    if consec_ >= 3:
                        buf2_start_scan = face_scan - (consec_ - 1) * 6
                        break
                else:
                    consec_ = 0
                prev_max_ = max(prev_max_, max(a, b, c))
                face_scan += 6
            if buf2_start_scan is None:
                buf2_start_scan = face_scan

            faces = []
            face_scan = buf1_start
            while face_scan < buf2_start_scan and face_scan + 6 <= len(data):
                a, b, c = struct.unpack_from('<3H', data, face_scan)
                if max(a, b, c) >= vert_count:
                    break
                faces.append((a, b, c))
                face_scan += 6
            face_scan = buf2_start_scan
            while face_scan + 6 <= len(data):
                a, b, c = struct.unpack_from('<3H', data, face_scan)
                if max(a, b, c) >= vert_count:
                    break
                faces.append((a, b, c))
                face_scan += 6
            data_end = face_scan

            # Texture paths in the region before this mesh's verts.
            tex_paths = []
            tex_pat = re.compile(rb'Content/[^\x00]+\.dds\x00')
            tex_scan_start = max(0, vert_data_off - 4096)
            for tm in tex_pat.finditer(data, tex_scan_start, vert_data_off):
                path = tm.group()[:-1].decode('ascii', errors='replace')
                if path not in tex_paths:
                    tex_paths.append(path)

            # No name available — caller (or downstream importer) can
            # derive one from the filename or bone-list mesh-entry.
            # Use a placeholder.
            base_name = f"Mesh_{len(meshes) + 1}" if meshes else "Mesh"

            meshes.append({
                'name':      base_name,
                'transform': [1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0],
                'verts':     verts,
                'normals':   normals,
                'uvs':       uvs,
                'faces':     faces,
                'tex_paths': tex_paths,
            })

            # Continue scanning from after this mesh's face buffer to
            # find any subsequent meshes (multi-mesh files like
            # Beffica have 2+ meshes spaced ~10kb apart in the file).
            scan = max(data_end, vert_data_off + vert_count * 64)

    return meshes


def _is_mesh_bone_name(name: str) -> bool:
    """Heuristic: bone-list entries that aren't actual joint bones.

    In xcache files, the LAST entry in the bone list is always the
    mesh entry — its name is the asset name (e.g. 'CarrotGeo',
    'skinnedGrumpus', 'HoneyStick'). Real skeletal joints follow
    the Maya HumanIK convention of ending in 'SHJnt'.

    However, some files have physics-rig joints with non-SHJnt
    names (DragonRoll's Root/Head/Body1..8/Tail, Pendulum's
    PendulumRoot/Pendulum, Trampoline's Root/Bounce, Olive's
    RootFix, dottedLine's line0..line8). These ARE joints and
    should NOT be flagged as mesh entries.

    The distinction: a name is a "mesh entry" only if it matches
    KNOWN mesh-asset-name patterns. Specifically:
      • ends in 'Geo' or 'Geometry' (CarrotGeo, PendulumGeometry)
      • ends in 'Geo' followed by a digit (OliveGeo1)
      • starts with 'skinned' (skinnedGrumpus)
      • is exactly 'HoneyStick' (Honey's mesh — unique pattern)
      • is exactly 'Trampoline' (its mesh entry — matches the
        file name with no other suffix)
      • is exactly 'line' (dottedLine's mesh entry — singular,
        vs the joint names line0..line8)

    Anything else — SHJnt-suffixed or not — is treated as a real
    joint. This brings back DragonRoll/CabinLift/Pendulum prop
    rigs and other physics-bone files that the old "not endswith
    SHJnt" check incorrectly filtered.
    """
    if name.endswith('SHJnt'):
        return False
    # Match mesh-name patterns
    if name.endswith('Geo') or name.endswith('Geometry'):
        return True
    if name.startswith('skinned'):
        return True
    if name in ('HoneyStick', 'Trampoline', 'line'):
        return True
    # Ends in 'Geo' + digit(s): OliveGeo1, FloorGeo2, etc.
    if re.search(r'Geo\d+$', name):
        return True
    # Anything else is a real joint (DragonRoll's Root/Head/Body*/Tail,
    # CabinLift's Root/Attach, Pendulum's PendulumRoot/Pendulum,
    # Olive's RootFix, dottedLine's line0..line8, etc.).
    return False


# XNode tree construction

def _num(v) -> tuple:
    return (TOK_NUM, repr(float(v)))


def _str(s: str) -> tuple:
    return (TOK_STR, s)


def _word(w: str) -> tuple:
    return (TOK_WORD, w)


def _make_ftm_node(matrix_flat: list[float]) -> XNode:
    node = XNode("FrameTransformMatrix", "")
    for v in matrix_flat:
        node.values.append(_num(v))
    return node


def _make_frame_node(name: str, ftm: list[float], children: list) -> XNode:
    node = XNode("Frame", name)
    node.children.append(_make_ftm_node(ftm))
    node.children.extend(children)
    return node


def _resolve_parent_indices(bones: list) -> list:
    """Recover each bone's parent for the .xcache flat bone list.

    The SEMS bone list is emitted in canonical depth-first order. The
    4-byte ``parent`` field at offset 0 of each bone header behaves
    as a binary descend-vs-pop signal rather than a literal index:

      * ``parent != 0`` → this bone is a direct child of the
        IMMEDIATELY PRECEDING bone in the list ("descend the DFS").
      * ``parent == 0`` → this bone is a sibling of some open ancestor
        ("pop the DFS until we find an ancestor whose world bind
        satisfies ftm @ bind == this_bone.bind").

    Geometric inference alone (`ftm @ parent_bind ≈ bind`) is not
    enough to disambiguate the pop case when several ancestor bones
    share an identical world-bind pose — e.g. Chili's ROOT, Broth_Aux
    and Noodles_Aux all sit at (0, 2.5255, 0) with identity rotation,
    so the eye/lid bones could legally claim any of them as parent.

    This function combines three signals:

      1. The first joint bone in the file is always the skeleton root
         (parent = -1).  This prevents false-positives like Chili's
         ROOT being parented to a later identity-bind bone such as
         HoldAuxSHJnt.

      2. When ``parent != 0``, attach to ``bones[i - 1]`` directly
         (after a quick sanity check on the geometric constraint).
         This is the definitive "descend" path and stops LCP
         tie-breaks from mis-picking ROOT instead of the real anchor.

      3. Otherwise fall back to geometric inference with name-LCP
         tie-break (Honey/Sandwich-style hierarchies rely on this),
         then apply a "sticky-descent" override: if the IMMEDIATELY
         PREVIOUS bone's resolved parent is also a valid geometric
         candidate, prefer it over the LCP-tied alternative.  This
         keeps siblings under the same anchor through any chain of
         identity-related parent bones (fixing the Chili pupil / lid
         case where r_Pupil through r_LowerLidClosed must all attach
         to Noodles_Aux rather than aliasing up to ROOT).

    Bones that fail the joint-name test (mesh frames like
    'CarrotGeo', 'HoneyStick') keep parents[i] == -1 — they are
    skipped by `_build_skeleton_frames`.
    """
    parents = [-1] * len(bones)
    if not bones:
        return parents

    TOL = 1e-3
    # Primary "is this a joint?" test: ends with the Bugsnax SHJnt
    # suffix. If the file uses any other naming convention (no bones
    # end in SHJnt at all) fall back to treating every non-mesh bone
    # as a joint candidate — signals 2 and 3 still work without the
    # suffix gate. This keeps files from other Horsepower-engine
    # games / future Bugsnax variants importable as a real hierarchy
    # rather than collapsing to a flat list of roots.
    #
    # Additionally, any bone with skin data is treated as a joint
    # regardless of suffix. This catches "fixup" frames like Olive's
    # `OliveRig_RootFix` — non-SHJnt names that nonetheless carry
    # 1000+ vertex weights. Without this, RootFix stays at parent=-1
    # (treated as an independent root) and the body verts weighted to
    # it don't follow the actual skeleton root during animation, even
    # though the file's intent is clearly for RootFix to be nested
    # inside ROOTSHJnt.
    is_joint = [b['name'].endswith('SHJnt') or bool(b.get('skin'))
                for b in bones]
    if not any(is_joint):
        is_joint = [not _is_mesh_bone_name(b['name']) for b in bones]

    def _lcp(a, b):
        n = min(len(a), len(b))
        for k in range(n):
            if a[k] != b[k]:
                return k
        return n

    first_joint_seen = False

    for i in range(len(bones)):
        if not is_joint[i]:
            continue

        # Signal 1: first joint bone is the skeleton root.
        if not first_joint_seen:
            parents[i] = -1
            first_joint_seen = True
            continue

        ftm  = bones[i]['ftm']
        bind = bones[i]['bind_pose']

        # Signal 2: parent_field != 0 means "child of previous bone".
        # Trust this over the geometric search — otherwise LCP
        # tie-breaks can pick a wrong parent when several ancestors
        # share an identical world-bind pose.
        pf = bones[i]['parent']
        if pf != 0 and i > 0 and is_joint[i - 1]:
            prev_bind = bones[i - 1]['bind_pose']
            comp = _mat_mul(ftm, prev_bind)
            err_prev = max(abs(comp[k] - bind[k]) for k in range(16))
            if err_prev < TOL:
                parents[i] = i - 1
                continue
            # Fall through to geometric search if the previous bone
            # doesn't satisfy the constraint (rare; some files have
            # a non-zero parent_field that doesn't actually mean
            # "child of previous").

        # Signal 3: geometric inference with LCP tie-break.
        best, best_err = -1, float('inf')
        for j in range(len(bones)):
            if j == i or not is_joint[j]:
                continue
            par_bind = bones[j]['bind_pose']
            comp = _mat_mul(ftm, par_bind)
            err = max(abs(comp[k] - bind[k]) for k in range(16))
            if err < best_err:
                best_err = err
                best = j

        if not (best_err < TOL and best != -1 and best != i):
            parents[i] = -1
            continue

        # Tie-break: prefer the candidate whose name shares the
        # longest common prefix with the child (handles bilateral
        # pairs like l_/r_ and name-hierarchy like Smoothie /
        # UmbrellaNeck / UmbrellaBase).
        child_name = bones[i]['name']
        best_lcp   = _lcp(child_name, bones[best]['name'])
        EPS_TIE = best_err * 10 + 1e-9
        for j in range(len(bones)):
            if j == i or j == best or not is_joint[j]:
                continue
            par_bind = bones[j]['bind_pose']
            comp = _mat_mul(ftm, par_bind)
            err_j = max(abs(comp[k] - bind[k]) for k in range(16))
            if err_j < EPS_TIE:
                lcp_j = _lcp(child_name, bones[j]['name'])
                if lcp_j > best_lcp:
                    best     = j
                    best_lcp = lcp_j

        # Sticky-descent override: if the immediately preceding bone's
        # resolved parent is ALSO a valid geometric candidate, prefer
        # it.  Fixes the Chili case where r_Pupil through
        # r_LowerLidClosed all have geometric LCP-tie that picks ROOT,
        # but their sibling l_Pupil already attached to Noodles_Aux
        # (via the parent_field signal above) — they should follow.
        prev_parent = parents[i - 1] if i > 0 else -1
        if (prev_parent >= 0 and prev_parent != best
                and is_joint[prev_parent]):
            par_bind = bones[prev_parent]['bind_pose']
            comp = _mat_mul(ftm, par_bind)
            err_pp = max(abs(comp[k] - bind[k]) for k in range(16))
            if err_pp < TOL:
                best = prev_parent

        parents[i] = best

    return parents


def _mat_inv(m16: list) -> list:
    """Invert a 4×4 row-major DirectX-style matrix stored as 16 floats."""
    try:
        from mathutils import Matrix
        m = Matrix([m16[0:4], m16[4:8], m16[8:12], m16[12:16]]).transposed()
        try:
            inv = m.inverted()
        except Exception:
            inv = Matrix.Identity(4)
        inv_t = inv.transposed()
        return [inv_t[r][c] for r in range(4) for c in range(4)]
    except ImportError:
        # Pure-Python fallback (only used outside Blender, e.g. for testing)
        # Use a generic 4x4 inverse via the adjugate / cofactor method.
        a = [[m16[r*4+c] for c in range(4)] for r in range(4)]
        n = 4
        # Build augmented matrix and Gauss-Jordan
        aug = [row[:] + [1.0 if r==c else 0.0 for c in range(n)] for r, row in enumerate(a)]
        for col in range(n):
            # Pivot
            piv = col
            for r in range(col+1, n):
                if abs(aug[r][col]) > abs(aug[piv][col]):
                    piv = r
            aug[col], aug[piv] = aug[piv], aug[col]
            d = aug[col][col]
            if d == 0:
                return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            for c in range(col, 2*n):
                aug[col][c] /= d
            for r in range(n):
                if r != col:
                    f = aug[r][col]
                    for c in range(col, 2*n):
                        aug[r][c] -= f * aug[col][c]
        flat = []
        for r in range(n):
            for c in range(n):
                flat.append(aug[r][n+c])
        return flat


def _mat_mul(a16: list, b16: list) -> list:
    """Multiply two 4×4 row-major matrices stored as 16 floats.

    Used only outside Blender (skeleton hierarchy resolution in parser.py).
    Inside Blender, mathutils.Matrix handles this directly.
    """
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += a16[r*4+k] * b16[k*4+c]
            out[r*4+c] = s
    return out


def _build_skeleton_frames(bones: list,
                           companion_hierarchy: dict | None = None) -> list:
    """Build a properly-nested Frame XNode tree from the flat bone list.

    companion_hierarchy, if provided, maps bone_name → parent_name (empty
    string means root-level) as read from the companion .x file.  When a
    bone appears in this dict its inferred parent is replaced by the
    explicit one from the .x, fixing cases where multiple bones share an
    identical world-bind transform (e.g. Chili's ROOT / Broth / Noodles
    all at the same position, causing eye bones to get the wrong parent).
    """
    if not bones:
        return []

    # A bone counts as a skeleton bone if it either ends in 'SHJnt'
    # (the standard Bugsnax joint naming convention) OR has skin
    # weights attached. The skin-weight case picks up "fixup" frames
    # like Olive's RootFix — it has 1136 verts weighted to it, so
    # even though the name doesn't follow the SHJnt convention, the
    # body geometry depends on it animating with the root. Excluding
    # it leaves the olive body sitting at world origin while the
    # rest of the model animates.
    is_skel = [
        (not _is_mesh_bone_name(b['name'])) or bool(b.get('skin'))
        for b in bones
    ]
    parents = _resolve_parent_indices(bones)

    # Override inferred parents with companion hierarchy where available.
    if companion_hierarchy:
        name_to_idx = {b['name']: i for i, b in enumerate(bones)}
        for i, b in enumerate(bones):
            bname = b['name']
            if bname not in companion_hierarchy:
                continue
            cx_parent_name = companion_hierarchy[bname]
            if not cx_parent_name:
                # Companion says root-level
                parents[i] = -1
                continue
            cx_parent_idx = name_to_idx.get(cx_parent_name, -1)
            if cx_parent_idx >= 0 and cx_parent_idx != i:
                parents[i] = cx_parent_idx

    # Use the FTM stored in the file directly — already parent-local
    # for nested bones, world-bind for top-level. For top-level bones
    # we substitute bind_pose since by rule 1 ftm == bind_pose anyway,
    # and for the root it disambiguates the fallback case.
    local_ftms: dict[int, list] = {}
    for i, b in enumerate(bones):
        if not is_skel[i]:
            continue
        if i == 0 or parents[i] < 0:
            local_ftms[i] = list(b['bind_pose'])
        else:
            local_ftms[i] = list(b['ftm'])

    # Build XNode Frame nodes (one per skeleton bone), then wire up children
    nodes: dict[int, XNode] = {}
    for i, b in enumerate(bones):
        if not is_skel[i]:
            continue
        nodes[i] = _make_frame_node(b['name'], local_ftms[i], [])

    # Top-level frames are collected as siblings (matches the dev .x
    # layout where most Bugsnax skeletons are flat at top level).
    root_nodes: list[XNode] = []
    for i, b in enumerate(bones):
        if not is_skel[i]:
            continue
        if i == 0 or parents[i] < 0:
            root_nodes.append(nodes[i])
        else:
            par_idx = parents[i]
            if par_idx in nodes:
                nodes[par_idx].children.append(nodes[i])
            elif root_nodes:
                # Orphaned (parent isn't a skeleton bone) — attach to
                # the first top-level frame to keep the tree connected.
                root_nodes[0].children.append(nodes[i])
            else:
                root_nodes.append(nodes[i])

    return root_nodes


def _build_mesh_frame_node(mesh: dict, bones: list) -> XNode:
    """Build a Frame + Mesh + MeshNormals + MeshTextureCoords + MeshMaterialList
    + XSkinMeshHeader + SkinWeights XNode tree from a parsed xcache mesh dict
    and the bone list. Returns (frame_node, list_of_top_level_materials)."""

    verts   = mesh['verts']
    normals = mesh['normals']
    uvs     = mesh['uvs']
    faces   = mesh['faces']
    name    = mesh['name']
    frame_name = name.replace('Geo', '') if name.endswith('Geo') else name

    # --- Mesh node ---
    mesh_node = XNode("Mesh", name)
    mesh_node.values.append(_num(len(verts)))
    for x, y, z in verts:
        mesh_node.values.extend([_num(x), _num(y), _num(z)])

    mesh_node.values.append(_num(len(faces)))
    for a, b, c in faces:
        mesh_node.values.extend([_num(3), _num(a), _num(b), _num(c)])

    # --- MeshNormals ---
    if normals:
        norm_node = XNode("MeshNormals", "")
        norm_node.values.append(_num(len(normals)))
        for nx, ny, nz in normals:
            norm_node.values.extend([_num(nx), _num(ny), _num(nz)])
        norm_node.values.append(_num(len(faces)))
        for idx, (a, b, c) in enumerate(faces):
            norm_node.values.extend([_num(3), _num(a), _num(b), _num(c)])
        mesh_node.children.append(norm_node)

    # --- MeshTextureCoords ---
    if uvs:
        uv_node = XNode("MeshTextureCoords", "")
        uv_node.values.append(_num(len(uvs)))
        for u, v in uvs:
            uv_node.values.extend([_num(u), _num(v)])
        mesh_node.children.append(uv_node)

    # --- Materials (one per sub-mesh) ---
    # Beffica's xcache packs two sub-meshes (body + limbs) with
    # different texture sets. The dev .x preserves this as two
    # Materials in the MeshMaterialList, with each face indexed to
    # the matching material. We mirror that structure here.
    per_submesh_tex = mesh.get('per_submesh_tex_paths', [mesh.get('tex_paths', [])])
    face_to_submesh = mesh.get('face_to_submesh', [0] * len(faces))
    n_submeshes = max(1, len(per_submesh_tex))

    def _make_material_node(sub_idx, tex_paths_for_sub):
        diffuse_path = tex_paths_for_sub[0] if tex_paths_for_sub else ""
        # Use a distinct name per sub-mesh so the dev-.x convention of
        # separate Body/Limbs materials is preserved. Single-mesh
        # files just get one "<frame_name>Material" entry.
        if n_submeshes > 1:
            this_mat_name = f"{frame_name}Material{sub_idx + 1}"
        else:
            this_mat_name = f"{frame_name}Material"
        m = XNode("Material", this_mat_name)
        for v in [1.0, 1.0, 1.0, 1.0]:        # RGBA face colour
            m.values.append(_num(v))
        # The .xcache binary format does NOT carry per-material shininess
        m.values.append(_num(1.0))             # power (low = matte)
        for v in [0.0, 0.0, 0.0]:              # specular (none)
            m.values.append(_num(v))
        for v in [0.0, 0.0, 0.0]:              # emissive
            m.values.append(_num(v))
        if diffuse_path:
            tex_node = XNode("TextureFileName", "")
            tex_node.values.append(_str(diffuse_path))
            m.children.append(tex_node)
        return m, this_mat_name

    # Build top-level Materials (one per sub-mesh) and inline copies
    submesh_mats = []   # list of (top_level_node, inline_node, mat_name)
    for sub_idx in range(n_submeshes):
        tex_paths_for_sub = per_submesh_tex[sub_idx] if sub_idx < len(per_submesh_tex) else []
        top_m, n_name = _make_material_node(sub_idx, tex_paths_for_sub)
        inline_m, _   = _make_material_node(sub_idx, tex_paths_for_sub)
        submesh_mats.append((top_m, inline_m, n_name))

    # First material is returned to the caller in the all_top_mats list
    # below; the singular "first material" is kept for backwards-compat
    # with single-mesh file importers. Additional sub-mesh materials
    # get attached inline to MeshMaterialList only.
    mat_name = submesh_mats[0][2]

    mat_list = XNode("MeshMaterialList", "")
    mat_list.values.append(_num(n_submeshes))       # nMaterials
    mat_list.values.append(_num(len(faces)))        # nFaceIndexes
    for fi in range(len(faces)):
        sub = face_to_submesh[fi] if fi < len(face_to_submesh) else 0
        mat_list.values.append(_num(sub))            # face -> material index
    # Inline copies of every sub-mesh material so importers that
    # prefer inline definitions still see them all.
    for top_m, inline_m, n_name in submesh_mats:
        mat_list.children.append(inline_m)
    # Plus a REF to the first one for top-level lookup compatibility
    ref_node = XNode("REF", "")
    ref_node.values.append((TOK_WORD, mat_name))
    mat_list.children.append(ref_node)
    mesh_node.children.append(mat_list)

    # --- XSkinMeshHeader + SkinWeights ---
    if bones:
        # Per-mesh skin data takes precedence when present (split mode):
        # use it to compute the influencing-bones count for the
        # XSkinMeshHeader so the count matches the SkinWeights blocks
        # we'll actually emit below. Without this, split-mode sub-mesh
        # 2 might emit an XSkinMeshHeader claiming count = chunk-0's
        # influencing bones instead of chunk-2's.
        per_mesh_skin_local = mesh.get('per_mesh_skin')
        if per_mesh_skin_local is not None:
            influencing_bones = [bones[bi] for bi, inf in per_mesh_skin_local if inf]
        else:
            # Count how many bones actually influence at least one vertex
            influencing_bones = [b for b in bones if b.get('skin', [])]
        n_inf = len(influencing_bones) if influencing_bones else 1

        # Per dev .x convention, ALL skel bones get a SkinWeights
        # block, even those with no influence (count=0). Beffica's
        # dev .x has 40 SkinWeights: 25 with data, 15 empty
        # placeholders for arm/leg/Hold bones. Emitting empties
        # round-trip-faithfully matches that structure.
        emit_empty_skinweights = True

        skin_header = XNode("XSkinMeshHeader", "")
        skin_header.values.extend([_num(n_inf), _num(n_inf), _num(n_inf)])
        mesh_node.children.append(skin_header)

        try:
            from mathutils import Matrix
            _has_mathutils = True
        except ImportError:
            _has_mathutils = False

        def _inv_ftm(ftm):
            if _has_mathutils:
                m = Matrix([ftm[0:4], ftm[4:8], ftm[8:12], ftm[12:16]]).transposed()
                try:
                    inv = m.inverted()
                except Exception:
                    inv = Matrix.Identity(4)
                inv_t = inv.transposed()
                return [inv_t[r][c] for r in range(4) for c in range(4)]
            else:
                return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]

        def _emit_sw_node(bone, influences):
            sw = XNode("SkinWeights", "")
            sw.values.append(_str(bone['name']))
            sw.values.append(_num(len(influences)))
            # Influences may be (vi, w) pairs (after the multi-chunk
            # shift loop processed them) or (vi, w, trailer) triples
            # (single-chunk files that skipped the shift loop). Strip
            # trailer for the .x emit.
            for entry in influences:
                sw.values.append(_num(entry[0]))
            for entry in influences:
                sw.values.append(_num(entry[1]))
            # Use the actual skinning offset matrix from the file
            # (read at ftm_offset + 296 in _parse_bones), which equals
            # the SkinWeights matrixOffset stored by the .x exporter
            # for the same character. Falls back to inv(bind_pose) if
            # for some reason skin_offset is missing or zero.
            skin_off = bone.get('skin_offset')
            if skin_off and any(abs(v) > 1e-9 for v in skin_off):
                for v in skin_off:
                    sw.values.append(_num(v))
            else:
                for v in _inv_ftm(bone['bind_pose']):
                    sw.values.append(_num(v))
            return sw

        # Check for per-mesh skin data (split-submeshes mode). When
        # set, `per_mesh_skin` is a list of (bone_idx, [(vi, w), ...])
        # for the bones that have influence in THIS sub-mesh, with
        # chunk-local vi values. We use it directly instead of walking
        # b['skin'] (which in split mode only holds chunk-0 data for
        # compatibility with animation extraction).
        per_mesh_skin = mesh.get('per_mesh_skin')
        if per_mesh_skin is not None:
            sw_bone_set = {bi for bi, _ in per_mesh_skin}
            sw_by_bone = {bi: inf for bi, inf in per_mesh_skin}
            for bi, b in enumerate(bones):
                if bi in sw_bone_set:
                    mesh_node.children.append(
                        _emit_sw_node(b, sw_by_bone[bi]))
                elif emit_empty_skinweights and not _is_mesh_bone_name(b['name']):
                    # Empty placeholder for skel joints that don't
                    # weight THIS sub-mesh — keeps the per-mesh
                    # SkinWeights count consistent across sub-meshes
                    # and matches the dev-.x convention of emitting
                    # all bones in every Mesh.
                    mesh_node.children.append(_emit_sw_node(b, []))
        elif influencing_bones:
            # Iterate ALL skel bones in their natural order, emitting
            # either the file's SkinWeights data or an empty
            # placeholder. The xcache mesh frame itself (names like
            # "skinnedGrumpus" or "CeleryGeo") is in the bone list
            # too — skip it for empty placeholders, but if it
            # somehow has skin data (rare), still emit. Fixup bones
            # like Olive's RootFix are non-SHJnt but DO have skin
            # data — those MUST be emitted.
            for b in bones:
                influences = b.get('skin', [])
                if influences:
                    # Always emit when there's data, regardless of name.
                    mesh_node.children.append(_emit_sw_node(b, influences))
                elif emit_empty_skinweights and not _is_mesh_bone_name(b['name']):
                    # Empty placeholder for unrigged skel joints (arm,
                    # leg, Hold bones in Beffica). Skip mesh-frame
                    # entries (the geo node, not a joint).
                    mesh_node.children.append(_emit_sw_node(b, []))
        else:
            # Fallback: bind all vertices 100% to root bone
            root_bone = bones[0]
            sw = XNode("SkinWeights", "")
            sw.values.append(_str(root_bone['name']))
            sw.values.append(_num(len(verts)))
            for i in range(len(verts)):
                sw.values.append(_num(i))
            for _ in range(len(verts)):
                sw.values.append(_num(1.0))
            skin_off = root_bone.get('skin_offset')
            if skin_off and any(abs(v) > 1e-9 for v in skin_off):
                for v in skin_off:
                    sw.values.append(_num(v))
            else:
                for v in _inv_ftm(root_bone['bind_pose']):
                    sw.values.append(_num(v))
            mesh_node.children.append(sw)

    # --- Frame wrapping the mesh ---
    frame_node = XNode("Frame", frame_name)
    frame_node.children.append(_make_ftm_node(mesh['transform']))
    frame_node.children.append(mesh_node)
    # Return list of all sub-mesh top-level materials. Single-mesh
    # files return a list of one for backwards compatibility.
    all_top_mats = [tm for tm, _im, _name in submesh_mats]
    return frame_node, all_top_mats


def _build_animation_set(bones: list, anim_data: dict[str, dict]) -> XNode:
    """Build an AnimationSet XNode from the extracted animation channels."""
    anim_set = XNode("AnimationSet", "anim")

    for bone in bones:
        name = bone['name']
        # Skip mesh entries — they don't carry animation tracks
        if _is_mesh_bone_name(name):
            continue
        channels = anim_data.get(name, {})
        pos_keys   = channels.get('pos',   {})
        scale_keys = channels.get('scale', {})
        rot_keys   = channels.get('rot',   {})

        if not pos_keys and not scale_keys and not rot_keys:
            continue

        anim_node = XNode("Animation", "")
        ref = XNode("REF")
        ref.values.append(_word(name))
        anim_node.children.append(ref)

        # Rotation keys (type 0): (qx,qy,qz,qw) → .x stores as (w,x,y,z)
        if rot_keys:
            rot_key = XNode("AnimationKey", "")
            rot_key.values.extend([_num(0), _num(len(rot_keys))])
            for tick in sorted(rot_keys):
                qx, qy, qz, qw = rot_keys[tick]
                rot_key.values.extend([_num(tick), _num(4),
                                        _num(qw), _num(qx), _num(qy), _num(qz)])
            anim_node.children.append(rot_key)
        else:
            # Synthesise identity rotation for every animated frame
            all_ticks = sorted(set(list(pos_keys.keys()) + list(scale_keys.keys())))
            if all_ticks:
                rot_key = XNode("AnimationKey", "")
                rot_key.values.extend([_num(0), _num(len(all_ticks))])
                for tick in all_ticks:
                    rot_key.values.extend([_num(tick), _num(4),
                                            _num(-1.0), _num(0.0), _num(0.0), _num(0.0)])
                anim_node.children.append(rot_key)

        # Scale keys (type 1)
        if scale_keys:
            sc_key = XNode("AnimationKey", "")
            sc_key.values.extend([_num(1), _num(len(scale_keys))])
            for tick, (sx, sy, sz) in sorted(scale_keys.items()):
                sc_key.values.extend([_num(tick), _num(3),
                                       _num(sx), _num(sy), _num(sz)])
            anim_node.children.append(sc_key)

        # Position keys (type 2)
        if pos_keys:
            pos_key = XNode("AnimationKey", "")
            pos_key.values.extend([_num(2), _num(len(pos_keys))])
            for tick, (px, py, pz) in sorted(pos_keys.items()):
                pos_key.values.extend([_num(tick), _num(3),
                                        _num(px), _num(py), _num(pz)])
            anim_node.children.append(pos_key)

        anim_set.children.append(anim_node)

    return anim_set


# Public entry point

def parse_xcache_file(filepath: str, split_submeshes: bool = False) -> XNode:
    """Parse a Horsepower Engine SEMS .xcache file and return an XNode tree.

    When `split_submeshes` is False (default), multi-mesh xcaches like
    Beffica (body + limbs) or Queen (11 chunks) have their sub-meshes
    concatenated into a single Mesh node. Skin weights are remapped to
    index into the combined vert array.

    When `split_submeshes` is True, each sub-mesh becomes its own Mesh
    XNode under the wrapping Frame, with its own per-chunk SkinWeights,
    MeshNormals, MeshTextureCoords, and MeshMaterialList. The Blender
    importer can then turn each Mesh into a separate object sharing the
    same armature, and round-trip export preserves the exact sub-mesh
    structure of the original file.
    """
    with open(filepath, 'rb') as fh:
        data = fh.read()

    if len(data) < 28:
        raise ValueError(f"Not a SEMS file: too short ({len(data)} bytes)")
    if data[:4] != b'SEMS':
        raise ValueError(f"Not a SEMS file: magic={data[:4]!r}")

    # Filename stem (used later for mesh-name fallback when the file
    # contains no ASCII mesh name near the vertex data — e.g.
    # IceCream.xcache, Apple.xcache, ~88% of the dev xcache corpus).
    _xcache_stem = os.path.splitext(os.path.basename(filepath))[0]

    # --- Header ---
    bone_count = _u32(data, 0x1C)

    # --- Parse bones ---
    bones, after_bones = _parse_bones(data, bone_count, 0x20)

    # --- Extract animation data and skin weights per bone ---
    anim_data: dict[str, dict] = {}
    for i, bone in enumerate(bones):
        if i + 1 < len(bones):
            next_bone_hdr = bones[i + 1]['data_start'] - 64 - len(bones[i + 1]['name'])
        else:
            # Last bone: there's no following bone header, so
            # `after_bones` equals this bone's data_start (zero-sized
            # region) and the rotation / skin scanners would have
            # nothing to scan. Use a generous lookahead instead — the
            # rotation scanner terminates on non-quat data and the
            # skin reader rejects bogus counts, so an over-large
            # window is safe.
            next_bone_hdr = min(bone['data_start'] + 200_000, len(data))
        # Per-bone failures should not abort the entire parse — an
        # animation channel that can't be decoded is preferable to a
        # zero-bones import. The empty-channels fallback preserves the
        # bone in the output but with no animation data attached.
        try:
            channels = _extract_anim(data, bone, next_bone_hdr)
        except (struct.error, ValueError, IndexError):
            channels = {'pos': {}, 'scale': {}, 'rot': {},
                        'skin': [], 'skin_pad': 0, 'skin_resets': []}
        bone['skin']        = channels.get('skin', [])
        bone['skin_pad']    = channels.get('skin_pad', 0)
        bone['skin_resets'] = channels.get('skin_resets', [])
        if channels['pos'] or channels['scale'] or channels['rot']:
            anim_data[bone['name']] = channels

    # --- Parse meshes ---
    # Mesh section parsing is best-effort: a corrupt mesh region should
    # not abort the entire import (skeleton + animation may still be
    # useful even without geometry).
    try:
        meshes = _parse_mesh_section(data, after_bones)
    except (struct.error, ValueError, IndexError):
        meshes = []

    # Direct-scan fallback uses placeholder names ("Mesh", "Mesh_2"...)
    # when no ASCII name was found near the vertex data (e.g. IceCream
    # has only a 3-char bone "Neo", no mesh name). Replace those with
    # filename-derived names so the Blender import gets a useful label.
    if meshes:
        for i, m in enumerate(meshes):
            if m.get('name') in ('Mesh', None, ''):
                m['name'] = _xcache_stem if i == 0 else f"{_xcache_stem}_{i+1}"
            elif m.get('name', '').startswith('Mesh_'):
                # Placeholder like 'Mesh_2' from multi-mesh fallback
                # path; replace with stem-suffixed variant.
                suffix = m['name'][5:]   # strip 'Mesh_'
                m['name'] = f"{_xcache_stem}_{suffix}"

    # Default per-mesh metadata for single-mesh files (single-mesh
    # files skip the multi-mesh merge below). For Beffica/Honey-class
    # files these get overwritten by the merge.
    for m in meshes:
        if 'face_to_submesh' not in m:
            m['face_to_submesh'] = [0] * len(m['faces'])
        if 'per_submesh_tex_paths' not in m:
            m['per_submesh_tex_paths'] = [list(m.get('tex_paths', []))]

    # If we found a continuation mesh (e.g. Honey's goop overlay), merge
    # it into the primary mesh. pad=1 SkinWeights reference the
    # continuation's verts using LOW indices [0, len(mesh2.verts)) —
    # after merging, those indices need to be offset by len(mesh1.verts)
    # so they index into the combined vert array. pad=0 SkinWeights
    # keep their original indices (they reference mesh1).
    #
    # When `split_submeshes` is True we instead keep each sub-mesh as
    # its own dict and distribute each bone's skin weights to whichever
    # sub-mesh they belong to (by chunk-index). The output XNode tree
    # then contains one Frame per sub-mesh, each holding its own Mesh.
    if len(meshes) > 1 and not split_submeshes:
        primary = meshes[0]
        n_primary_verts = len(primary['verts'])
        # Concatenate verts/normals/uvs from continuation meshes.
        # Faces from continuation meshes get their indices shifted by
        # the cumulative vert count.
        merged_faces = list(primary['faces'])
        cumulative = n_primary_verts
        offsets_per_mesh = [0]  # mesh i's verts start at offsets_per_mesh[i]
        # Per-face sub-mesh index, so each segment can keep its own
        # material on export. Beffica has two sub-meshes — body and
        # limbs — with different textures. Without this, the export
        # produces one merged material per segment AND loses the
        # body-vs-limbs distinction from the original Maya rig.
        face_to_submesh = [0] * len(primary['faces'])
        # Per-sub-mesh texture-path lists. Indexed by submesh idx.
        per_submesh_tex_paths = [list(primary['tex_paths'])]
        for sub_idx, cont in enumerate(meshes[1:], start=1):
            offsets_per_mesh.append(cumulative)
            primary['verts'].extend(cont['verts'])
            primary['normals'].extend(cont['normals'])
            primary['uvs'].extend(cont['uvs'])
            for a, b, c in cont['faces']:
                merged_faces.append((a + cumulative, b + cumulative, c + cumulative))
                face_to_submesh.append(sub_idx)
            # Keep this sub-mesh's textures separate from primary's
            per_submesh_tex_paths.append(list(cont['tex_paths']))
            # Also append to primary's tex_paths for backwards-compat
            # callers that look at the flat list (unchanged behaviour).
            for tp in cont['tex_paths']:
                if tp not in primary['tex_paths']:
                    primary['tex_paths'].append(tp)
            cumulative += len(cont['verts'])
        primary['faces'] = merged_faces
        primary['face_to_submesh'] = face_to_submesh
        primary['per_submesh_tex_paths'] = per_submesh_tex_paths
        meshes = [primary]

        # Remap SkinWeights vi to point into the merged vert array.
        #
        # Each bone's skin block has a "pad"/"marker" value identifying
        # which mesh chunk it deforms; vi values are RELATIVE to that
        # chunk's start. Universal remap:
        #     shifted_vi = raw_vi + offsets_per_mesh[marker]
        #
        # Validated against dev .x for Queen (markers 0-11), Honey
        # (marker=1 wing bones), and 2-chunk files (Beffica/Floofty/
        # Wambus). Within-bone resets typically mark "layered override"
        # re-weights of the same chunk; we dedupe last-write-wins.
        total_verts = cumulative
        n_chunks = len(offsets_per_mesh)

        # Build a vert lookup for the side-aware filter (below).
        merged_verts = meshes[0]['verts']

        # Pre-compute bone bind-space positions (used by side filter).
        # skin_offset is the inverse-bind transform; inverting it gives
        # the bind-space position of the bone.
        bone_bind_pos = {}
        try:
            import numpy as _np
            _has_numpy = True
        except ImportError:
            _has_numpy = False
        for b in bones:
            so = b.get('skin_offset')
            if not so:
                continue
            try:
                if _has_numpy:
                    M = _np.array(so).reshape(4, 4)
                    Minv = _np.linalg.inv(M)
                    bone_bind_pos[b['name']] = (Minv[3, 0], Minv[3, 1], Minv[3, 2])
                else:
                    inv = _mat_inv(so)
                    bone_bind_pos[b['name']] = (inv[12], inv[13], inv[14])
            except Exception:
                pass

        for b in bones:
            skin = b.get('skin')
            if not skin:
                continue
            pad = b.get('skin_pad', 0)
            resets = b.get('skin_resets') or []

            # Each skin entry carries its own chunk index in the
            # trailer field; a single bone's block can span multiple
            # chunks (Queen.xcache ROOT/MidTier/TopTier weight chunks
            # 0, 1, and 6 simultaneously). Legacy entries that are
            # (vi, w) tuples are treated as trailer=pad.

            # Side-aware filter setup.
            bp_xyz = bone_bind_pos.get(b['name'])
            bone_x = bp_xyz[0] if bp_xyz is not None else 0.0
            side_filter_active = abs(bone_x) > 1.0
            SIDE_TOLERANCE = 5.0

            def _passes_side(new_vi: int) -> bool:
                if not side_filter_active:
                    return True
                if new_vi >= len(merged_verts):
                    return True
                vx = merged_verts[new_vi][0]
                if bone_x > 0:
                    return vx >= -SIDE_TOLERANCE
                else:
                    return vx <= SIDE_TOLERANCE

            def _entry_chunk_offset(entry_trailer: int) -> int:
                """Look up chunk offset for an entry's trailer.

                The trailer is the chunk index. Unknown trailers fall
                back to chunk 0.
                """
                if 0 <= entry_trailer < n_chunks:
                    return offsets_per_mesh[entry_trailer]
                return 0

            # Apply shift (with dedupe via last-write-wins for resets and
            # for duplicate (chunk, vi) tuples).
            merged: dict = {}

            def _process_entry(entry):
                if len(entry) == 3:
                    vi, w, trailer = entry
                else:
                    # Legacy (vi, w) shape — use bone's pad
                    vi, w = entry
                    trailer = pad
                chunk_offset = _entry_chunk_offset(trailer)
                new_vi = vi + chunk_offset
                if new_vi < total_verts and _passes_side(new_vi):
                    merged[new_vi] = w

            if resets:
                segs = [0] + resets + [len(skin)]
                for j in range(len(segs) - 1):
                    for entry in skin[segs[j]:segs[j+1]]:
                        _process_entry(entry)
            else:
                for entry in skin:
                    _process_entry(entry)
            b['skin'] = list(merged.items())

    # ---- Split-aware path: distribute skin weights per sub-mesh ----
    #
    # In split mode, each sub-mesh becomes its own Mesh XNode under
    # its own wrapping Frame. Skin weights need to be distributed
    # per-chunk so each sub-mesh's SkinWeights only references its
    # own verts (with chunk-LOCAL indices, not offset into a combined
    # array).
    #
    # We stash `per_mesh_skin` on each mesh dict as
    #     [(bone_idx, [(vi, w), ...]), ...]
    # so _build_mesh_frame_node can emit SkinWeights blocks specific
    # to that sub-mesh. The bones list still gets its skin attribute
    # set (using chunk-0 weights so the merge path is preserved for
    # single-mesh files, and so animation code paths that read
    # b['skin'] still see something reasonable for multi-mesh files
    # in split mode).
    elif len(meshes) > 1 and split_submeshes:
        n_chunks = len(meshes)

        # Initialize per-mesh skin dicts. Outer dict keyed by bone
        # index; inner dict keyed by chunk-local vi → weight (so
        # duplicate-vi resets last-write-wins, matching the merge
        # path's `merged` dict semantics).
        per_mesh_bone_weights: list[dict[int, dict[int, float]]] = [
            {} for _ in range(n_chunks)
        ]

        for bi, b in enumerate(bones):
            skin = b.get('skin')
            if not skin:
                continue
            pad = b.get('skin_pad', 0)
            resets = b.get('skin_resets') or []

            def _process_entry_split(entry, _bi=bi):
                if len(entry) == 3:
                    vi, w, trailer = entry
                else:
                    vi, w = entry
                    trailer = pad
                # Out-of-range trailer falls back to chunk 0
                if not (0 <= trailer < n_chunks):
                    trailer = 0
                # Reject vi values that exceed this chunk's vert count
                chunk_vc = len(meshes[trailer]['verts'])
                if vi >= chunk_vc:
                    return
                bdict = per_mesh_bone_weights[trailer].setdefault(_bi, {})
                bdict[vi] = w

            if resets:
                segs = [0] + resets + [len(skin)]
                for j in range(len(segs) - 1):
                    for entry in skin[segs[j]:segs[j+1]]:
                        _process_entry_split(entry)
            else:
                for entry in skin:
                    _process_entry_split(entry)

        # Stash the per-mesh weights on each mesh dict, and prepare
        # the bones for downstream code (animation extraction expects
        # b['skin'] to be (vi, w) tuples; use chunk-0 weights for
        # compatibility — animation doesn't actually consume vi/w,
        # it just needs the entry count for skin_pad detection).
        for mi, mesh in enumerate(meshes):
            # Build list-of-tuples for this sub-mesh, indexed by bone
            #   [(bone_idx, [(vi, w), ...]), ...]
            mesh_sw = []
            for bi, vi_w_dict in per_mesh_bone_weights[mi].items():
                influences = list(vi_w_dict.items())
                if influences:
                    mesh_sw.append((bi, influences))
            mesh['per_mesh_skin'] = mesh_sw

        # For compatibility with the merge path's b['skin'] shape,
        # synthesize a flat skin list per bone (using chunk-0 weights
        # if present, else the first chunk that has any). This isn't
        # read by SkinWeights emission in split mode (which uses
        # per_mesh_skin), but is kept for animation-extraction
        # routines that probe b['skin'] length / shape.
        for bi, b in enumerate(bones):
            for mi in range(n_chunks):
                influences = list(per_mesh_bone_weights[mi].get(bi, {}).items())
                if influences:
                    b['skin'] = influences
                    break

    # --- Build XNode tree ---
    root = XNode("ROOT", "")

    # AnimTicksPerSecond
    tps_node = XNode("AnimTicksPerSecond", "")
    tps_node.values.append(_num(30))
    root.children.append(tps_node)

    # Build mesh frames first so we can collect their top-level Materials and
    mesh_frame_nodes = []
    top_level_mats = []
    for mesh in meshes:
        frame_n, mesh_top_mats = _build_mesh_frame_node(mesh, bones)
        mesh_frame_nodes.append(frame_n)
        if mesh_top_mats:
            # _build_mesh_frame_node now returns a list of top-level
            # mats (one per sub-mesh). Extend, don't append, to keep
            # the flat list.
            top_level_mats.extend(mesh_top_mats)

    # Top-level Materials (matching .x file convention)
    for tlm in top_level_mats:
        root.children.append(tlm)

    # If a companion .x file exists (same stem, same dir), use its
    # Frame nesting as the authoritative parent hierarchy. Needed when
    # several bones share an identical world-bind transform — Chili's
    # ROOT/Broth_Aux/Noodles_Aux all sit at the same position, so the
    # pupil/lid bones can legally claim any as parent and geometric
    # inference picks the wrong one. Bones absent from the companion
    # (e.g. LOD/helper bones added in the xcache) keep the inferred
    # result.
    _companion_hierarchy: dict[str, str] = {}   # bone_name -> parent_name
    _xcache_dir  = os.path.dirname(os.path.abspath(filepath))
    for _ext in ('.x', '.X'):
        _cx_path = os.path.join(_xcache_dir, _xcache_stem + _ext)
        if os.path.exists(_cx_path):
            try:
                _cx_root = parse_x_file(_cx_path)
                def _collect_hierarchy(node, parent_name=None):
                    if node.kind == 'Frame':
                        if node.name:
                            _companion_hierarchy[node.name] = parent_name or ''
                        for _c in node.children:
                            _collect_hierarchy(_c, node.name)
                    else:
                        for _c in node.children:
                            _collect_hierarchy(_c, parent_name)
                _collect_hierarchy(_cx_root)
            except Exception:
                _companion_hierarchy = {}  # companion parse failure → keep inferred
            break

    # Skeleton frames (hierarchy resolution happens inside here)
    skel_frames = _build_skeleton_frames(bones, _companion_hierarchy)
    root.children.extend(skel_frames)

    # Mesh frames
    for fn in mesh_frame_nodes:
        root.children.append(fn)

    # AnimationSet
    if anim_data:
        root.children.append(_build_animation_set(bones, anim_data))

    return root

# WRITER (export side)

# ---------- Header ----------

def _split_x_mesh_by_material(root: 'XNode') -> None:
    """Post-process: split multi-material Mesh nodes in a parsed .x tree
    into multiple single-material sibling Mesh nodes.

    Walks the tree, and for each Mesh node whose MeshMaterialList assigns
    DIFFERENT materials to different faces, replaces that Mesh node with
    N siblings — one per material — sharing the same wrapping Frame.

    Subsets all per-vertex / per-face / per-loop data:
      - Mesh values: vert positions + face index lists
      - MeshNormals: normals + face-normal-index lists
      - MeshTextureCoords: per-vertex UVs
      - MeshMaterialList: kept with just the chosen material (single entry)
      - SkinWeights: vi values filtered + remapped to the new vert indices

    Single-material meshes (the common case for Bugsnax characters: Olive,
    Watermelon, Apple, Beffica, etc.) are left UNCHANGED.

    Mutates the tree in place.
    """
    def _walk(node: 'XNode'):
        # Recurse first so children-of-children get split too, before we
        # mutate the parent's children list at this level.
        for child in list(node.children):
            _walk(child)

        # Find all Mesh children of *this* node that need splitting and
        # rebuild the children list in one pass.
        new_children = []
        for child in node.children:
            if child.kind != 'Mesh':
                new_children.append(child)
                continue
            split = _split_one_mesh(child)
            if split is None:
                new_children.append(child)
            else:
                new_children.extend(split)
        node.children = new_children

    _walk(root)


def _split_one_mesh(mesh: 'XNode') -> 'Optional[list]':
    """Returns a list of replacement Mesh XNodes if this mesh has >1
    distinct material assignment; otherwise None (caller keeps it as-is).
    """
    ml = mesh.child('MeshMaterialList')
    if ml is None:
        return None
    ml_nums = ml.nums()
    if len(ml_nums) < 2:
        return None
    n_mats       = int(ml_nums[0])
    n_face_mats  = int(ml_nums[1])
    if n_mats < 2 or n_face_mats < 1:
        return None
    face_mat_idx = [int(x) for x in ml_nums[2:2 + n_face_mats]]
    distinct_mats = sorted(set(face_mat_idx))
    if len(distinct_mats) < 2:
        # Only 1 material actually used (per requested behavior — even if
        # MML declares 2 materials but all faces use mat 0, don't split).
        return None

    # Parse mesh values into structured form. Mesh layout:
    #   n_verts; vert[0..n_verts-1] (3 floats each);
    #   n_faces; for each face: nv_in_face, vi[0..nv-1]
    mesh_nums = mesh.nums()
    if not mesh_nums:
        return None
    n_verts = int(mesh_nums[0])
    if 1 + 3 * n_verts >= len(mesh_nums):
        return None
    verts = []
    for i in range(n_verts):
        ox = 1 + 3 * i
        verts.append((mesh_nums[ox], mesh_nums[ox + 1], mesh_nums[ox + 2]))
    pos = 1 + 3 * n_verts
    n_faces = int(mesh_nums[pos]); pos += 1
    if n_face_mats != n_faces:
        # Sanity check failed — bail out and leave mesh unchanged.
        return None
    faces = []   # list of tuples of vert indices (variable length)
    for fi in range(n_faces):
        nv_in_face = int(mesh_nums[pos]); pos += 1
        f = tuple(int(mesh_nums[pos + k]) for k in range(nv_in_face))
        pos += nv_in_face
        faces.append(f)

    # Parse MeshNormals (optional). Same face count, but its own
    # normal-index buffer (may differ from mesh's vert indices since
    # normals are per-loop).
    norm_node = mesh.child('MeshNormals')
    normals = None
    norm_faces = None
    if norm_node is not None:
        nn_nums = norm_node.nums()
        if nn_nums:
            n_norms = int(nn_nums[0])
            if 1 + 3 * n_norms < len(nn_nums):
                normals = []
                for i in range(n_norms):
                    ox = 1 + 3 * i
                    normals.append((nn_nums[ox], nn_nums[ox + 1], nn_nums[ox + 2]))
                npos = 1 + 3 * n_norms
                if npos < len(nn_nums):
                    n_norm_faces = int(nn_nums[npos]); npos += 1
                    if n_norm_faces == n_faces:
                        norm_faces = []
                        for fi in range(n_norm_faces):
                            nv = int(nn_nums[npos]); npos += 1
                            nf = tuple(int(nn_nums[npos + k]) for k in range(nv))
                            npos += nv
                            norm_faces.append(nf)
                    else:
                        # Face count mismatch — don't try to split normals
                        normals = None
                        norm_faces = None

    # Parse MeshTextureCoords (optional). One UV per vertex.
    uv_node = mesh.child('MeshTextureCoords')
    uvs = None
    if uv_node is not None:
        uvn = uv_node.nums()
        if uvn:
            n_uv = int(uvn[0])
            if 1 + 2 * n_uv <= len(uvn) and n_uv == n_verts:
                uvs = [(uvn[1 + 2 * i], uvn[1 + 2 * i + 1]) for i in range(n_uv)]

    # Collect Material/REF children of MeshMaterialList (in order). The
    # MML's children are the material list — index N in face_mat_idx
    # picks the Nth Material/REF child.
    ml_mat_children = []
    for c in ml.children:
        if c.kind in ('Material', 'REF'):
            ml_mat_children.append(c)
    if not ml_mat_children:
        return None

    # Collect SkinWeights nodes (kept as-is; we filter their entries
    # per-output mesh below).
    sw_nodes = [c for c in mesh.children if c.kind == 'SkinWeights']

    # Build a sub-mesh per material index that is actually used.
    output_meshes: 'list[XNode]' = []
    base_name = mesh.name
    for ord_idx, mat_idx in enumerate(distinct_mats):
        # Which faces belong to this material?
        face_mask = [face_mat_idx[fi] == mat_idx for fi in range(n_faces)]

        # Collect used vert indices in order of first appearance, build
        # remap old_vi → new_vi.
        old_to_new: 'dict[int, int]' = {}
        new_verts: list = []
        for fi, kept in enumerate(face_mask):
            if not kept:
                continue
            for vi in faces[fi]:
                if vi not in old_to_new:
                    old_to_new[vi] = len(new_verts)
                    new_verts.append(verts[vi])

        if not new_verts:
            continue

        # Build new face list with remapped indices.
        new_faces = [
            tuple(old_to_new[vi] for vi in faces[fi])
            for fi, kept in enumerate(face_mask) if kept
        ]

        # Subset UVs: indexed by old vi, kept entries at new vi position.
        new_uvs = None
        if uvs is not None:
            new_uvs = [None] * len(new_verts)
            for old_vi, new_vi in old_to_new.items():
                if old_vi < len(uvs):
                    new_uvs[new_vi] = uvs[old_vi]
            # Fill any holes (shouldn't happen but defensive)
            new_uvs = [uv if uv is not None else (0.0, 0.0) for uv in new_uvs]

        # Subset normals: normals are independent of vert indices —
        # they use their own per-face index buffer. So we keep ALL
        # normals (or only those referenced by kept faces) and remap
        # the face-normal-index buffer accordingly.
        new_normals = None
        new_norm_faces = None
        if normals is not None and norm_faces is not None:
            # Collect used normal indices in order of first appearance
            n_old_to_new: 'dict[int, int]' = {}
            new_normals = []
            for fi, kept in enumerate(face_mask):
                if not kept:
                    continue
                for ni in norm_faces[fi]:
                    if ni not in n_old_to_new:
                        n_old_to_new[ni] = len(new_normals)
                        new_normals.append(normals[ni])
            new_norm_faces = [
                tuple(n_old_to_new[ni] for ni in norm_faces[fi])
                for fi, kept in enumerate(face_mask) if kept
            ]

        # Build new Mesh XNode.
        sub_name = f"{base_name}_{ord_idx + 1}" if ord_idx > 0 else base_name
        sub_mesh = XNode("Mesh", sub_name)
        # Tag with split-group metadata so the importer can stash
        # provenance on the resulting Blender objects, enabling the
        # exporter to merge them back into a single multi-material
        # Mesh node on round-trip.
        sub_mesh.meta = {
            'split_source_mesh': base_name,
            'split_group_idx': ord_idx,
            'split_group_total': len(distinct_mats),
            'split_source_mat_idx': mat_idx,
        }
        sub_mesh.values.append(_num(len(new_verts)))
        for x, y, z in new_verts:
            sub_mesh.values.extend([_num(x), _num(y), _num(z)])
        sub_mesh.values.append(_num(len(new_faces)))
        for f in new_faces:
            sub_mesh.values.append(_num(len(f)))
            for vi in f:
                sub_mesh.values.append(_num(vi))

        # New MeshNormals.
        if new_normals is not None:
            nn = XNode("MeshNormals", "")
            nn.values.append(_num(len(new_normals)))
            for nx, ny, nz in new_normals:
                nn.values.extend([_num(nx), _num(ny), _num(nz)])
            nn.values.append(_num(len(new_norm_faces)))
            for f in new_norm_faces:
                nn.values.append(_num(len(f)))
                for ni in f:
                    nn.values.append(_num(ni))
            sub_mesh.children.append(nn)

        # New MeshTextureCoords.
        if new_uvs is not None:
            uvc = XNode("MeshTextureCoords", "")
            uvc.values.append(_num(len(new_uvs)))
            for u, v in new_uvs:
                uvc.values.extend([_num(u), _num(v)])
            sub_mesh.children.append(uvc)

        # New MeshMaterialList: single material, all faces -> 0.
        new_ml = XNode("MeshMaterialList", "")
        new_ml.values.append(_num(1))                  # nMaterials
        new_ml.values.append(_num(len(new_faces)))     # nFaceIndexes
        for _ in new_faces:
            new_ml.values.append(_num(0))              # all map to material 0
        # Attach the chosen Material/REF (preserve the original child node)
        if mat_idx < len(ml_mat_children):
            new_ml.children.append(ml_mat_children[mat_idx])
        sub_mesh.children.append(new_ml)

        # SkinWeights — for each bone's block, filter entries whose vi
        # is in old_to_new and remap. Keep all other Mesh children
        # unchanged (XSkinMeshHeader, DeclData, etc., live on the
        # PARENT scope and apply to the whole skeleton — they don't
        # need splitting). Actually they DO live as children of Mesh
        # so we copy them as-is, with SkinWeights filtered.
        for child in mesh.children:
            if child.kind in ('MeshNormals', 'MeshTextureCoords',
                              'MeshMaterialList'):
                continue   # already replaced above
            if child.kind == 'SkinWeights':
                sw_nums = child.nums()
                sw_strs = child.strings()
                if not sw_nums or not sw_strs:
                    sub_mesh.children.append(child)
                    continue
                n_inf = int(sw_nums[0])
                # sw_nums: [n_inf, vi*n_inf, w*n_inf, matrix[16]]
                # _Some_ files store n_inf=0 as an empty placeholder.
                # Keep those without trying to subset.
                if n_inf == 0:
                    sub_mesh.children.append(child)
                    continue
                if 1 + 2 * n_inf + 16 > len(sw_nums):
                    sub_mesh.children.append(child)
                    continue
                vis = [int(sw_nums[1 + i]) for i in range(n_inf)]
                ws  = [float(sw_nums[1 + n_inf + i]) for i in range(n_inf)]
                matrix = sw_nums[1 + 2 * n_inf:1 + 2 * n_inf + 16]
                # Filter to entries whose old vi is used by THIS sub-mesh
                kept = [(old_to_new[v], w)
                        for v, w in zip(vis, ws)
                        if v in old_to_new]
                new_sw = XNode("SkinWeights", "")
                new_sw.values.append((TOK_STR, sw_strs[0]))
                new_sw.values.append(_num(len(kept)))
                for nv, _w in kept:
                    new_sw.values.append(_num(nv))
                for _v, w in kept:
                    new_sw.values.append(_num(w))
                for m in matrix:
                    new_sw.values.append(_num(m))
                sub_mesh.children.append(new_sw)
                continue
            # XSkinMeshHeader, DeclData, and any other child kinds:
            # copy as-is. These describe the skeletal envelope or
            # vertex-decl format which apply equally to each sub-mesh.
            sub_mesh.children.append(child)

        output_meshes.append(sub_mesh)

    if not output_meshes:
        return None
    return output_meshes


def _triangulate_mesh_quads(root: 'XNode') -> None:
    """Post-process: convert quad (and n-gon) faces in every Mesh node
    of the parsed tree into pairs of triangles.

    Bugsnax xcache files are already all-triangles, but .x text files
    often store the original Maya quad topology (e.g. BalloonLow.x is
    mostly quads). This converts everything to triangles for downstream
    consistency.

    Triangulation strategy: for a face (v0, v1, ..., vN), emit a fan
    starting from v0: (v0, v1, v2), (v0, v2, v3), ..., (v0, vN-1, vN).
    This is the simple convex-fan split — fine for quads from Maya
    exports (which are guaranteed convex by Maya's authoring).

    Updates:
      - Mesh values: replaces face index list with all-tris version
      - MeshNormals: replaces face-normal-index list (same fan split)
      - MeshMaterialList: replaces per-face material-index list
        (duplicates the source face's material index into both tri pieces)

    SkinWeights and MeshTextureCoords are per-vertex (not per-face),
    so they don't need updating.

    Mutates the tree in place. Idempotent — re-running on an all-tri
    tree leaves it unchanged.
    """
    def _walk(node: 'XNode'):
        for child in node.children:
            if child.kind == 'Mesh':
                _triangulate_one_mesh(child)
            _walk(child)
    _walk(root)


def _triangulate_one_mesh(mesh: 'XNode') -> None:
    """Triangulate the faces of a single Mesh XNode in place."""
    mesh_nums = mesh.nums()
    if not mesh_nums:
        return
    n_verts = int(mesh_nums[0])
    if 1 + 3 * n_verts >= len(mesh_nums):
        return

    # Extract verts as (x,y,z) tuples.
    verts = []
    for i in range(n_verts):
        ox = 1 + 3 * i
        verts.append((mesh_nums[ox], mesh_nums[ox + 1], mesh_nums[ox + 2]))

    # Parse face list.
    pos = 1 + 3 * n_verts
    n_faces = int(mesh_nums[pos]); pos += 1
    faces = []
    for fi in range(n_faces):
        if pos >= len(mesh_nums):
            return
        nv_in_face = int(mesh_nums[pos]); pos += 1
        f = tuple(int(mesh_nums[pos + k]) for k in range(nv_in_face))
        pos += nv_in_face
        faces.append(f)

    # Early exit: nothing to do if every face is already a triangle.
    if all(len(f) == 3 for f in faces):
        return

    # Triangulate: fan from v0. Track the source face index for each
    # output tri so we can fan-replicate MeshNormals and the per-face
    # MeshMaterialList entries the same way.
    new_faces: list = []
    new_face_src: list = []   # source face index for each output tri
    for src_fi, f in enumerate(faces):
        if len(f) < 3:
            # Degenerate — keep as-is (downstream filter drops it)
            new_faces.append(f)
            new_face_src.append(src_fi)
            continue
        if len(f) == 3:
            new_faces.append(f)
            new_face_src.append(src_fi)
            continue
        v0 = f[0]
        for k in range(1, len(f) - 1):
            new_faces.append((v0, f[k], f[k + 1]))
            new_face_src.append(src_fi)

    # Rebuild Mesh.values from scratch. The text parser stores values
    # with interleaved ';' and ',' separator tokens, but the xcache
    # emission path (and downstream code via .nums()) ignores them.
    # We re-emit using just NUM tokens, matching the xcache convention.
    new_values = []
    new_values.append(_num(n_verts))
    for x, y, z in verts:
        new_values.extend([_num(x), _num(y), _num(z)])
    new_values.append(_num(len(new_faces)))
    for f in new_faces:
        new_values.append(_num(3))
        for vi in f:
            new_values.append(_num(vi))
    mesh.values = new_values

    # MeshNormals: same fan split.
    mn = mesh.child('MeshNormals')
    if mn is not None:
        mn_nums = mn.nums()
        if mn_nums:
            n_norms = int(mn_nums[0])
            if 1 + 3 * n_norms < len(mn_nums):
                # Extract normals
                normals = []
                for i in range(n_norms):
                    ox = 1 + 3 * i
                    normals.append((mn_nums[ox], mn_nums[ox + 1], mn_nums[ox + 2]))
                # Parse normal face list
                npos = 1 + 3 * n_norms
                n_norm_faces = int(mn_nums[npos]); npos += 1
                if n_norm_faces == n_faces:
                    norm_faces: list = []
                    valid = True
                    for fi in range(n_norm_faces):
                        if npos >= len(mn_nums):
                            valid = False
                            break
                        nv = int(mn_nums[npos]); npos += 1
                        nf = tuple(int(mn_nums[npos + k]) for k in range(nv))
                        npos += nv
                        norm_faces.append(nf)

                    if valid:
                        # Apply same fan split to normal faces.
                        new_norm_faces: list = []
                        for src_fi, nf in enumerate(norm_faces):
                            if len(nf) < 3:
                                new_norm_faces.append(nf)
                                continue
                            if len(nf) == 3:
                                new_norm_faces.append(nf)
                                continue
                            n0 = nf[0]
                            for k in range(1, len(nf) - 1):
                                new_norm_faces.append((n0, nf[k], nf[k + 1]))

                        # Rebuild MeshNormals.values.
                        new_mn = []
                        new_mn.append(_num(n_norms))
                        for nx, ny, nz in normals:
                            new_mn.extend([_num(nx), _num(ny), _num(nz)])
                        new_mn.append(_num(len(new_norm_faces)))
                        for f in new_norm_faces:
                            new_mn.append(_num(3))
                            for ni in f:
                                new_mn.append(_num(ni))
                        mn.values = new_mn

    # MeshMaterialList: per-face material indices.
    ml = mesh.child('MeshMaterialList')
    if ml is not None:
        ml_nums = ml.nums()
        if len(ml_nums) >= 2:
            n_mats = int(ml_nums[0])
            n_face_mats = int(ml_nums[1])
            if n_face_mats == n_faces:
                face_mat_idx = [int(x) for x in ml_nums[2:2 + n_face_mats]]
                # Map each new tri to its source face's material index.
                new_face_mat = [face_mat_idx[src_fi] for src_fi in new_face_src]
                # Rebuild MeshMaterialList.values. Material child blocks
                # live in ml.children (not values), so we only need to
                # rewrite the header/face-index portion.
                new_ml_values = []
                new_ml_values.append(_num(n_mats))
                new_ml_values.append(_num(len(new_face_mat)))
                for mi in new_face_mat:
                    new_ml_values.append(_num(mi))
                ml.values = new_ml_values



def parse_x_file(filepath: str, split_submeshes: bool = False,
                 triangulate_quads: bool = False) -> XNode:
    """Parse a .x or .xcache file and return its XNode tree.

    Auto-detects format by magic bytes:
      * "xof " (offset 0)            → DirectX .x (text/binary/compressed)
      * "SEMS" (offset 0)            → Bugsnax .xcache binary

    Falls back to extension matching for files with non-standard magic.

    `split_submeshes` applies to both:
      - xcache files: split internal multi-mesh structure into separate
        Mesh nodes (see parse_xcache_file).
      - .x files: when a Mesh has a MeshMaterialList assigning DIFFERENT
        materials to different faces (e.g. BalloonLow.x has BoatTrimSheet
        on 12,525 faces and blinn2 on 736), split it into N sibling Mesh
        nodes — one per distinct material. Single-material Mesh nodes
        (Olive, Watermelon, Apple, Beffica, etc.) are NOT split.

    `triangulate_quads` converts quad and n-gon faces into pairs of
    triangles. Only relevant for .x files (xcache is already all-tris).
    """
    with open(filepath, "rb") as fh:
        raw = fh.read()

    if len(raw) < 16:
        raise ValueError("File too short to contain a valid header")

    # Bugsnax .xcache routing — by magic, then by extension as fallback
    if raw[:4] == b"SEMS" or filepath.lower().endswith(".xcache"):
        try:
            return parse_xcache_file(filepath, split_submeshes=split_submeshes)
        except struct.error as exc:
            raise ValueError(
                f"Truncated or corrupt .xcache file: {exc}"
            ) from exc

    if raw[:4] != b"xof ":
        raise ValueError(f"Not a DirectX .x file (bad magic: {raw[:4]!r})")

    fmt_tag    = raw[8:12]
    try:
        float_size = int(raw[12:16])
    except ValueError:
        float_size = 32
    if float_size not in (32, 64):
        float_size = 32

    is_binary     = fmt_tag in (b"bin ", b"bzip")
    is_compressed = fmt_tag in (b"tzip", b"bzip")

    if is_compressed:
        start = 16
        if len(raw) > 18:
            potential_magic = struct.unpack_from('H', raw, start + 2)[0]
            if potential_magic != _MSZIP_MAGIC:
                start += 6
        payload = _mszip_decompress(raw, start)
    else:
        payload = raw[16:]

    if is_binary:
        try:
            root = _BinaryParser(payload, float_size).parse_file()
        except struct.error as exc:
            raise ValueError(
                f"Truncated or corrupt DirectX binary .x file: {exc}"
            ) from exc
    else:
        text = payload.decode("utf-8", errors="replace")
        nl   = text.find("\n")
        body = text[nl + 1:] if nl >= 0 else text
        tokens = _tokenize(body)
        try:
            root = _TextParser(tokens).parse_file()
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Malformed DirectX text .x file: {exc}"
            ) from exc

    # Post-process: triangulate quad faces. Do this BEFORE splitting by
    # material so the material-split's face-index mapping is correct.
    if triangulate_quads:
        _triangulate_mesh_quads(root)

    # Post-process: split multi-material Mesh nodes into per-material
    # sibling Mesh nodes. Only fires when split_submeshes is True AND a
    # Mesh has >1 distinct material in its MeshMaterialList.
    if split_submeshes:
        _split_x_mesh_by_material(root)

    return root
