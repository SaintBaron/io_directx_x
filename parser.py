"""
DirectX .x and Bugsnax .xcache parsers
============================================
Produces a tree of XNode objects that the importer walks, regardless of
whether the source file is text, binary, MS-ZIP compressed (tzip/bzip), or
the SEMS .xcache binary format used by the Horsepower engine (Bugsnax).
"""

import math
import re
import struct
import zlib
from typing import List, Optional, Tuple

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
    r'([A-Za-z_][A-Za-z0-9_.]*)',
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
    __slots__ = ("kind", "name", "children", "values")

    def __init__(self, kind, name=""):
        self.kind     = kind
        self.name     = name
        self.children = []
        self.values   = []

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

        _n_vals = {0: 4, 1: 3, 2: 3, 3: 16, 4: 16}
        n_vals = _n_vals.get(key_type, 4)

        for _ in range(key_count):
            tick = self.read_int()
            nv   = self.read_int()
            node.values.append((TOK_NUM, str(tick)))
            node.values.append((TOK_NUM, str(nv)))
            for _ in range(n_vals):
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
    """Read a 4×4 float32 matrix at *offset* (64 bytes).  Returns a flat list"""
    return list(struct.unpack_from('<16f', data, offset))


def _find_vertex_block(data: bytes, start: int, end: int):
    """Scan *data[start:end]* for a plausible vertex-count uint32 followed by"""
    end = min(end, len(data) - 16)
    STRIDE = 64  # bytes per vertex (16 floats)

    def vert_ok(off):
        """Returns (looks_like_vert, is_nonzero)."""
        if off + 12 > len(data):
            return False, False
        fx = _f32(data, off)
        fy = _f32(data, off + 4)
        fz = _f32(data, off + 8)
        if math.isnan(fx) or math.isnan(fy) or math.isnan(fz):
            return False, False
        if abs(fx) > 500 or abs(fy) > 500 or abs(fz) > 500:
            return False, False
        nonzero = (abs(fx) + abs(fy) + abs(fz)) >= 1e-4
        return True, nonzero

    PROBE_N = 5      # how many vertices to look at for confidence
    MIN_NONZERO = 3  # how many of those must be non-zero

    for scan in range(start, end):
        candidate = struct.unpack_from('<I', data, scan)[0]
        if not (100 < candidate < 300_000):
            continue
        v0_off = scan + 4
        # Reject candidates whose claimed vertex count doesn't physically fit
        if v0_off + candidate * STRIDE + 8 > len(data):
            continue
        # Require the first PROBE_N (or vert_count, whichever is smaller)
        n_probe = min(PROBE_N, candidate)
        all_ok = True
        nonzero_count = 0
        for vi in range(n_probe):
            ok, nz = vert_ok(v0_off + vi * STRIDE)
            if not ok:
                all_ok = False
                break
            if nz:
                nonzero_count += 1
        if not all_ok or nonzero_count < min(MIN_NONZERO, n_probe):
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

        if name_len < 1 or name_len > 128:
            break  # corrupt

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
            # Row 0..2 from the 3x4, row 3 from the translation block.
            for r in range(3):
                for c in range(4):
                    bind_pose[r * 4 + c] = mat3x4[r * 4 + c]
            for c in range(4):
                bind_pose[12 + c] = trans4[c]
        else:
            # Fallback: identity
            bind_pose[0] = bind_pose[5] = bind_pose[10] = bind_pose[15] = 1.0

        bones.append({
            'name':       name,
            'parent':     parent_idx,
            'ftm':        ftm,
            'bind_pose':  bind_pose,        # 4×4 world-space bind pose (DX row-major)
            'data_start': ftm_offset + 64,
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
    """Scan forward from *search_from* for the next valid"""
    end = min(search_from + 200_000, len(data) - 16)
    for off in range(search_from, end, 4):
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
        while cur + 20 <= len(data):
            f = struct.unpack_from('<5f', data, cur)
            qlen = math.sqrt(f[1]*f[1] + f[2]*f[2] + f[3]*f[3] + f[4]*f[4])
            if abs(qlen - 1.0) > 0.05:
                break
            entries[int(f[0])] = (f[1], f[2], f[3], f[4])
            cur += 20
        if len(entries) > 10:
            return entries, cur

    return {}, search_start


def _read_skin_weights(data: bytes, offset: int, bone_end: int):
    """Parse the skin-weight block that immediately follows the rotation section."""
    if offset + 6 > bone_end:
        return []
    count = struct.unpack_from('<I', data, offset)[0]
    if count == 0 or count > 50_000:
        return []
    pad = struct.unpack_from('<H', data, offset + 4)[0]
    if pad != 0:
        return []
    entries_start = offset + 6
    if entries_start + count * 10 > bone_end + 20:  # small tolerance
        return []
    influences = []
    for i in range(count):
        off = entries_start + i * 10
        if off + 10 > len(data):
            break
        vi = struct.unpack_from('<H', data, off)[0]
        w  = struct.unpack_from('<f', data, off + 4)[0]
        if not math.isfinite(w) or w < 0 or w > 1.0 + 1e-4:
            break
        influences.append((vi, w))
    if len(influences) != count:
        return []
    return influences


def _extract_anim(data: bytes, bone: dict, next_bone_start: int):
    """Extract position, scale, and rotation animation keyframes plus skin"""
    anim_start = bone['data_start']
    anim_end   = next_bone_start if next_bone_start else len(data)

    pos_keys   = {}
    scale_keys = {}

    # --- Find stride-16 section ---

    stride16_start = None
    for off in range(anim_start, anim_end - 16, 1):
        try:
            f0, f1, f2, f3 = struct.unpack_from('<4f', data, off)
        except struct.error:
            break
        if any(math.isnan(v) or abs(v) > 1000 for v in (f0, f1, f2, f3)):
            continue
        if f2 >= 1.0 and f2 == math.floor(f2) and f2 <= 10000:
            valid = True
            for step in range(1, 4):
                noff = off + step * 16
                if noff + 16 > anim_end:
                    valid = False; break
                try:
                    g0, g1, g2, g3 = struct.unpack_from('<4f', data, noff)
                except struct.error:
                    valid = False; break
                if any(math.isnan(v) or abs(v) > 1000 for v in (g0, g1, g2, g3)):
                    valid = False; break
                if abs(g2 - (f2 + step)) > 0.5:
                    valid = False; break
            if valid:
                stride16_start = off
                break

    if stride16_start is None:
        return {'pos': pos_keys, 'scale': scale_keys, 'rot': {}, 'skin': []}

    # --- Parse stride-16 POSITION records ---

    pos_records = []
    off = stride16_start
    while off + 16 <= anim_end:
        try:
            f0, f1, f2, f3 = struct.unpack_from('<4f', data, off)
        except struct.error:
            break
        if any(math.isnan(v) or abs(v) > 1000 for v in (f0, f1, f2, f3)):
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
        if any(math.isnan(v) or abs(v) > 1000 for v in (f0, f1, f2, f3)):
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
    skin = _read_skin_weights(data, rot_end, anim_end)

    return {'pos': pos_keys, 'scale': scale_keys, 'rot': rot_keys, 'skin': skin}


# Mesh parsing

def _parse_mesh_section(data: bytes, search_start: int):
    """Locate and parse all mesh sections from *search_start* onwards."""
    meshes = []

    # Search across the whole file for mesh-bone-named entries.  Two name
    pattern = re.compile(
        rb'(?:[A-Za-z][A-Za-z0-9_]*Geo|skinned[A-Za-z0-9_]+)\x00'
    )
    for m in pattern.finditer(data):
        geo_off  = m.start()   # start of name bytes
        name_end = m.end() - 1  # position of the \x00 null

        # Recover name_len and verify the u32 prefix
        name_len = name_end - geo_off
        if name_len < 3 or name_len > 80:
            continue

        name = m.group()[:-1].decode('ascii', errors='replace')

        # The FTM immediately follows the null byte (which is also the first
        # byte of the first matrix float, same convention as bone sections)
        ftm_off = name_end   # null byte = first byte of FTM
        if ftm_off + 64 > len(data):
            continue

        transform = _read_matrix(data, ftm_off)

        # Validate transform looks like a real matrix (not garbage)
        if not any(abs(transform[i*4+i] - 1.0) < 0.1 for i in range(3)):
            continue

        # Vertex block: scan up to 4096 bytes after FTM
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
        tex_paths = []
        tex_pat = re.compile(rb'Content/[^\x00]+\.dds\x00')
        # The region between end-of-FTM and start-of-vertex-data
        mat_block_start = ftm_off + 64   # = data_start for this mesh
        mat_block_end   = vert_data_off  # vertex data begins here
        for tm in tex_pat.finditer(data[mat_block_start:mat_block_end]):
            path = tm.group()[:-1].decode('ascii', errors='replace')
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
        })

    return meshes


def _is_mesh_bone_name(name: str) -> bool:
    """Heuristic: does this bone name look like a mesh entry rather than a"""
    return name.endswith('Geo') or name.startswith('skinned')


def extract_mesh_blocks_from_source(source_path: str) -> list:
    """Read a source .xcache file and return the raw bytes of every mesh"""
    try:
        with open(source_path, 'rb') as fh:
            data = fh.read()
    except (OSError, IOError):
        return []

    if len(data) < 32 or data[:4] != b'SEMS':
        return []

    # Find every mesh-bone-looking entry by name pattern.  The 8 bytes before
    pattern = re.compile(
        rb'(?:[A-Za-z][A-Za-z0-9_]*Geo|skinned[A-Za-z0-9_]+)\x00'
    )
    geo_positions = []
    for m in pattern.finditer(data):
        name = m.group()[:-1].decode('ascii', errors='replace')
        block_start = m.start() - 8
        if block_start < 0:
            continue
        # Sanity check: the u32 at block_start+4 should equal len(name)
        try:
            n_len = struct.unpack_from('<I', data, block_start + 4)[0]
        except struct.error:
            continue
        if n_len != len(name):
            continue
        geo_positions.append((block_start, name))

    if not geo_positions:
        return []

    # Compute block end for each: next mesh block_start, or file end for the last.
    blocks = []
    for i, (start, name) in enumerate(geo_positions):
        if i + 1 < len(geo_positions):
            end = geo_positions[i + 1][0]
        else:
            end = len(data)
        blocks.append({
            'name': name,
            'bytes': bytes(data[start:end]),
        })
    return blocks


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
    """Decode the SEMS bone-parent encoding into absolute indices into *bones*."""
    parents = [-1] * len(bones)
    for i in range(1, len(bones)):
        p = bones[i]['parent']
        if p == 0:
            parents[i] = 0
        elif p == 1:
            parents[i] = i - 1
        else:
            # Sentinel — fall back to ROOT
            parents[i] = 0
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
    """Multiply two 4×4 row-major matrices stored as 16 floats:"""
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += a16[r*4+k] * b16[k*4+c]
            out[r*4+c] = s
    return out


def _build_skeleton_frames(bones: list) -> list:
    """Build a properly-nested Frame XNode tree from the flat bone list."""
    if not bones:
        return []

    # Build a fast lookup of skeleton (non-mesh) bones, but keep their
    # original indices so we can resolve parent indices that point to them.
    is_skel = [not b['name'].endswith('Geo') for b in bones]
    parents = _resolve_parent_indices(bones)

    # Compute parent-local FTM for every skeleton bone:
    local_ftms: dict[int, list] = {}
    for i, b in enumerate(bones):
        if not is_skel[i]:
            continue
        if i == 0 or parents[i] < 0:
            local_ftms[i] = list(b['bind_pose'])
        else:
            par_idx = parents[i]
            par_world = bones[par_idx]['bind_pose']
            local_ftms[i] = _mat_mul(b['bind_pose'], _mat_inv(par_world))

    # Build XNode Frame nodes (one per skeleton bone), then wire up children
    nodes: dict[int, XNode] = {}
    for i, b in enumerate(bones):
        if not is_skel[i]:
            continue
        nodes[i] = _make_frame_node(b['name'], local_ftms[i], [])

    # Attach each child to its parent's children list
    root_node = None
    for i, b in enumerate(bones):
        if not is_skel[i]:
            continue
        if i == 0 or parents[i] < 0:
            root_node = nodes[i]
        else:
            par_idx = parents[i]
            if par_idx in nodes:
                nodes[par_idx].children.append(nodes[i])
            else:
                # Fallback: attach to root if parent index resolves to a
                # mesh bone (shouldn't happen for valid xcache files)
                if root_node is not None:
                    root_node.children.append(nodes[i])

    return [root_node] if root_node is not None else []


def _build_mesh_frame_node(mesh: dict, bones: list) -> XNode:
    """Build a Frame + Mesh + MeshNormals + MeshTextureCoords + MeshMaterialList"""

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

    # --- Materials ---
    tex_paths = mesh.get('tex_paths', [])
    diffuse_path = tex_paths[0] if tex_paths else ""
    mat_name = f"{frame_name}Material"

    def _make_material_node():
        m = XNode("Material", mat_name)
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
        return m

    top_level_mat = _make_material_node()
    inline_mat    = _make_material_node()

    mat_list = XNode("MeshMaterialList", "")
    mat_list.values.append(_num(1))               # 1 material
    mat_list.values.append(_num(len(faces)))      # face count
    for _ in faces:
        mat_list.values.append(_num(0))           # all faces use material 0
    # Keep the inline material so importers that prefer it still get a value;
    mat_list.children.append(inline_mat)
    ref_node = XNode("REF", "")
    ref_node.values.append((TOK_WORD, mat_name))
    mat_list.children.append(ref_node)
    mesh_node.children.append(mat_list)

    # --- XSkinMeshHeader + SkinWeights ---
    if bones:
        # Count how many bones actually influence at least one vertex
        influencing_bones = [b for b in bones if b.get('skin', [])]
        n_inf = len(influencing_bones) if influencing_bones else 1

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

        if influencing_bones:
            for b in influencing_bones:
                influences = b['skin']  # list of (vertex_index, weight)
                sw = XNode("SkinWeights", "")
                sw.values.append(_str(b['name']))
                sw.values.append(_num(len(influences)))
                for vi, _ in influences:
                    sw.values.append(_num(vi))
                for _, w in influences:
                    sw.values.append(_num(w))
                for v in _inv_ftm(b['bind_pose']):
                    sw.values.append(_num(v))
                mesh_node.children.append(sw)
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
            for v in _inv_ftm(root_bone['bind_pose']):
                sw.values.append(_num(v))
            mesh_node.children.append(sw)

    # --- Frame wrapping the mesh ---
    frame_node = XNode("Frame", frame_name)
    frame_node.children.append(_make_ftm_node(mesh['transform']))
    frame_node.children.append(mesh_node)
    return frame_node, top_level_mat


def _build_animation_set(bones: list, anim_data: dict[str, dict], ticks: int) -> XNode:
    """Build an AnimationSet XNode from the extracted animation channels."""
    anim_set = XNode("AnimationSet", "anim")

    for bone in bones:
        name = bone['name']
        # Mesh-container bones (ending in "Geo") must not get animation tracks
        if name.endswith('Geo'):
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

def parse_xcache_file(filepath: str) -> XNode:
    """Parse a Horsepower Engine SEMS .xcache file and return an XNode tree"""
    with open(filepath, 'rb') as fh:
        data = fh.read()

    if len(data) < 28:
        raise ValueError(f"Not a SEMS file: too short ({len(data)} bytes)")
    if data[:4] != b'SEMS':
        raise ValueError(f"Not a SEMS file: magic={data[:4]!r}")

    # --- Header ---
    bone_count = _u32(data, 0x1C)

    # --- Parse bones ---
    bones, after_bones = _parse_bones(data, bone_count, 0x20)

    # --- Extract animation data and skin weights per bone ---
    anim_data: dict[str, dict] = {}
    for i, bone in enumerate(bones):
        next_bone_hdr = after_bones if i + 1 >= len(bones) else (
            bones[i + 1]['data_start'] - 64 - len(bones[i + 1]['name']))
        channels = _extract_anim(data, bone, next_bone_hdr)
        bone['skin'] = channels.get('skin', [])   # attach skin weights to bone dict
        if channels['pos'] or channels['scale'] or channels['rot']:
            anim_data[bone['name']] = channels

    # Keyframe count for AnimTicksPerSecond
    all_ticks: set = set()
    for ch in anim_data.values():
        all_ticks.update(ch.get('pos',   {}).keys())
        all_ticks.update(ch.get('scale', {}).keys())
        all_ticks.update(ch.get('rot',   {}).keys())
    max_tick = max(all_ticks) if all_ticks else 0

    # --- Parse meshes ---
    meshes = _parse_mesh_section(data, after_bones)

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
        frame_n, top_mat = _build_mesh_frame_node(mesh, bones)
        mesh_frame_nodes.append(frame_n)
        if top_mat is not None:
            top_level_mats.append(top_mat)

    # Top-level Materials (matching .x file convention)
    for tlm in top_level_mats:
        root.children.append(tlm)

    # Skeleton frames
    skel_frames = _build_skeleton_frames(bones)
    root.children.extend(skel_frames)

    # Mesh frames
    for fn in mesh_frame_nodes:
        root.children.append(fn)

    # AnimationSet
    if anim_data:
        root.children.append(_build_animation_set(bones, anim_data, int(max_tick)))

    return root

# WRITER (export side)

# ---------- Header ----------

def parse_x_file(filepath: str) -> XNode:
    """Parse a .x or .xcache file and return its XNode tree.

    Auto-detects format by magic bytes:
      * "xof " (offset 0)            → DirectX .x (text/binary/compressed)
      * "SEMS" (offset 0)            → Bugsnax .xcache binary

    Falls back to extension matching for files with non-standard magic.
    """
    with open(filepath, "rb") as fh:
        raw = fh.read()

    if len(raw) < 16:
        raise ValueError("File too short to contain a valid header")

    # Bugsnax .xcache routing — by magic, then by extension as fallback
    if raw[:4] == b"SEMS" or filepath.lower().endswith(".xcache"):
        return parse_xcache_file(filepath)

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
        return _BinaryParser(payload, float_size).parse_file()

    text = payload.decode("utf-8", errors="replace")
    nl   = text.find("\n")
    body = text[nl + 1:] if nl >= 0 else text
    tokens = _tokenize(body)
    return _TextParser(tokens).parse_file()
