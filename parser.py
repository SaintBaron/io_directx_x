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

        # The TRUE skinning offset matrix (= inverse of the bone's mesh-space
        # bind pose, used as the SkinWeights "matrixOffset" in DirectX .x
        # format) is stored at ftm_offset + 296, NOT at ftm_offset + 64.
        # The first three 64-byte blocks after the FTM are duplicates of
        # the chained-FTM walk and the FTM itself, used by the engine for
        # other purposes; they are NOT the skin-bind offset. The 40 bytes
        # after those duplicates are header/decoration, and the actual
        # offset matrix sits at byte +296 from the FTM start.
        # Verified against the working .x exports of Orange/Spaghetti/
        # Potato/Taco where 100% of bones have their .x SkinWeights offset
        # bit-identical to the matrix at this position.
        skin_offset_at = ftm_offset + 296
        skin_offset = [0.0] * 16
        if skin_offset_at + 64 <= len(data):
            skin_offset = list(struct.unpack_from('<16f', data, skin_offset_at))
        else:
            # Fallback: identity (better than zero-matrix, which would be
            # singular and break vertex skinning entirely)
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
    """
    end = min(search_from + 200_000, len(data) - 16)
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
    """
    end = min(search_end - 6, search_start + 100_000)
    for off in range(search_start, end):
        count = struct.unpack_from('<I', data, off)[0]
        if count < 10 or count > 20_000:
            continue
        pad = struct.unpack_from('<H', data, off + 4)[0]
        if pad not in (0, 1):
            continue
        block_end = off + 6 + count * 10
        if block_end > search_end:
            continue
        # Validate by checking the first 20 entries look like skin data:
        # each is (vi: u16, _: u32, w: float) with sane values, and
        # weights must be REAL skin weights (≥ 0.001) — not denormalized
        # noise (~1e-39) that happens to pass a 0 ≤ w ≤ 1 check.
        sample_n = min(count, 20)
        prev_vi = -1
        ok = True
        n_real_weights = 0
        max_vi = 0
        for j in range(sample_n):
            eo = off + 6 + j * 10
            vi = struct.unpack_from('<H', data, eo)[0]
            w  = struct.unpack_from('<f', data, eo + 4)[0]
            # Reject denormalized floats by requiring weight to be
            # zero exactly OR ≥ 0.001. (Most real skin weights are
            # 0.25, 0.5, 1.0; the smallest meaningful weight in any
            # production file is well above 0.001.)
            if not math.isfinite(w):
                ok = False; break
            if not (0.0 <= w <= 1.001):
                ok = False; break
            if 0 < w < 0.001:
                # denormalized noise
                ok = False; break
            if w >= 0.001:
                n_real_weights += 1
            if vi >= 50_000:
                ok = False; break
            if vi > max_vi:
                max_vi = vi
            # Allow ONE reset (mesh2 continuation pattern from Honey)
            # but not multiple disorder events.
            if prev_vi >= 0 and vi < prev_vi:
                if prev_vi - vi < 100:
                    ok = False; break
            prev_vi = vi
        # Require at least half of the sampled weights to be real
        # (≥ 0.001). Skin blocks may have a few padding zeros, but
        # not 18-of-20 noise.
        if ok and n_real_weights >= sample_n // 2:
            return off
    return None


def _read_skin_weights(data: bytes, offset: int, bone_end: int):
    """Parse the skin-weight block that immediately follows the rotation section.

    Returns (influences, pad, reset_at).
      • pad=0 means full-weight bones; pad=1 means partial-weight bones.
      • reset_at is the index in `influences` where the vi sequence drops
        back to near zero (indicating a continuation onto a second mesh).
        For Honey's root bone, this is at index 2200 — the first 2200
        entries reference mesh1, the remaining reference mesh2 (with
        vi-relative-to-mesh2 indexing). reset_at=None means no reset.
    """
    if offset + 6 > bone_end:
        return [], 0, None
    count = struct.unpack_from('<I', data, offset)[0]
    if count == 0 or count > 50_000:
        return [], 0, None
    pad = struct.unpack_from('<H', data, offset + 4)[0]
    if pad not in (0, 1):
        return [], 0, None
    entries_start = offset + 6
    if entries_start + count * 10 > bone_end + 20:
        return [], 0, None
    influences = []
    reset_at = None
    prev_vi = -1
    for i in range(count):
        off = entries_start + i * 10
        if off + 10 > len(data):
            break
        vi = struct.unpack_from('<H', data, off)[0]
        w  = struct.unpack_from('<f', data, off + 4)[0]
        if not math.isfinite(w) or w < 0 or w > 1.0 + 1e-4:
            break
        # Detect reset: vi drops by 100+ from previous vi. This signals
        # a continuation onto another mesh (Honey's root SkinWeights
        # crosses from mesh1's high indices back to mesh2's vi=0).
        if reset_at is None and prev_vi >= 100 and vi < prev_vi - 100:
            reset_at = i
        influences.append((vi, w))
        prev_vi = vi
    if len(influences) != count:
        return [], 0, None
    return influences, pad, reset_at


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
        # No animation data for this bone, but a skin block might
        # still exist further along (e.g. Olive's RootFix fixup
        # bone has 1136 skin weights but no rotation/pos/scale).
        skin, skin_pad, skin_reset = [], 0, None
        scan_from = _find_skin_block(data, anim_start, anim_end)
        if scan_from is not None:
            skin, skin_pad, skin_reset = _read_skin_weights(data, scan_from, anim_end)
        return {'pos': pos_keys, 'scale': scale_keys, 'rot': {},
                'skin': skin, 'skin_pad': skin_pad, 'skin_reset': skin_reset}

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
    # For mesh-frame bones like Olive's "OliveRig_RootFix" there is
    # no rotation section, so _find_rot20_section returns rot_end ==
    # stride16_end — pointing at FTM/matrix data rather than the
    # skin block. The skin block still exists further along the
    # bone's data, so scan for it by signature.
    skin, skin_pad, skin_reset = _read_skin_weights(data, rot_end, anim_end)
    if not skin:
        scan_from = _find_skin_block(data, rot_end, anim_end)
        if scan_from is not None:
            skin, skin_pad, skin_reset = _read_skin_weights(data, scan_from, anim_end)

    return {'pos': pos_keys, 'scale': scale_keys, 'rot': rot_keys,
            'skin': skin, 'skin_pad': skin_pad, 'skin_reset': skin_reset}


# Mesh parsing

def _parse_mesh_section(data: bytes, search_start: int):
    """Locate and parse all mesh sections from *search_start* onwards."""
    meshes = []

    # Match any printable name (3-80 chars) followed by \x00 — the null is the
    # LSByte of the first float in the following FTM matrix.  Mesh entries can
    # have any name (e.g. CeleryGeo, skinnedGrumpus, CrinkleFryder), so we
    # don't constrain the name shape here.  False positives are filtered by
    # the FTM-validity and vertex-block-scan checks downstream.
    pattern = re.compile(rb'[A-Za-z][A-Za-z0-9_]{2,79}\x00')
    seen_offsets = set()
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
        # structure (matrix dumps, animation curves, etc).
        # Real meshes have all-finite positions in a reasonable
        # range; fixup frames (e.g. Olive's "OliveRig_RootFix")
        # have a regex-matchable name but no real geometry — the
        # bytes scanned as "verts" are mostly identity-matrix
        # patterns or garbage that misleads _find_vertex_block.
        bad = 0
        n_zero = 0
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
            if x == 0.0 and y == 0.0 and z == 0.0:
                n_zero += 1
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
        # All-zero vert ratio: real character meshes have
        # essentially zero verts at the exact origin. Fixup-frame
        # fake "vert blocks" are dominated by zeros (the matrix
        # entries that aren't on the diagonal). Reject if >5% of
        # verts are at (0, 0, 0).
        if n_zero > max(4, vert_count // 20):
            continue
        # Tight bbox check: a real character mesh has spatial
        # extent in all three axes. Fixup-frame fake "vert blocks"
        # are typically clamped to the unit cube with most entries
        # at integer coordinates. Require at least one axis to
        # span > 0.1 units.
        spans = (sx_max - sx_min, sy_max - sy_min, sz_max - sz_min)
        if max(spans) < 0.1:
            continue

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
            'data_end':  scan,   # offset where this mesh's face buffer ended
        })

    # Pass 2: scan for UNNAMED continuation meshes that follow the
    # named ones. Honey.xcache stores its model in two parts:
    #   • Mesh1 (named "HoneyStick") = the honey-stick body
    #   • Mesh2 (unnamed)            = the goop overlay (head + wings)
    # Mesh2 starts immediately after mesh1's face buffer with a
    # bbox(6 floats) + 4x4 identity transform + texture paths +
    # vert block layout — same shape as mesh1's data, just without
    # a preceding name. The pad=1 SkinWeights (wing/aux bones)
    # reference verts in this mesh.
    while meshes:
        last_end = meshes[-1].get('data_end')
        if last_end is None or last_end + 88 > len(data):
            break
        # Try to read a bbox+matrix header. 6 floats bbox + 16 floats matrix = 88 bytes.
        try:
            hdr = struct.unpack_from('<22f', data, last_end)
        except struct.error:
            break
        # Validate: matrix diag should be ~1 (identity), and bbox should be finite.
        if not all(math.isfinite(v) for v in hdr[:22]):
            break
        # The 16-float matrix is at index 6. Diag positions in row-major are 6+0, 6+5, 6+10, 6+15.
        if not (abs(hdr[6] - 1.0) < 0.1 and abs(hdr[11] - 1.0) < 0.1 and
                abs(hdr[16] - 1.0) < 0.1 and abs(hdr[21] - 1.0) < 0.1):
            break
        # Find the vert block in the region after the header.
        try:
            vert_count, vert_data_off = _find_vertex_block(
                data, last_end + 88, last_end + 88 + 8192)
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
        tex_paths = []
        tex_pat = re.compile(rb'Content/[^\x00]+\.dds\x00')
        for tm in tex_pat.finditer(data[last_end + 88:vert_data_off]):
            path = tm.group()[:-1].decode('ascii', errors='replace')
            if path not in tex_paths:
                tex_paths.append(path)

        # Synthesize a unique name for this continuation mesh.
        anon_name = f"{meshes[-1]['name']}_Part{len(meshes)+1}"
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

    return meshes


def _is_mesh_bone_name(name: str) -> bool:
    """Heuristic: bone-list entries that aren't actual joint bones.

    Joint bones in .xcache files end in 'SHJnt' (Maya HumanIK convention).
    Anything else in the bone list is a mesh entry — common patterns are
    names ending in 'Geo' (e.g. CeleryGeo), starting with 'skinned'
    (skinnedGrumpus), or arbitrary mesh-asset names (CrinkleFryder).
    """
    return not name.endswith('SHJnt')


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
    """Recover each bone's parent by geometric inference.

    Strategy: search prior bones for the parent that satisfies
    `ftm @ parent_bind == bind` within tolerance. Only joint bones
    (names ending in 'SHJnt') are considered as parent candidates —
    mesh frames like 'CarrotGeo' or 'HoneyStick' happen to have
    identity transforms and would otherwise match falsely.

    If no good parent is found, fall back to top-level (parent = -1).
    This correctly handles both:
      - Honey-style hierarchies where '_01' wing bones (FTM == bind
        in world space) belong under an identity-bind root bone, and
      - Carrot-style flat hierarchies where every spine bone is a
        top-level sibling because no other joint bone has the right
        bind to be its parent.
    """
    parents = [-1] * len(bones)
    if not bones:
        return parents

    TOL = 1e-3
    is_joint = [b['name'].endswith('SHJnt') for b in bones]

    for i in range(len(bones)):
        ftm  = bones[i]['ftm']
        bind = bones[i]['bind_pose']

        # Search for parent satisfying ftm @ parent_bind == bind.
        # Only joint bones are eligible as parents.
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

        if best_err < TOL and best != -1:
            # Self-parenting would mean ftm == bind AND this bone's
            # own bind happens to satisfy ftm @ bind == bind, which
            # only holds when bind is identity-ish. Avoid that.
            if best != i:
                parents[i] = best
            else:
                parents[i] = -1
        else:
            parents[i] = -1

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
                # Use the actual skinning offset matrix from the file
                # (read at ftm_offset + 296 in _parse_bones), which equals
                # the SkinWeights matrixOffset stored by the .x exporter
                # for the same character. Falls back to inv(bind_pose) if
                # for some reason skin_offset is missing or zero (e.g. very
                # old / non-standard files).
                skin_off = b.get('skin_offset')
                if skin_off and any(abs(v) > 1e-9 for v in skin_off):
                    for v in skin_off:
                        sw.values.append(_num(v))
                else:
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
    return frame_node, top_level_mat


def _build_animation_set(bones: list, anim_data: dict[str, dict], ticks: int) -> XNode:
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
        channels = _extract_anim(data, bone, next_bone_hdr)
        bone['skin']       = channels.get('skin', [])
        bone['skin_pad']   = channels.get('skin_pad', 0)
        bone['skin_reset'] = channels.get('skin_reset', None)
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

    # If we found a continuation mesh (e.g. Honey's goop overlay), merge
    # it into the primary mesh. pad=1 SkinWeights reference the
    # continuation's verts using LOW indices [0, len(mesh2.verts)) —
    # after merging, those indices need to be offset by len(mesh1.verts)
    # so they index into the combined vert array. pad=0 SkinWeights
    # keep their original indices (they reference mesh1).
    if len(meshes) > 1:
        primary = meshes[0]
        n_primary_verts = len(primary['verts'])
        # Concatenate verts/normals/uvs from continuation meshes.
        # Faces from continuation meshes get their indices shifted by
        # the cumulative vert count.
        merged_faces = list(primary['faces'])
        cumulative = n_primary_verts
        offsets_per_mesh = [0]  # mesh i's verts start at offsets_per_mesh[i]
        for cont in meshes[1:]:
            offsets_per_mesh.append(cumulative)
            primary['verts'].extend(cont['verts'])
            primary['normals'].extend(cont['normals'])
            primary['uvs'].extend(cont['uvs'])
            for a, b, c in cont['faces']:
                merged_faces.append((a + cumulative, b + cumulative, c + cumulative))
            # Texture paths: append unique
            for tp in cont['tex_paths']:
                if tp not in primary['tex_paths']:
                    primary['tex_paths'].append(tp)
            cumulative += len(cont['verts'])
        primary['faces'] = merged_faces
        meshes = [primary]

        # Remap SkinWeights vi to point into the merged vert array.
        # Two cases:
        #   • pad=1 bones: every entry references the continuation
        #     mesh, so the whole list gets offset by cont_offset.
        #   • pad=0 bones with a reset_at boundary: the entries up to
        #     reset_at reference mesh1 (no shift), and entries from
        #     reset_at onwards reference the continuation mesh and
        #     get shifted by cont_offset. Honey's root bone is the
        #     canonical case — 2200 mesh1 entries followed by 1224
        #     mesh2 entries inside a single SkinWeights block.
        cont_offset = offsets_per_mesh[1] if len(offsets_per_mesh) > 1 else 0
        for b in bones:
            skin = b.get('skin')
            if not skin:
                continue
            pad = b.get('skin_pad', 0)
            reset_at = b.get('skin_reset')
            if pad == 1:
                b['skin'] = [(vi + cont_offset, w) for vi, w in skin]
            elif reset_at is not None and 0 < reset_at < len(skin):
                b['skin'] = (skin[:reset_at]
                             + [(vi + cont_offset, w) for vi, w in skin[reset_at:]])

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
