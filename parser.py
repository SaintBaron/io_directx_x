"""
DirectX .x parser — text and binary formats
============================================
Produces a tree of XNode objects that the importer walks, regardless of
whether the source file is text, binary, or MS-ZIP compressed (tzip/bzip).
"""

import re
import struct
import zlib


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


# ─────────────────────────────────────────────────────────────
#  Text parser
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  Binary token constants
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  Binary parser
# ─────────────────────────────────────────────────────────────
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
        for _ in range(4):   # diffuse RGBA
            node.values.append((TOK_NUM, repr(self.read_float())))
        node.values.append((TOK_NUM, repr(self.read_float())))  # shininess
        for _ in range(3):   # specular RGB
            node.values.append((TOK_NUM, repr(self.read_float())))
        for _ in range(3):   # emissive RGB
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


# ─────────────────────────────────────────────────────────────
#  MS-ZIP decompressor
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────
def parse_x_file(filepath: str) -> XNode:
    """Parse a DirectX .x file (text, binary, or MS-ZIP compressed).
    Returns an XNode tree with kind="ROOT".
    """
    with open(filepath, "rb") as fh:
        raw = fh.read()

    if len(raw) < 16:
        raise ValueError("X file too short to contain a valid header")
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
