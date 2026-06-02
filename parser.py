"""
DirectX .x parser
============================================
Produces a tree of XNode objects that the importer walks, regardless of
whether the source file is text, binary, MS-ZIP compressed (tzip/bzip), or
related binary cache formats (handled by a separate addon).
"""

import math
import os
import re
import struct
import warnings
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
    r'((?://|\#)[^\n]*)|'
    r'([{};,])|'
    r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)|'
    r'([A-Za-z_][A-Za-z0-9_.:\-]*)',
)
# The comment alternative matches BOTH '//' (C++ style) and '#' (number-
# sign) comments, each running to end of line. The DirectX .x spec
# (Paul Bourke / cgdev DX2-3 reference) explicitly permits both styles;
# some exporters emit '#'. Without this, a '#' line drops the '#' but
# leaks the comment words as identifiers and derails the parse.
# Note on the word pattern's trailing '-': DirectX .x identifiers are
# officially [A-Za-z_][A-Za-z0-9_]*, but real exporters (some editors,
# some exporters) emit hyphenated frame/animation names like "Anim-1" and
# "Anim-Bip01". Allowing '-' as an identifier CONTINUATION char captures
# those as a single name. Because the number alternative is tried first
# and the word must START with a letter/underscore, a standalone "-1" in
# a value list still tokenizes as a number, not a name.

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
        # Separators (SEMICOLON/COMMA tokens) only occur BETWEEN values
        # at the token level, never inside an active INT_LIST/FLT_LIST
        # run. While _num_count > 0 we are mid-list: the upcoming bytes
        # are raw list payload (float/int data), and a value like the
        # float 0x403a0014 starts with bytes `14 00` that look exactly
        # like a SEMICOLON token (0x0014). Reading a u16 here would
        # misinterpret that payload as a separator, consume 2 bytes, and
        # desync the whole stream (e.g. AIKO.X's 7482-vert mesh, whose
        # vertex data contains such byte patterns — the face count then
        # reads as 0 and the mesh imports as a point cloud). So: never
        # touch the stream while a list is still being consumed.
        if self._num_count > 0:
            return
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
            if tok == _BIN_TOK_TEMPLATE:
                # A `template <Name> { GUID member... }` definition.
                # We must consume the WHOLE definition here. Otherwise
                # the TEMPLATE token gets discarded and the template's
                # *name* (e.g. "Mesh", "Material", "Frame") is then
                # dispatched as if it were a real data object — which
                # mis-parses the template body as geometry/material
                # data. Files exported by tools that emit the full
                # standard template set up front (e.g. some exporters) hit
                # this; some exporters files happened not to.
                self._eat_peeked()
                self._p_template()
                continue
            if tok != _BIN_TOK_NAME:
                self._eat_peeked(); continue
            p_before = self._p
            try:
                node = self._dispatch(val)
            except (struct.error, ValueError, IndexError):
                # A child object failed to parse. On a healthy file this
                # never fires; it happens when a salvaged payload has a
                # zero-filled (corrupt) region that derails token-level
                # parsing. Rather than abort — losing every object after
                # the damage — scan forward to the next plausible
                # top-level object header and resume. This recovers, for
                # example, the entire AnimationSet that follows a corrupt
                # mesh-data block.
                if not self._resync_after(p_before):
                    break
                continue
            if node:
                root.children.append(node)
        return root

    def _resync_after(self, after_p: int) -> bool:
        """Scan forward from after_p+1 for the next valid NAME token that
        names a known top-level object, and position the cursor there.
        Returns True if a resync point was found, False at end-of-buffer.
        """
        known = (b"Frame", b"Mesh", b"AnimationSet", b"Animation",
                 b"Material", b"AnimTicksPerSecond", b"Header")
        scan = max(after_p + 1, 0)
        buf, end = self._buf, self._end
        while scan + 6 <= end:
            # Binary NAME token: u16 tag (0x0001) + u32 length + ascii.
            i = buf.find(b"\x01\x00", scan)
            if i < 0 or i + 6 > end:
                return False
            n = struct.unpack_from('I', buf, i + 2)[0]
            if 2 <= n <= 40 and i + 6 + n <= end:
                name = buf[i + 6 : i + 6 + n]
                if name in known and all(32 <= b < 127 for b in name):
                    self._p = i
                    self._peeked = None
                    self._num_count = 0
                    return True
            scan = i + 2
        return False

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
        # MeshNormals carries its OWN nFaceNormals count, which is a
        # separate DWORD in the stream — it is NOT guaranteed to equal
        # the parent mesh's face count, and even when it does it still
        # occupies a slot in the (flattened) binary integer list. The
        # old code used the passed-in n_faces and skipped reading this
        # count, which desynced the binary stream by one integer (the
        # leading count was then misread as the first face's vertex
        # count). Read it from the stream like _p_mesh does for faces.
        n_face_normals = self.read_int()
        node.values.append((TOK_NUM, str(n_face_normals)))
        for _ in range(n_face_normals):
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
            elif tok == _BIN_TOK_NAME:
                # Unknown named child object — consume its full body so
                # its braces/data don't desync the AnimationSet loop.
                self._eat_peeked()
                self._p_generic(val or "Unknown")
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
            elif tok == _BIN_TOK_NAME:
                # Some other named child object inside the Animation,
                # e.g. AnimationOptions { openclosed; positionquality; }.
                # We must consume its ENTIRE brace-delimited body, not
                # just drop the name token — otherwise its '{', data and
                # '}' are misparsed as siblings, desyncing the stream and
                # dropping every Animation/AnimationKey that follows.
                # (SS-officer.X hit this: its first Animation carries an
                # AnimationOptions block, and the old code lost all 27
                # animations + keys after it.)
                self._eat_peeked()
                self._p_generic(val or "Unknown")
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

_MSZIP_BLOCK = 32768   # MSZIP decompresses 32 KiB per block (last block may be shorter)


# ---------------------------------------------------------------------------
# Lenient pure-Python DEFLATE (RFC 1951), used ONLY to salvage a genuinely
# corrupt MSZIP block that strict zlib refuses to decode. On healthy data it
# mirrors zlib exactly; on a damaged block it (a) trusts the stored-block LEN
# even when its one's-complement NLEN check fails and (b) clamps an
# out-of-range back-reference to emit zeros rather than aborting. Recovering
# real bytes from the bad block keeps the shared 32 KiB LZ77 window valid for
# every later block, so the binary token parser no longer desyncs and drops
# the animation channels that follow. some exporters tolerates the same on-disk
# damage; this brings the importer to parity instead of zero-filling (which
# poisoned the window and lost ~all animation downstream of the damage).
# This runs only on the rare failed block, so the cost is negligible.
# ---------------------------------------------------------------------------

_INF_LEXTRA = [(257, 0, 3), (258, 0, 4), (259, 0, 5), (260, 0, 6), (261, 0, 7),
    (262, 0, 8), (263, 0, 9), (264, 0, 10), (265, 1, 11), (266, 1, 13),
    (267, 1, 15), (268, 1, 17), (269, 2, 19), (270, 2, 23), (271, 2, 27),
    (272, 2, 31), (273, 3, 35), (274, 3, 43), (275, 3, 51), (276, 3, 59),
    (277, 4, 67), (278, 4, 83), (279, 4, 99), (280, 4, 115), (281, 5, 131),
    (282, 5, 163), (283, 5, 195), (284, 5, 227), (285, 0, 258)]
_INF_LBASE = {c: (e, b) for c, e, b in _INF_LEXTRA}
_INF_DEXTRA = [(0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 0, 4), (4, 1, 5), (5, 1, 7),
    (6, 2, 9), (7, 2, 13), (8, 3, 17), (9, 3, 25), (10, 4, 33), (11, 4, 49),
    (12, 5, 65), (13, 5, 97), (14, 6, 129), (15, 6, 193), (16, 7, 257),
    (17, 7, 385), (18, 8, 513), (19, 8, 769), (20, 9, 1025), (21, 9, 1537),
    (22, 10, 2049), (23, 10, 3073), (24, 11, 4097), (25, 11, 6145),
    (26, 12, 8193), (27, 12, 12289), (28, 13, 16385), (29, 13, 24577)]
_INF_DBASE = {c: (e, b) for c, e, b in _INF_DEXTRA}
_INF_CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
_INF_FIXED = None


class _InfBitReader:
    __slots__ = ("data", "pos", "bitbuf", "bitcnt")

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.bitbuf = 0
        self.bitcnt = 0

    def bit(self):
        if self.bitcnt == 0:
            self.bitbuf = self.data[self.pos]
            self.pos += 1
            self.bitcnt = 8
        b = self.bitbuf & 1
        self.bitbuf >>= 1
        self.bitcnt -= 1
        return b

    def bits(self, n):
        v = 0
        for i in range(n):
            v |= self.bit() << i
        return v


def _inf_build(lengths):
    """Build a canonical Huffman decode table {(bitlen, code): symbol}."""
    maxlen = max(lengths) if lengths else 0
    blcount = [0] * (maxlen + 1)
    for l in lengths:
        if l:
            blcount[l] += 1
    code = 0
    nextcode = [0] * (maxlen + 1)
    for bits in range(1, maxlen + 1):
        code = (code + blcount[bits - 1]) << 1
        nextcode[bits] = code
    table = {}
    for sym, l in enumerate(lengths):
        if l:
            table[(l, nextcode[l])] = sym
            nextcode[l] += 1
    return table, maxlen


def _inf_decode_sym(br, table, maxlen):
    code = 0
    for l in range(1, maxlen + 1):
        code = (code << 1) | br.bit()
        hit = table.get((l, code))
        if hit is not None:
            return hit
    raise ValueError("bad huffman code")


def _lenient_inflate(data, window=b"", max_out=_MSZIP_BLOCK):
    """Tolerant raw-DEFLATE decode of a single MSZIP block.

    `window` seeds the LZ77 history (the previous 32 KiB of decoded output);
    back-references resolve into it. Returns the freshly decoded bytes with
    the window prefix stripped. Tolerant of a corrupt stored-block length and
    of an out-of-range distance (emits zeros) so a damaged block yields
    usable data instead of raising. May raise on a hopelessly broken stream;
    callers treat that as "fall back to zero-fill".
    """
    global _INF_FIXED
    br = _InfBitReader(data)
    out = bytearray(window)
    wlen = len(window)
    n = len(data)
    while True:
        bfinal = br.bit()
        btype = br.bits(2)
        if btype == 0:
            br.bitcnt = 0           # skip to byte boundary (discard partial byte)
            if br.pos + 4 > n:
                break
            length = br.data[br.pos] | (br.data[br.pos + 1] << 8)
            br.pos += 4             # consume LEN + NLEN; trust LEN (lenient on NLEN)
            out += br.data[br.pos:br.pos + length]
            br.pos += length
        elif btype == 1 or btype == 2:
            if btype == 1:
                if _INF_FIXED is None:
                    ll = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
                    _INF_FIXED = (_inf_build(ll), _inf_build([5] * 30))
                (lt, lm), (dt, dm) = _INF_FIXED
            else:
                hlit = br.bits(5) + 257
                hdist = br.bits(5) + 1
                hclen = br.bits(4) + 4
                cl = [0] * 19
                for i in range(hclen):
                    cl[_INF_CL_ORDER[i]] = br.bits(3)
                clt, clm = _inf_build(cl)
                lens = []
                while len(lens) < hlit + hdist:
                    s = _inf_decode_sym(br, clt, clm)
                    if s < 16:
                        lens.append(s)
                    elif s == 16:
                        lens += [lens[-1]] * (br.bits(2) + 3)
                    elif s == 17:
                        lens += [0] * (br.bits(3) + 3)
                    else:
                        lens += [0] * (br.bits(7) + 11)
                lt, lm = _inf_build(lens[:hlit])
                dt, dm = _inf_build(lens[hlit:hlit + hdist])
            while True:
                s = _inf_decode_sym(br, lt, lm)
                if s == 256:
                    break
                if s < 256:
                    out.append(s)
                else:
                    e, b = _INF_LBASE[s]
                    length = b + br.bits(e)
                    ds = _inf_decode_sym(br, dt, dm)
                    de, db = _INF_DBASE[ds]
                    dist = db + br.bits(de)
                    start = len(out) - dist
                    if start < 0:
                        out += b"\x00" * length     # history unavailable -> zeros
                        continue
                    for i in range(length):
                        out.append(out[start + i])
        else:
            break       # invalid block type -> stop, keep what decoded
        if bfinal:
            break
        if len(out) - wlen >= max_out:
            break
    return bytes(out[wlen:])


def _mszip_decompress(buf: bytes, start: int, salvage: bool = False) -> bytes:
    """Decompress an MSZIP-framed DirectX .x payload.

    MSZIP blocks share a single LZ77 history window: every block after
    the first may emit back-references into the previous block's last
    32 KiB of *decompressed* output. We seed each block's inflater with
    the tail of everything decompressed so far via zdict. Without this,
    multi-block files (anything whose decompressed size exceeds one
    32 KiB block) fail with zlib error -3 "invalid distance too far
    back". Single-block files are unaffected (empty zdict).

    By default this is strict: a corrupt block or bad framing raises
    ValueError, so healthy files behave exactly as before. When
    `salvage=True`, a block that fails to inflate is replaced with a
    32 KiB filler block (keeping byte offsets of later blocks aligned)
    and decoding continues; a framing break triggers a forward scan for
    the next 'CK' marker. This recovers the bulk of a file that has a
    localized on-disk corruption (e.g. one damaged compressed block)
    instead of losing the whole import. Salvaged blocks are recorded on
    the function's `last_salvage` attribute for the caller to inspect /
    warn about.
    """
    chunks, p, end = [], start, len(buf)
    window = b""
    bad_blocks = []
    lenient_blocks = []
    resyncs = 0
    block_idx = 0
    while p + 4 <= end:
        chunk_size = struct.unpack_from('H', buf, p)[0]
        magic      = struct.unpack_from('H', buf, p + 2)[0]
        if magic != _MSZIP_MAGIC:
            if not salvage:
                raise ValueError(
                    f"X: bad MSZIP magic 0x{magic:04x} at offset {p+2}")
            # Framing break — scan forward for the next 'CK' (0x4b43)
            # marker and resume from the length word preceding it.
            nxt = buf.find(b"\x43\x4b", p + 1)
            if nxt < 0:
                break
            p = nxt - 2
            resyncs += 1
            continue
        raw = buf[p + 4 : p + 4 + chunk_size]
        p += 4 + chunk_size
        try:
            d = zlib.decompressobj(wbits=-15, zdict=window) if window \
                else zlib.decompressobj(wbits=-15)
            out = d.decompress(raw) + d.flush()
        except zlib.error as exc:
            if not salvage:
                raise ValueError(
                    f"X: zlib error in MSZIP chunk: {exc}") from exc
            # A block strict zlib rejects (corrupt stored-block length or an
            # out-of-range distance). First try a tolerant pure-Python
            # inflate seeded with the shared window: it recovers real bytes,
            # keeping the 32 KiB LZ77 window valid for every later block so
            # the token parser no longer desyncs and loses the animation
            # channels that follow the damage. Only if that also fails do we
            # zero-fill as a last resort (keeps later blocks' byte offsets
            # aligned but loses this block's data).
            out = None
            try:
                rec = _lenient_inflate(raw, window=window)
                if rec:
                    if len(rec) < _MSZIP_BLOCK:
                        rec = rec + b"\x00" * (_MSZIP_BLOCK - len(rec))
                    out = bytes(rec[:_MSZIP_BLOCK])
                    lenient_blocks.append(block_idx)
            except Exception:
                out = None
            if out is None:
                # Last resort: keep whatever partial output zlib decoded,
                # then pad to a full block with zeros. A full-size filler
                # keeps later blocks' back-references in range (distance <=
                # 32 KiB) so they decode structurally even though bytes that
                # referenced the lost block are wrong.
                partial = bytearray()
                try:
                    d2 = zlib.decompressobj(wbits=-15, zdict=window) if window \
                        else zlib.decompressobj(wbits=-15)
                    partial += d2.decompress(raw)
                except zlib.error:
                    pass
                if len(partial) < _MSZIP_BLOCK:
                    partial += b"\x00" * (_MSZIP_BLOCK - len(partial))
                out = bytes(partial[:_MSZIP_BLOCK])
                bad_blocks.append(block_idx)
        chunks.append(out)
        window = (window + out)[-32768:]
        block_idx += 1
    _mszip_decompress.last_salvage = {
        "bad_blocks": bad_blocks,
        "lenient_blocks": lenient_blocks,
        "resyncs": resyncs,
        "total_blocks": block_idx,
    }
    return b"".join(chunks)


_mszip_decompress.last_salvage = {
    "bad_blocks": [], "lenient_blocks": [], "resyncs": 0, "total_blocks": 0,
}


def _rescue_binary_animations(payload: bytes, float_size: int, root) -> int:
    """Recover animation channels from a salvaged (corruption-zeroed) payload.

    When an MSZIP block that lands inside the animation data is corrupt on
    disk, the salvage decompressor zero-fills it to keep later blocks byte
    aligned. Those zero bytes desync the sequential binary token parser: it
    typically recovers only the first Animation channel and then loses every
    channel after the damaged region (they become misaligned orphans, and a
    bare top-level `Animation` isn't even dispatched into an AnimationSet).

    This re-scans the raw payload for every `Animation` channel header and
    parses each one independently, re-aligning the cursor at the start of
    each channel so a single damaged block can't cascade past itself. Only
    channels with a resolvable bone REF and at least one non-empty key are
    kept; channels physically inside the corrupt block are unrecoverable and
    skipped (the bones they target simply keep their rest pose, which is the
    correct fallback). The first clean channel per target bone wins.

    Returns the number of channels attached. Intended to be called ONLY when
    MSZIP salvage reported damage, so healthy files are never touched.
    """
    # Binary NAME token for the word "Animation":
    #   u16(0x0001)  +  u32(9)  +  b"Animation"
    hdr = b"\x01\x00" + struct.pack("I", 9) + b"Animation"
    positions = []
    i = payload.find(hdr)
    while i >= 0:
        positions.append(i)
        i = payload.find(hdr, i + 1)
    if not positions:
        return 0

    channels = []
    seen_refs = set()
    for pos in positions:
        bp = _BinaryParser(payload, float_size)
        bp._p = pos
        bp._peeked = None
        bp._num_count = 0
        tok, val = bp._peek()
        if tok != _BIN_TOK_NAME or (val or "").lower() != "animation":
            continue
        bp._eat_peeked()
        try:
            node = bp._p_animation()
        except (struct.error, ValueError, IndexError):
            continue
        ref = node.child("REF")
        bname = None
        if ref:
            bname = next(
                (v for t, v in ref.values if t in (TOK_WORD, TOK_STR)), None)
        if not bname or bname in seen_refs:
            continue
        # Require at least one AnimationKey that actually carries frames.
        good = False
        for k in node.children_of("AnimationKey"):
            kn = k.nums()
            if len(kn) >= 3 and int(kn[1]) >= 1:
                good = True
                break
        if not good:
            continue
        seen_refs.add(bname)
        channels.append(node)

    if not channels:
        return 0

    aset = next((n for n in root.children if n.kind == "AnimationSet"), None)
    if aset is None:
        aset = XNode("AnimationSet", "Anim-1")
        root.children.append(aset)
    # Drop whatever sparse / partially-corrupt Animation children the
    # sequential parser produced and substitute the full rescued set.
    aset.children = [c for c in aset.children
                     if c.kind != "Animation"] + channels
    return len(channels)


def _sanitize_anim_keys(root) -> int:
    """Clamp implausible matrix-key values left by a genuinely-corrupt block.

    Keyframes that physically reside inside a damaged MSZIP block can decode
    to garbage (e.g. a 7e+32 matrix entry on the root bone, which would blow
    the whole character's transform to infinity for that frame). For each
    matrix key (type 3 or 4), any entry that is non-finite or absurdly large
    is replaced with the same entry from the previous keyframe, or the
    identity matrix for the first frame. The caller gates this on detected
    MSZIP corruption, so healthy files are never altered. Returns the number
    of entries clamped.
    """
    ident = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    fixed = 0

    def walk(n):
        nonlocal fixed
        if n.kind == "AnimationKey":
            vals = n.values
            try:
                ktype = int(float(vals[0][1]))
                kcount = int(float(vals[1][1]))
            except (IndexError, ValueError):
                ktype = kcount = -1
            if ktype in (3, 4):
                i = 2
                prev = None
                for _ in range(kcount):
                    if i + 1 >= len(vals):
                        break
                    try:
                        nv = int(float(vals[i + 1][1]))
                    except (ValueError, IndexError):
                        break
                    fstart = i + 2
                    if fstart + nv > len(vals):
                        break
                    if nv == 16:
                        cur = [0.0] * 16
                        for j in range(16):
                            try:
                                f = float(vals[fstart + j][1])
                            except (ValueError, IndexError):
                                f = float("nan")
                            if not math.isfinite(f) or abs(f) > 1.0e6:
                                f = prev[j] if prev is not None else ident[j]
                                vals[fstart + j] = (TOK_NUM, repr(f))
                                fixed += 1
                            cur[j] = f
                        prev = cur
                    i = fstart + nv
        for c in n.children:
            walk(c)

    walk(root)
    return fixed


# =============================================================================

# Shared XNode value-builder helpers (used by .x mesh splitting too).
def _num(v) -> tuple:
    return (TOK_NUM, repr(float(v)))


def _str(s: str) -> tuple:
    return (TOK_STR, s)


def _word(w: str) -> tuple:
    return (TOK_WORD, w)




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

    Single-material meshes (the common case: Watermelon, Apple, Beffica, etc.) are left UNCHANGED.

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

    Some cache formats are already all-triangles, but .x text files
    often store the original quad topology (e.g. BalloonLow.x is
    mostly quads). This converts everything to triangles for downstream
    consistency.

    Triangulation strategy: for a face (v0, v1, ..., vN), emit a fan
    starting from v0: (v0, v1, v2), (v0, v2, v3), ..., (v0, vN-1, vN).
    This is the simple convex-fan split — fine for quads from common tools
    exports (which are guaranteed convex by the authoring tool).

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
    """Parse a .x file and return its XNode tree.

    Auto-detects the DirectX .x sub-format by magic bytes:
      * "xof " (offset 0)            → DirectX .x (text/binary/compressed)

    `split_submeshes`: when a Mesh has a MeshMaterialList assigning DIFFERENT
    materials to different faces, split it into N sibling Mesh nodes — one per
    distinct material. Single-material Mesh nodes are not split.

    `triangulate_quads` converts quad and n-gon faces into pairs of triangles.
    """
    with open(filepath, "rb") as fh:
        raw = fh.read()

    if len(raw) < 16:
        raise ValueError("File too short to contain a valid header")

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

    mszip_corrupted = False
    if is_compressed:
        start = 16
        if len(raw) > 18:
            potential_magic = struct.unpack_from('H', raw, start + 2)[0]
            if potential_magic != _MSZIP_MAGIC:
                start += 6
        try:
            payload = _mszip_decompress(raw, start)
        except ValueError as exc:
            # Strict decode hit a corrupt block or bad framing. Rather
            # than fail the whole import, retry in salvage mode: decode
            # every healthy block and replace damaged ones with filler,
            # recovering the bulk of a file with localized on-disk
            # corruption. Warn so the caller knows the result is partial.
            payload = _mszip_decompress(raw, start, salvage=True)
            info = _mszip_decompress.last_salvage
            lenient = info.get("lenient_blocks", [])
            lost = info["bad_blocks"]
            if lost or lenient or info["resyncs"]:
                mszip_corrupted = True
                recovered = info["total_blocks"] - len(lost)
                if lost:
                    warnings.warn(
                        "X: MSZIP stream is corrupt; salvaged "
                        f"{recovered}/{info['total_blocks']} blocks "
                        f"(tolerant-decoded: {lenient}, lost/zero-filled: "
                        f"{lost}, resyncs: {info['resyncs']}). "
                        "Geometry/animation in the lost region may be "
                        "missing or wrong.",
                        stacklevel=2,
                    )
                else:
                    # Every damaged block was recovered by the tolerant
                    # inflate; data is intact apart from any bytes inside a
                    # genuinely-corrupt block, which are reconstructed as
                    # closely as the stream allows.
                    warnings.warn(
                        "X: MSZIP stream had "
                        f"{len(lenient)} damaged block(s) {lenient} that were "
                        "recovered via a tolerant decoder; "
                        f"{recovered}/{info['total_blocks']} blocks decoded. "
                        "Animation in the damaged region may be slightly "
                        "imperfect.",
                        stacklevel=2,
                    )
    else:
        payload = raw[16:]

    if is_binary:
        try:
            root = _BinaryParser(payload, float_size).parse_file()
        except struct.error as exc:
            raise ValueError(
                f"Truncated or corrupt DirectX binary .x file: {exc}"
            ) from exc
        # If MSZIP salvage zero-filled a damaged block, the sequential token
        # parser may have desynced inside the AnimationSet and dropped most
        # channels. Re-scan the payload and rebuild the channel list from
        # independently-aligned headers. Gated on detected corruption so
        # healthy files are byte-for-byte unaffected.
        if mszip_corrupted:
            try:
                _rescue_binary_animations(payload, float_size, root)
            except Exception:
                # Rescue is best-effort; never let it abort an import that
                # already produced geometry + a partial skeleton.
                pass
            try:
                # Clamp any garbage matrix values left by the damaged block
                # so a single corrupt frame can't blow a bone to infinity.
                _sanitize_anim_keys(root)
            except Exception:
                pass
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
