"""
DirectX .x text-format parser
==============================
Produces a lightweight tree of XNode objects that the importer walks.

Handles the full Burger.x grammar:
  xof 0303txt 0032
  AnimTicksPerSecond { N; }
  Material name { diffuse;; shininess; specular;; emissive;; TextureFileName { "..."; } }
  Frame name { FrameTransformMatrix { 16 floats;; } Frame ... Mesh ... }
  Mesh name { vcount; verts;; facecount; faces;; MeshNormals MeshTextureCoords
              MeshMaterialList XSkinMeshHeader SkinWeights ... }
  AnimationSet name { Animation { { BoneName } AnimationKey { type; count; data;; } } }
"""

import re


# ─────────────────────────────────────────────────────────────
#  Token types
# ─────────────────────────────────────────────────────────────
TOK_WORD   = "WORD"     # identifier / keyword
TOK_STR    = "STR"      # "quoted string"
TOK_NUM    = "NUM"      # int or float
TOK_LBRACE = "{"
TOK_RBRACE = "}"
TOK_SEMI   = ";"
TOK_COMMA  = ","
TOK_EOF    = "EOF"

_RE_TOKEN = re.compile(
    r'\"([^\"]*)\"|'          # quoted string
    r'(//[^\n]*)|'            # line comment  (skip)
    r'([{};,])|'              # single-char punctuation
    r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)|'   # number
    r'([A-Za-z_][A-Za-z0-9_.]*)',                      # identifier
)


def tokenize(text):
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


# ─────────────────────────────────────────────────────────────
#  AST node
# ─────────────────────────────────────────────────────────────
class XNode:
    __slots__ = ("kind", "name", "children", "values")

    def __init__(self, kind, name=""):
        self.kind     = kind   # str  – block keyword
        self.name     = name   # str  – optional identifier after keyword
        self.children = []     # list[XNode]
        self.values   = []     # list of raw tokens (TOK_NUM / TOK_STR / …)

    def child(self, *kinds):
        """Return first child whose kind matches any of *kinds*, or None."""
        for c in self.children:
            if c.kind in kinds:
                return c
        return None

    def children_of(self, kind):
        return [c for c in self.children if c.kind == kind]

    def nums(self):
        """All numeric values as floats."""
        return [float(v) for t, v in self.values if t == TOK_NUM]

    def ints(self):
        return [int(float(v)) for t, v in self.values if t == TOK_NUM]

    def strings(self):
        return [v for t, v in self.values if t == TOK_STR]

    def __repr__(self):
        return f"<XNode {self.kind!r} name={self.name!r} vals={len(self.values)} children={len(self.children)}>"


# ─────────────────────────────────────────────────────────────
#  Parser
# ─────────────────────────────────────────────────────────────
class Parser:
    def __init__(self, tokens):
        self._tok = tokens
        self._pos = 0

    # ── primitives ──────────────────────────────────────────
    def peek(self):
        return self._tok[self._pos]

    def consume(self):
        t = self._tok[self._pos]
        self._pos += 1
        return t

    def expect(self, ttype):
        t = self.consume()
        if t[0] != ttype:
            raise SyntaxError(f"Expected {ttype!r} got {t!r} at token {self._pos}")
        return t

    def maybe(self, ttype):
        if self._tok[self._pos][0] == ttype:
            return self.consume()
        return None

    # ── top-level ───────────────────────────────────────────
    def parse_file(self):
        root = XNode("ROOT", "")
        # skip the "xof 0303txt 0032" header tokens
        while self.peek()[0] == TOK_WORD and self.peek()[1].startswith("xof"):
            self.consume()
        # skip rest of header line tokens (version, format, float-size)
        for _ in range(3):
            if self.peek()[0] in (TOK_WORD, TOK_NUM):
                self.consume()

        while self.peek()[0] != TOK_EOF:
            n = self._parse_block()
            if n:
                root.children.append(n)
        return root

    def _parse_block(self):
        t = self.peek()

        # reference  { BoneName }
        if t[0] == TOK_LBRACE:
            self.consume()
            ref = XNode("REF")
            while self.peek()[0] != TOK_RBRACE and self.peek()[0] != TOK_EOF:
                ref.values.append(self.consume())
            self.maybe(TOK_RBRACE)
            return ref

        # stray semicolons / commas between data items
        if t[0] in (TOK_SEMI, TOK_COMMA):
            self.consume()
            return None

        # a named or anonymous block: KEYWORD [name] { ... }
        if t[0] == TOK_WORD:
            kind = self.consume()[1]
            # optional name
            name = ""
            if self.peek()[0] == TOK_WORD:
                name = self.consume()[1]
            # must be followed by {
            if self.peek()[0] != TOK_LBRACE:
                # bare word / value token that slipped through
                return None
            self.consume()  # eat {
            node = XNode(kind, name)
            self._fill_block(node)
            self.maybe(TOK_RBRACE)
            return node

        # orphan numeric/string token – skip
        self.consume()
        return None

    def _fill_block(self, node):
        """Read tokens inside { } into node.values / node.children."""
        while True:
            t = self.peek()
            if t[0] in (TOK_RBRACE, TOK_EOF):
                return
            # nested block or reference
            if t[0] == TOK_WORD:
                # peek-ahead: is it a keyword/block or just data?
                # heuristic: if next-next token is '{' or a name then a name then '{'
                p1 = self._tok[self._pos + 1][0] if self._pos + 1 < len(self._tok) else TOK_EOF
                p2 = self._tok[self._pos + 2][0] if self._pos + 2 < len(self._tok) else TOK_EOF
                if p1 == TOK_LBRACE or p1 == TOK_WORD:
                    child = self._parse_block()
                    if child:
                        node.children.append(child)
                    continue
                # otherwise it's a data token (bone name in SkinWeights etc.)
                node.values.append(self.consume())
                continue
            if t[0] == TOK_LBRACE:
                child = self._parse_block()
                if child:
                    node.children.append(child)
                continue
            # numeric / string / punctuation data
            if t[0] in (TOK_NUM, TOK_STR, TOK_SEMI, TOK_COMMA):
                node.values.append(self.consume())
                continue
            # anything else – skip
            self.consume()


def parse_x_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    tokens = tokenize(text)
    return Parser(tokens).parse_file()
