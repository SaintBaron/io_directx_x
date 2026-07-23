"""
Blender  ──►  DirectX .x  exporter

Round-trips with the importer: importing Burger.x then exporting should
produce a file equivalent in content to the original.

Key properties matched with the importer:
  • Materials are top-level (not nested in frames).
  • Meshes live inside a sibling Frame of the skeleton root, not as a bone child.
  • Vertices are exported in DX space (inverse of the importer's conv_mat).
  • Normals are rotated back to DX space (transpose of conv_mat).
  • UVs are stored with V negated (the importer does 1.0 - v on read; the
    original file uses negative V values, so we preserve that here).
  • Faces can stay as N-gons (the importer handles quads and tris).
  • SkinWeights offset matrix is inv(bind_pose) in DX world space.
  • Animations export absolute DX-space quaternions (not Blender pose deltas)
    with LH-convention sign flips so the importer's conjugate round-trips.

Compatible with Blender <=3.x, 4.x, and 5.x:
  • Uses me.corner_normals (4.1+) with fallback to calc_normals_split().
  • Uses me.attributes["sharp_edge"] or edge.use_edge_sharp as appropriate.
  • Handles Blender 5.x layered action API transparently (we read pose_bone
    evaluated state, not F-curves directly, so the API change is irrelevant).
"""

import os
import re
import bpy
import math
import struct
import zlib
import bmesh
from mathutils import Matrix, Vector

_BT_NAME      = 0x0001
_BT_STRING    = 0x0002
_BT_INT_LIST  = 0x0006
_BT_FLT_LIST  = 0x0007
_BT_OBRACE    = 0x000a
_BT_CBRACE    = 0x000b
_BT_COMMA     = 0x0013
_BT_SEMICOLON = 0x0014

# Standard DirectX .x template declarations. Many DirectX consumers
# (including some games's engine and some engines' loaders) expect these
# templates to be declared at the top of every text-format .x file —
# they describe the layout of subsequent data nodes. When we have a
# stashed template block from an importer round-trip we use that;
# otherwise we fall back to this canonical set covering everything our
# exporter can emit.
_DEFAULT_TEMPLATES = """template ColorRGBA {
 <35ff44e0-6c7c-11cf-8f52-0040333594a3>
 FLOAT red;
 FLOAT green;
 FLOAT blue;
 FLOAT alpha;
}

template ColorRGB {
 <d3e16e81-7835-11cf-8f52-0040333594a3>
 FLOAT red;
 FLOAT green;
 FLOAT blue;
}

template IndexedColor {
 <1630b820-7842-11cf-8f52-0040333594a3>
 DWORD index;
 ColorRGBA indexColor;
}

template Material {
 <3d82ab4d-62da-11cf-ab39-0020af71e433>
 ColorRGBA faceColor;
 FLOAT power;
 ColorRGB specularColor;
 ColorRGB emissiveColor;
 [...]
}

template TextureFilename {
 <a42790e1-7810-11cf-8f52-0040333594a3>
 STRING filename;
}

template Coords2d {
 <f6f23f44-7686-11cf-8f52-0040333594a3>
 FLOAT u;
 FLOAT v;
}

template Vector {
 <3d82ab5e-62da-11cf-ab39-0020af71e433>
 FLOAT x;
 FLOAT y;
 FLOAT z;
}

template Matrix4x4 {
 <f6f23f45-7686-11cf-8f52-0040333594a3>
 array FLOAT matrix[16];
}

template FrameTransformMatrix {
 <f6f23f41-7686-11cf-8f52-0040333594a3>
 Matrix4x4 frameMatrix;
}

template Frame {
 <3d82ab46-62da-11cf-ab39-0020af71e433>
 [...]
}

template MeshFace {
 <3d82ab5f-62da-11cf-ab39-0020af71e433>
 DWORD nFaceVertexIndices;
 array DWORD faceVertexIndices[nFaceVertexIndices];
}

template Mesh {
 <3d82ab44-62da-11cf-ab39-0020af71e433>
 DWORD nVertices;
 array Vector vertices[nVertices];
 DWORD nFaces;
 array MeshFace faces[nFaces];
 [...]
}

template MeshNormals {
 <f6f23f43-7686-11cf-8f52-0040333594a3>
 DWORD nNormals;
 array Vector normals[nNormals];
 DWORD nFaceNormals;
 array MeshFace faceNormals[nFaceNormals];
}

template MeshTextureCoords {
 <f6f23f40-7686-11cf-8f52-0040333594a3>
 DWORD nTextureCoords;
 array Coords2d textureCoords[nTextureCoords];
}

template MeshVertexColors {
 <1630b821-7842-11cf-8f52-0040333594a3>
 DWORD nVertexColors;
 array IndexedColor vertexColors[nVertexColors];
}

template MeshMaterialList {
 <f6f23f42-7686-11cf-8f52-0040333594a3>
 DWORD nMaterials;
 DWORD nFaceIndexes;
 array DWORD faceIndexes[nFaceIndexes];
 [Material <3d82ab4d-62da-11cf-ab39-0020af71e433>]
}

template VertexElement {
 <f752461c-1e23-48f6-b9f8-8350850f336f>
 DWORD Type;
 DWORD Method;
 DWORD Usage;
 DWORD UsageIndex;
}

template DeclData {
 <bf22e553-292c-4781-9fea-62bd554bdd93>
 DWORD nElements;
 array VertexElement Elements[nElements];
 DWORD nDWords;
 array DWORD data[nDWords];
}

template FVFData {
 <b6e70a0e-8ef9-4e83-94ad-ecc8b0c04897>
 DWORD dwFVF;
 DWORD nDWords;
 array DWORD data[nDWords];
}

template VertexDuplicationIndices {
 <b8d65549-d7c9-4995-89cf-53a9a8b031e3>
 DWORD nIndices;
 DWORD nOriginalVertices;
 array DWORD indices[nIndices];
}

template XSkinMeshHeader {
 <3cf169ce-ff7c-44ab-93c0-f78f62d172e2>
 WORD nMaxSkinWeightsPerVertex;
 WORD nMaxSkinWeightsPerFace;
 WORD nBones;
}

template SkinWeights {
 <6f0d123b-bad2-4167-a0d0-80224f25fabb>
 STRING transformNodeName;
 DWORD nWeights;
 array DWORD vertexIndices[nWeights];
 array FLOAT weights[nWeights];
 Matrix4x4 matrixOffset;
}

template AnimTicksPerSecond {
 <9e415a43-7ba6-4a73-8743-b73d47e88476>
 DWORD AnimTicksPerSecond;
}

template FloatKeys {
 <10dd46a9-775b-11cf-8f52-0040333594a3>
 DWORD nValues;
 array FLOAT values[nValues];
}

template TimedFloatKeys {
 <f406b180-7b3b-11cf-8f52-0040333594a3>
 DWORD time;
 FloatKeys tfkeys;
}

template AnimationKey {
 <10dd46a8-775b-11cf-8f52-0040333594a3>
 DWORD keyType;
 DWORD nKeys;
 array TimedFloatKeys keys[nKeys];
}

template AnimationOptions {
 <e2bf56c0-840f-11cf-8f52-0040333594a3>
 DWORD openclosed;
 DWORD positionquality;
}

template Animation {
 <3d82ab4f-62da-11cf-ab39-0020af71e433>
 [...]
}

template AnimationSet {
 <3d82ab50-62da-11cf-ab39-0020af71e433>
 [Animation <3d82ab4f-62da-11cf-ab39-0020af71e433>]
}"""

_TEMPLATE_DECL_RE = re.compile(r'template\s+([A-Za-z_][A-Za-z0-9_]*)')

def _split_default_templates():
    """Return the canonical template block as an ordered {name: text} map."""
    chunks = {}
    starts = [m for m in _TEMPLATE_DECL_RE.finditer(_DEFAULT_TEMPLATES)]
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(_DEFAULT_TEMPLATES)
        chunks[m.group(1)] = _DEFAULT_TEMPLATES[m.start():end].rstrip()
    return chunks

def _assemble_templates(passthrough_text=""):
    """Build the template block to write at the top of an export.

    Starts from the stashed source-file templates (if any) so round-trips
    stay faithful, then appends any canonical template the stash lacks.
    Dependency order is preserved: the stash keeps its own order, and the
    canonical set is already declared dependency-first, so appending the
    missing subset in canonical order keeps every referenced type declared
    before use for legacy RegisterTemplates-style validators.
    """
    passthrough_text = (passthrough_text or "").strip()
    have = set(_TEMPLATE_DECL_RE.findall(passthrough_text))
    missing = [txt for name, txt in _split_default_templates().items()
               if name not in have]
    parts = []
    if passthrough_text:
        parts.append(passthrough_text)
    parts.extend(missing)
    return "\n\n".join(parts)


class _NameSanitizer:
    """Map arbitrary Blender names to legal, unique .x identifiers.

    Text-format .x names must match [A-Za-z_][A-Za-z0-9_]* — Blender's
    default names ('Cube.001', 'Material.001', action names with '|' or
    spaces) are illegal and break strict parsers (and even D3DX). Every
    name-emission site shares one instance of this class so a given
    source name always maps to the same output identifier, keeping
    Frame names, SkinWeights transformNodeName strings and { Name }
    references mutually consistent.
    """

    _BAD = re.compile(r'[^A-Za-z0-9_]')

    def __init__(self):
        self._map = {}
        self._used = set()

    def get(self, name, fallback="unnamed"):
        key = "" if name is None else str(name)
        if key in self._map:
            return self._map[key]
        s = self._BAD.sub('_', key) or fallback
        if s[0].isdigit():
            s = '_' + s
        base, n = s, 1
        while s in self._used:
            s = f"{base}_{n}"
            n += 1
        self._used.add(s)
        self._map[key] = s
        return s


class _BinarySerializer:
    """Convert the text-oriented write stream to DirectX binary tokens."""

    _RE = re.compile(
        r'"([^"]*)"'
        r'|([{};,])'
        r'|([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        r'|([A-Za-z_][A-Za-z0-9_.]*)',
    )

    def __init__(self):
        self._buf   = bytearray()

        self._pending_floats : list[float] = []
        self._pending_ints   : list[int]   = []
        self._in_float_ctx   = False

    @staticmethod
    def _strip_comments(text: str) -> str:
        """Remove text-format comments ('//' and '#' to end of line)
        outside quoted strings. The token regex has no comment rule, so
        without this, comment words (e.g. slender.x's '//End of ...'
        annotations inside stashed passthrough blocks) would be emitted
        as bogus NAME tokens and corrupt the binary stream.
        """
        if '//' not in text and '#' not in text:
            return text
        out = []
        in_str = False
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            if in_str:
                out.append(c)
                if c == '"':
                    in_str = False
                i += 1
                continue
            if c == '"':
                in_str = True
                out.append(c)
                i += 1
                continue
            if c == '#' or (c == '/' and i + 1 < n and text[i + 1] == '/'):
                while i < n and text[i] != '\n':
                    i += 1
                continue
            out.append(c)
            i += 1
        return ''.join(out)

    def feed(self, text: str):
        text = self._strip_comments(text)
        for m in self._RE.finditer(text):
            qstr, punc, num, word = m.groups()
            if qstr is not None:
                self._flush_pending()
                self._emit_string(qstr)
            elif punc is not None:
                if punc == '{':
                    self._flush_pending()
                    self._emit_u16(_BT_OBRACE)
                elif punc == '}':
                    self._flush_pending()
                    self._emit_u16(_BT_CBRACE)
                elif punc == ';':

                    pass
                elif punc == ',':
                    pass
            elif num is not None:

                is_plain_int = ('.' not in num and 'e' not in num.lower())
                if is_plain_int:
                    v = int(num)
                    self._pending_ints.append(v)
                    if self._pending_floats:
                        self._flush_floats()
                else:
                    v = float(num)
                    self._pending_floats.append(v)
                    if self._pending_ints:
                        self._flush_ints()
            elif word is not None:
                self._flush_pending()
                self._emit_name(word)

    def getvalue(self) -> bytes:
        self._flush_pending()
        return bytes(self._buf)

    # ---- binary template declarations -----------------------------------
    # Legacy IDirectXFile consumers (FPSC/DBPro's ConvX) fail the whole
    # load when a data object's template is neither pre-registered nor
    # declared in-file, in binary exactly as in text. Templates cannot go
    # through feed(): the text tokenizer would turn 'template', member
    # type keywords and <GUID>s into bogus NAME tokens. This emits the
    # proper token forms, verified 1:1 against the template sections of
    # native binary files (Virtual Sailor's boat.x):
    #   TEMPLATE NAME(name) OBRACE GUID
    #     primitive member : TYPE_TOKEN NAME(member) SEMICOLON
    #     named-type member: NAME(type) NAME(member) SEMICOLON
    #     array member     : ARRAY TYPE NAME(member)
    #                        { OBRACKET (INTEGER n | NAME dim) CBRACKET }+
    #                        SEMICOLON
    #     open class       : OBRACKET DOT DOT DOT CBRACKET
    #     restricted class : OBRACKET NAME(type) [GUID] CBRACKET
    #   CBRACE
    _BT_TEMPLATE = 0x001f
    _BT_GUID     = 0x0005
    _BT_INTEGER  = 0x0003
    _BT_OBRACKET = 0x000e
    _BT_CBRACKET = 0x000f
    _BT_DOT      = 0x0012
    _BT_ARRAY    = 0x0034
    _PRIM_TOKENS = {
        "WORD": 0x0028, "DWORD": 0x0029, "FLOAT": 0x002a, "DOUBLE": 0x002b,
        "CHAR": 0x002c, "UCHAR": 0x002d, "SWORD": 0x002e, "SDWORD": 0x002f,
        # The text keyword STRING maps to TOKEN_LPSTR in binary form.
        "STRING": 0x0031, "LPSTR": 0x0031, "UNICODE": 0x0032,
        "CSTRING": 0x0033,
    }
    _TPL_RE   = re.compile(
        r'template\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*'
        r'<([0-9a-fA-F-]{36})>\s*(.*?)\}',
        re.DOTALL)
    _GUID_RE  = re.compile(r'<([0-9a-fA-F-]{36})>')
    _ARRAY_RE = re.compile(
        r'array\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)'
        r'((?:\s*\[\s*[A-Za-z0-9_]+\s*\])+)')
    _MEMBER_RE = re.compile(
        r'([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)')
    _DIM_RE   = re.compile(r'\[\s*([A-Za-z0-9_]+)\s*\]')

    def _emit_guid(self, guid_text: str):
        h = guid_text.replace('-', '')
        d1 = int(h[0:8], 16)
        d2 = int(h[8:12], 16)
        d3 = int(h[12:16], 16)
        self._emit_u16(self._BT_GUID)
        self._buf += struct.pack('<IHH', d1, d2, d3)
        self._buf += bytes.fromhex(h[16:32])

    def emit_templates(self, templates_text: str):
        templates_text = self._strip_comments(templates_text)
        for m in self._TPL_RE.finditer(templates_text):
            name, guid, body = m.group(1), m.group(2), m.group(3)
            self._emit_u16(self._BT_TEMPLATE)
            self._emit_name(name)
            self._emit_u16(_BT_OBRACE)
            self._emit_guid(guid)
            for raw in body.split(';'):
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('['):
                    inner = line.strip('[]').strip()
                    self._emit_u16(self._BT_OBRACKET)
                    if inner in ('...', '. . .'):
                        for _ in range(3):
                            self._emit_u16(self._BT_DOT)
                    else:
                        gm = self._GUID_RE.search(inner)
                        tname = inner.split('<')[0].strip()
                        self._emit_name(tname)
                        if gm:
                            self._emit_guid(gm.group(1))
                    self._emit_u16(self._BT_CBRACKET)
                    continue    # class restrictions carry no semicolon
                am = self._ARRAY_RE.match(line)
                if am:
                    a_type, a_name, dims = am.group(1), am.group(2), am.group(3)
                    self._emit_u16(self._BT_ARRAY)
                    prim = self._PRIM_TOKENS.get(a_type)
                    if prim is not None:
                        self._emit_u16(prim)
                    else:
                        self._emit_name(a_type)
                    self._emit_name(a_name)
                    for dm in self._DIM_RE.finditer(dims):
                        d = dm.group(1)
                        self._emit_u16(self._BT_OBRACKET)
                        if d.isdigit():
                            self._emit_u16(self._BT_INTEGER)
                            self._emit_u32(int(d))
                        else:
                            self._emit_name(d)
                        self._emit_u16(self._BT_CBRACKET)
                    self._emit_u16(_BT_SEMICOLON)
                    continue
                mm = self._MEMBER_RE.match(line)
                if mm:
                    m_type, m_name = mm.group(1), mm.group(2)
                    prim = self._PRIM_TOKENS.get(m_type)
                    if prim is not None:
                        self._emit_u16(prim)
                    else:
                        self._emit_name(m_type)
                    self._emit_name(m_name)
                    self._emit_u16(_BT_SEMICOLON)
            self._emit_u16(_BT_CBRACE)


    def _emit_u16(self, v: int):
        self._buf += struct.pack('<H', v)

    def _emit_u32(self, v: int):
        self._buf += struct.pack('<I', v)

    def _emit_f32(self, v: float):
        self._buf += struct.pack('<f', v)

    def _emit_name(self, s: str):
        enc = s.encode('latin-1')
        self._emit_u16(_BT_NAME)
        self._emit_u32(len(enc))
        self._buf += enc

    def _emit_string(self, s: str):
        enc = s.encode('latin-1')
        self._emit_u16(_BT_STRING)
        self._emit_u32(len(enc))
        self._buf += enc
        # Every STRING in the binary format ends with a terminator TOKEN
        # (TOKEN_SEMICOLON or TOKEN_COMMA). Real files use SEMICOLON;
        # a zero word here is an invalid token that makes D3DX-family
        # readers reject the whole file.
        self._emit_u16(_BT_SEMICOLON)

    def _flush_ints(self):
        if not self._pending_ints:
            return
        self._emit_u16(_BT_INT_LIST)
        self._emit_u32(len(self._pending_ints))
        for v in self._pending_ints:
            # Two's-complement wrap: DWORD fields are unsigned on disk,
            # but signed values can legitimately appear (e.g. clamped
            # negative animation ticks from odd source files).
            self._emit_u32(v & 0xFFFFFFFF)
        self._pending_ints = []

    def _flush_floats(self):
        if not self._pending_floats:
            return
        self._emit_u16(_BT_FLT_LIST)
        self._emit_u32(len(self._pending_floats))
        for v in self._pending_floats:
            self._emit_f32(v)
        self._pending_floats = []

    def _flush_pending(self):

        if self._pending_ints and self._pending_floats:
            self._flush_ints()
            self._flush_floats()
        elif self._pending_ints:
            self._flush_ints()
        elif self._pending_floats:
            self._flush_floats()

def _mat4_to_dx(mat):
    t = mat.transposed()
    return ",".join(f"{t[r][c]:.6f}" for r in range(4) for c in range(4))

def _axis_matrix(axis_forward, axis_up):
    import numpy as np
    _AXES = {'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0),'Z':(0,0,1),'-Z':(0,0,-1)}
    fwd = np.array(_AXES[axis_forward], float)
    upv = np.array(_AXES[axis_up],      float)
    rgt = np.cross(fwd, upv)

    F   = np.column_stack([rgt, upv, fwd])
    B   = np.column_stack([(1,0,0),(0,0,1),(0,1,0)])
    M3  = F @ np.linalg.inv(B)
    return Matrix([[float(M3[r,c]) for c in range(3)] + [0] for r in range(3)] + [[0,0,0,1]])

def _get_principled(mat):
    if not mat or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _effective_materials(obj):
    """Return the list of materials assigned to a mesh object, falling
    back to the mesh's data-block materials when the object has no
    material slots. This handles the case where materials live on the
    mesh datablock (the default for our importer) but obj.material_slots
    appears empty — which can happen when the object's slot link mode
    diverges, or in older Blender versions where the slot list is lazy."""
    slot_mats = [s.material for s in obj.material_slots if s.material]
    if slot_mats:
        return slot_mats
    # Fall back to the mesh data block.
    me = getattr(obj, "data", None)
    if me is not None:
        return [m for m in getattr(me, "materials", []) if m]
    return []

def _tex_path(mat):
    if not mat or not mat.use_nodes:
        return ""
    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            return node.image.filepath_from_user()
    return ""

def _get_corner_normals(me):
    if hasattr(me, "corner_normals") and len(me.corner_normals) > 0:

        return [tuple(cn.vector) for cn in me.corner_normals]
    if hasattr(me, "calc_normals_split"):

        me.calc_normals_split()
        return [tuple(l.normal) for l in me.loops]

    return [tuple(me.polygons[me.loops[li].polygon_index].normal) if hasattr(me.loops[li], 'polygon_index')
            else (0.0, 0.0, 1.0) for li in range(len(me.loops))]

def export_x(context, filepath,
             use_selection=False,
             use_mesh_modifiers=True,
             global_scale=1.0,
             axis_forward="-Z",
             axis_up="Y",
             export_normals=True,
             export_uvs=True,
             export_materials=True,
             export_textures=True,
             export_armature=True,
             export_weights=True,
             export_animation=True,
             unweld_on_export=True,
             anim_fps=30.0,
             anim_frame_start=1,
             anim_frame_end=250,
             triangulate=False,
             binary_format=False,
             compress=False,
             pz_compat=False,
             write_templates=True,
             inline_materials=False,
             fill_missing_uvs=False,
             anim_key_format="TRS",
             **_):

    scene     = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    objects   = context.selected_objects if use_selection else list(context.scene.objects)

    # One shared name registry for the whole export: every emitted
    # identifier (Frame/Mesh/Material/AnimationSet names, { Ref }
    # targets and SkinWeights transformNodeName strings) goes through
    # it, so a given Blender name always maps to the same legal, unique
    # .x identifier and cross-references stay consistent.
    _names = _NameSanitizer()

    # Compose with the importer's axis_fix (180° around Z) so a re-import
    axis_fix = Matrix.Rotation(math.pi, 4, 'Z').to_3x3()
    bl_to_dx_3 = _axis_matrix(axis_forward, axis_up).to_3x3() @ axis_fix
    inv_scale  = 1.0 / global_scale if global_scale != 0.0 else 1.0

    if binary_format:
        _ser = _BinarySerializer()
        out  = _ser
        w    = _ser.feed
    else:
        _ser = None
        out  = []
        w    = out.append

    # AnimTicksPerSecond is both the file's declared tick rate AND the
    # target real-time playback rate. To preserve real-time correctness
    # regardless of scene FPS, we scale Blender frame numbers up/down so
    # that the written tick values represent the same wall-clock duration:
    #   file_tick = frame * (anim_ticks_per_second / scene_fps)
    # When scene_fps == anim_fps the scale is 1.0 and ticks equal frames.
    # When pz_compat is on, anim_fps is forced to 4800 (high-precision convention)
    # regardless of what the FPS field says.
    if pz_compat:
        file_ticks_per_second = 4800
    else:
        file_ticks_per_second = int(anim_fps)

    if hasattr(scene.render, "fps_base"):
        scene_fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    else:
        scene_fps = float(scene.render.fps)
    if scene_fps <= 0:
        scene_fps = 30.0
    tick_scale_out = float(file_ticks_per_second) / scene_fps

    # Pull the source-file passthrough text (templates and auxiliary
    # frames like Translation_Data) off the armature, if it was set by
    # the importer. These are re-emitted verbatim so a load → save round
    # trip preserves the original file layout for blocks the exporter
    # doesn't natively reproduce.
    arm_objs_for_passthrough = [o for o in objects if o.type == "ARMATURE"]
    _passthrough_templates = ""
    _passthrough_aux_frames = ""
    _passthrough_aux_animations = ""
    if arm_objs_for_passthrough:
        _ad = arm_objs_for_passthrough[0].data
        try:
            _passthrough_templates       = str(_ad.get("_x_templates", "") or "")
            _passthrough_aux_frames      = str(_ad.get("_x_aux_frames", "") or "")
            _passthrough_aux_animations  = str(_ad.get("_x_aux_animations", "") or "")
        except Exception:
            pass

    # AnimTicksPerSecond is only meaningful when the file will actually
    # carry an AnimationSet. Emitting it on static meshes breaks strict
    # legacy loaders (FPSC's IDirectXFile registers only the D3DRM
    # template set, which predates AnimTicksPerSecond) and no known
    # engine needs it without animation — none of the native Virtual
    # Sailor files carry it at all.
    _will_write_animation = bool(export_animation and arm_objs_for_passthrough)

    # Assemble the template block to declare in-file. Strict template-
    # driven consumers (FPSC/DBPro's legacy IDirectXFile converter,
    # ModelConverterX, Virtual Sailor NG) hard-fail on any data object
    # whose template is neither built in nor declared in the file, so by
    # default every export declares the full canonical set. A stashed
    # source-file copy is preferred for round-trip fidelity, topped up
    # with any canonical templates it lacks for objects we may emit.
    _templates_out = ""
    if write_templates or pz_compat:
        _templates_out = _assemble_templates(_passthrough_templates)
    elif _passthrough_templates:
        _templates_out = _passthrough_templates

    if binary_format:

        _bin_header = b"xof 0303bin 0032"

        if _templates_out:
            _ser.emit_templates(_templates_out)
        if _will_write_animation:
            w(f"AnimTicksPerSecond {{ {file_ticks_per_second}; }}\n")
    else:
        w("xof 0303txt 0032\n\n")
        if _templates_out:
            w(_templates_out)
            w("\n\n")
        if _will_write_animation:
            w(f"AnimTicksPerSecond {{\n\t{file_ticks_per_second};\n}}\n")

    mesh_objs     = [o for o in objects if o.type == "MESH"]
    armature_objs = [o for o in objects if o.type == "ARMATURE"]
    arm_obj       = armature_objs[0] if armature_objs else None

    written_mats = {}
    if export_materials:
        if not inline_materials:
            for obj in mesh_objs:
                for mat in _effective_materials(obj):
                    if not mat or mat.name in written_mats:
                        continue
                    written_mats[mat.name] = mat
                    _write_material_block(w, mat, "", _names.get(mat.name),
                                          export_textures, filepath)

    # Meshes parented to a BONE (FPSC weapon attachments like Colonel-X's
    # FIRESPOT gun; HUD gun parts) must be written NESTED inside that
    # bone's Frame with a bone-relative FTM. Writing them as top-level
    # frames — worse, with an Identity FTM, since obj.parent == arm_obj —
    # dropped both the attachment hierarchy and the placement, so the
    # re-exported prop sat at the origin and never followed the hand.
    _bone_child_meshes = {}
    _bone_nested_ids = set()
    _nested_mesh_warnings = []
    _nested_temp_cleanup = []
    if arm_obj is not None:
        for _bobj in mesh_objs:
            if (_bobj.parent == arm_obj
                    and getattr(_bobj, "parent_type", "") == "BONE"
                    and getattr(_bobj, "parent_bone", "")):
                _bone_child_meshes.setdefault(
                    _bobj.parent_bone, []).append(_bobj)
                _bone_nested_ids.add(id(_bobj))

    if export_armature and arm_obj:
        arm_data = arm_obj.data

        def write_bone(bone, indent):
            ind = "\t" * indent

            stashed = arm_data.get(f"_x_ftm:{bone.name}")
            if stashed is not None and len(stashed) >= 16:
                ftm_string = ",".join(f"{float(v):.6f}" for v in stashed[:16])
            else:
                if bone.parent:
                    local_mat = bone.parent.matrix_local.inverted() @ bone.matrix_local

                else:
                    # Root bones carry the armature OBJECT transform (when
                    # non-identity) so the exported hierarchy composes to
                    # true world space — matching the offsets and keys.
                    _arm_M = _arm_world_or_none(arm_obj)
                    _root_rest = (_arm_M @ bone.matrix_local
                                  if _arm_M is not None
                                  else bone.matrix_local)
                    local_mat = _bl_bone_to_dx_world(_root_rest, bl_to_dx_3, inv_scale)
                ftm_string = _mat4_to_dx(local_mat)

            w(f"{ind}Frame {_names.get(bone.name)} {{\n")
            w(f"{ind}\tFrameTransformMatrix {{\n")
            w(f"{ind}\t\t{ftm_string};;\n")
            w(f"{ind}\t}}\n")
            # Bone-parented meshes nest here, with an FTM relative to
            # THIS bone. Parent-relative locals are identical in Blender
            # and DX space under the rotation-only axis conversion (the
            # same fact the child-bone FTMs above rely on), so the local
            # is computed purely in Blender space.
            # Bone-nested meshes that were SPLIT from one multi-material
            # source Mesh on import re-merge into a single Mesh node,
            # exactly as the standalone path does — otherwise a
            # multi-material container mesh (kwXport cube) exports as N
            # separate Mesh nodes and the re-import mis-places pieces.
            _pending_nested = list(_bone_child_meshes.get(bone.name, ()))
            _nst_groups = {}
            _nst_order = []
            for _cobj in _pending_nested:
                _src = _cobj.get('_x_split_source_mesh')
                _gkey = ('split', str(_src)) if _src else ('solo', id(_cobj))
                if _gkey not in _nst_groups:
                    _nst_groups[_gkey] = []
                    _nst_order.append(_gkey)
                _nst_groups[_gkey].append(_cobj)
            _emit_nested = []
            for _gkey in _nst_order:
                _grp = _nst_groups[_gkey]
                if _gkey[0] == 'split' and len(_grp) > 1:
                    _grp = sorted(
                        _grp,
                        key=lambda o: int(o.get('_x_split_group_idx', 0)))
                    try:
                        _tmp = _merge_split_group_for_export(_grp, _gkey[1])
                        _nested_temp_cleanup.append(_tmp)
                        _emit_nested.append(_tmp)
                        continue
                    except Exception as _me:
                        _nested_mesh_warnings.append(
                            f"Failed to re-merge bone-nested split group "
                            f"'{_gkey[1]}': {_me}. Writing "
                            f"{len(_grp)} separate Mesh nodes.")
                _emit_nested.extend(_grp)
            for _cobj in _emit_nested:
                try:
                    _arm_inv = arm_obj.matrix_world.inverted()
                except ValueError:
                    _arm_inv = Matrix.Identity(4)
                try:
                    _local = (bone.matrix_local.inverted()
                              @ (_arm_inv @ _cobj.matrix_world))
                except ValueError:
                    _local = Matrix.Identity(4)
                # A mesh whose bone WAS its own container frame on import
                # (stashed by the importer) is written directly inside this
                # bone's Frame with its object-local vertices verbatim —
                # the source layout ('Frame X { FTM  Mesh X {...} }').
                # The old nested holder frame duplicated the bone's name
                # and flipped the re-importer's placement heuristic on
                # some meshes (PZ Wrench/Handgun/scope drift).
                # DeclData-stashed meshes keep the old path: the verbatim
                # DeclData block indexes untransformed vertices.
                _as_container = (
                    str(_cobj.get("_x_container_bone") or "") == bone.name
                    and not _cobj.get("_x_decldata"))
                _nested_mesh_warnings.extend(_write_mesh_frame(
                    _cobj, out, indent + (0 if _as_container else 1),
                    depsgraph, use_mesh_modifiers,
                    export_normals, export_uvs,
                    export_materials, export_weights,
                    arm_obj, bl_to_dx_3, inv_scale,
                    triangulate, written_mats,
                    unweld_on_export=unweld_on_export,
                    names=_names,
                    frame_matrix_override=(None if _as_container else _local),
                    suppress_frame_wrapper=_as_container,
                    inline_materials=inline_materials,
                    fill_missing_uvs=fill_missing_uvs,
                    export_textures=export_textures,
                    filepath=filepath))
            for child in bone.children:
                write_bone(child, indent + 1)
            w(f"{ind}}}\n")

        for rb in (b for b in arm_data.bones if not b.parent):
            write_bone(rb, 0)

    # Delete any temp objects created for bone-nested split-group
    # merges — they were only needed while write_bone emitted them.
    for _tmp in _nested_temp_cleanup:
        try:
            _tmp_data = _tmp.data
            bpy.data.objects.remove(_tmp, do_unlink=True)
            bpy.data.meshes.remove(_tmp_data)
        except Exception:
            pass

    # Re-emit any auxiliary top-level frames (e.g. Translation_Data)
    # captured from the source .x file. These sit between the skeleton
    # and the mesh in the original layout, so that's where we put them.
    if _passthrough_aux_frames and not binary_format:
        w(_passthrough_aux_frames)
        w("\n\n")

    all_warnings = []
    all_warnings.extend(_nested_mesh_warnings)

    # Group mesh objects by _x_split_source_mesh — objects that were
    # split from a single multi-material .x Mesh on import need to be
    # re-merged into ONE Mesh node with a multi-material MeshMaterialList
    # so the round-trip preserves the original .x structure. Groups are
    # only formed for ≥2 objects sharing the same _x_split_source_mesh
    # value; everything else (single objects, ungrouped meshes) writes
    # exactly as before.
    split_groups: dict = {}        # source_name → list of objects
    standalone_objs: list = []
    for obj in mesh_objs:
        if id(obj) in _bone_nested_ids:
            continue          # already written nested under its bone
        src = obj.get('_x_split_source_mesh')
        if src:
            split_groups.setdefault(str(src), []).append(obj)
        else:
            standalone_objs.append(obj)

    # Build the final emission list: standalone objects in their
    # original order, plus one merged temp object per split group.
    # Temporary objects are tracked so we can clean them up after.
    temp_objects_to_cleanup: list = []
    write_order: list = []

    seen_split_sources: set = set()
    for obj in mesh_objs:
        if id(obj) in _bone_nested_ids:
            continue          # already written nested under its bone
        src = obj.get('_x_split_source_mesh')
        if src:
            if src in seen_split_sources:
                continue
            seen_split_sources.add(src)
            group = split_groups[src]
            if len(group) == 1:
                # Group of one — no merge needed, write as-is.
                write_order.append(group[0])
            else:
                # Merge the group into a temp object. Sort group by
                # _x_split_group_idx so material slots come out in
                # the original order.
                group_sorted = sorted(
                    group,
                    key=lambda o: int(o.get('_x_split_group_idx', 0))
                )
                try:
                    temp = _merge_split_group_for_export(group_sorted, src)
                    write_order.append(temp)
                    temp_objects_to_cleanup.append(temp)
                except Exception as e:
                    # Merge failed — fall back to writing each piece
                    # as its own Mesh (the v1.7.27 behaviour). Better
                    # than aborting the export.
                    all_warnings.append(
                        f"Failed to re-merge multi-material .x group "
                        f"'{src}': {e}. Writing as {len(group)} separate "
                        f"Mesh nodes instead."
                    )
                    write_order.extend(group_sorted)
        else:
            write_order.append(obj)

    try:
        for obj in write_order:
            all_warnings.extend(_write_mesh_frame(obj, out, 0,
                              depsgraph, use_mesh_modifiers,
                              export_normals, export_uvs,
                              export_materials, export_weights,
                              arm_obj, bl_to_dx_3, inv_scale,
                              triangulate, written_mats,
                              unweld_on_export=unweld_on_export,
                              names=_names,
                              inline_materials=inline_materials,
                              fill_missing_uvs=fill_missing_uvs,
                              export_textures=export_textures,
                              filepath=filepath))
    finally:
        # Clean up temporary merged objects regardless of success/failure
        for temp in temp_objects_to_cleanup:
            try:
                tmp_data = temp.data
                bpy.data.objects.remove(temp, do_unlink=True)
                if tmp_data and tmp_data.users == 0:
                    bpy.data.meshes.remove(tmp_data)
            except Exception:
                pass

    if export_animation and arm_obj:
        orig_frame = scene.frame_current

        baked = {b.name: {"rot": {}, "scale": {}, "pos": {}, "mat": {}} for b in arm_obj.pose.bones}

        # Collect per-bone keyframe times directly from the F-curves so
        # that a sparse imported animation stays sparse on export — the
        # original high-precision file has e.g. 41 rotation keys but only 2 scale
        # keys, and we want to preserve that pattern. If the user added
        # keyframes in Blender, those will appear in the F-curves and
        # naturally extend the exported key set. If there are no F-curves
        # (a static skeleton), fall back to dense per-frame baking so a
        # constant pose still gets written.
        action = None
        try:
            ad = arm_obj.animation_data
            if ad is not None:
                action = ad.action
        except Exception:
            action = None

        def _fcurve_frames(channels_paths_indices):
            """Collect the union of keyframe x-coords across the given
            (data_path, array_index) pairs. Returns a sorted list of
            int frames, restricted to [anim_frame_start, anim_frame_end]."""
            frames = set()
            if action is None:
                return []
            # Blender 4.4+ may store F-curves under action.layers[].strips[]
            # .channelbags[].fcurves; older versions use action.fcurves
            # directly. Try both.
            fcurves_iters = []
            if hasattr(action, "fcurves") and action.fcurves:
                fcurves_iters.append(action.fcurves)
            if hasattr(action, "layers"):
                for layer in action.layers:
                    for strip in layer.strips:
                        for cb in strip.channelbags:
                            fcurves_iters.append(cb.fcurves)
            for fcurves in fcurves_iters:
                for path, idx in channels_paths_indices:
                    fc = fcurves.find(path, index=idx)
                    if fc is None:
                        continue
                    for kp in fc.keyframe_points:
                        f = int(round(kp.co[0]))
                        if anim_frame_start <= f <= anim_frame_end:
                            frames.add(f)
            return sorted(frames)

        def _bone_key_frames(bone_name):
            rot_paths = [(f'pose.bones["{bone_name}"].rotation_quaternion', i)
                         for i in range(4)]
            sc_paths  = [(f'pose.bones["{bone_name}"].scale', i)
                         for i in range(3)]
            tr_paths  = [(f'pose.bones["{bone_name}"].location', i)
                         for i in range(3)]
            return (_fcurve_frames(rot_paths),
                    _fcurve_frames(sc_paths),
                    _fcurve_frames(tr_paths))

        # First pass: figure out for each bone which frames have which
        # kind of key. Build a master set of all frames we need to sample.
        per_bone_frames = {}
        all_sample_frames = set()
        for pb in arm_obj.pose.bones:
            rot_fr, sc_fr, tr_fr = _bone_key_frames(pb.name)
            # If a bone has NO keyframes anywhere (static), fall back to
            # writing two keys (start and end) so the file still records
            # the bone's pose. This keeps DirectX consumers that expect
            # at least one key from getting confused.
            if not rot_fr and not sc_fr and not tr_fr:
                fallback = [anim_frame_start, anim_frame_end]
                if fallback[0] == fallback[1]:
                    fallback = [anim_frame_start]
                rot_fr = sc_fr = tr_fr = list(fallback)
            per_bone_frames[pb.name] = (set(rot_fr), set(sc_fr), set(tr_fr))
            all_sample_frames.update(rot_fr)
            all_sample_frames.update(sc_fr)
            all_sample_frames.update(tr_fr)

        # Second pass: for each frame we need, set the scene to that
        # frame and bake whichever bones have a key there.
        # Pre-resolve the per-bone data ONCE: iterating arm_obj.pose.bones
        # re-creates the RNA collection every frame, and dict lookups +
        # three membership tests per bone per frame add up across dense
        # rigs (55 bones x 2500 frames). Bones with no keyed frames at
        # all are dropped here instead of being re-tested every frame.
        _bone_iter = []
        for pb in arm_obj.pose.bones:
            rs, ss, ts = per_bone_frames[pb.name]
            if rs or ss or ts:
                _bone_iter.append((pb, pb.name, rs, ss, ts, pb.parent))
        _need_mat = (anim_key_format == "MATRIX")
        for fr in sorted(all_sample_frames):
            scene.frame_set(fr)

            for pb, name, rot_set, sc_set, tr_set, pb_parent in _bone_iter:
                in_rot = fr in rot_set
                in_sc  = fr in sc_set
                in_tr  = fr in tr_set
                if not (in_rot or in_sc or in_tr):
                    continue

                world_bl = pb.matrix

                if pb_parent:
                    local_bl = pb_parent.matrix.inverted() @ world_bl

                    dx_local = local_bl
                    dx_t = dx_local.to_translation() if (in_tr or in_sc) else None
                    dx_s = dx_local.to_scale() if in_sc else None
                    q    = dx_local.to_quaternion() if in_rot else None
                else:
                    # ROOT: write absolute rotation in un-conv'd file space,
                    # not just the pose offset. The importer reconstructs
                    # via local_rest_q = (conv.inv @ matrix_local).to_quat()
                    # and pose_q = local_rest_q.inv @ abs_q, so we must
                    # write abs_q = rotation of (conv.inv @ pb.matrix).
                    # Translation/scale come from the world matrix. A
                    # non-identity armature object transform enters here
                    # too, keeping keys in the same space as the FTMs.
                    _arm_M = _arm_world_or_none(arm_obj)
                    _world_full = (_arm_M @ world_bl if _arm_M is not None
                                   else world_bl)
                    dx_world = _bl_bone_to_dx_world(_world_full, bl_to_dx_3, inv_scale)
                    dx_t = dx_world.to_translation()
                    dx_s = dx_world.to_scale()
                    _pose_full = (_arm_M @ pb.matrix if _arm_M is not None
                                  else pb.matrix)
                    un_conv_pose = bl_to_dx_3 @ _pose_full.to_3x3()
                    q = un_conv_pose.to_quaternion()

                if in_rot:
                    baked[name]["rot"]  [fr] = (q.w, -q.x, -q.y, -q.z)
                if in_sc:
                    baked[name]["scale"][fr] = (dx_s.x, dx_s.y, dx_s.z)
                if in_tr:
                    baked[name]["pos"]  [fr] = (dx_t.x, dx_t.y, dx_t.z)

                # Matrix-key export: store the full DX-space local 4x4 for
                # this frame. The importer reads matrix keys as dx_local and
                # rebuilds the pose via rest^-1 @ parent_rest @ dx_local, so we
                # store the same parent-relative local transform the TRS path
                # encodes. For the root we store the DX-space world transform
                # (already computed for the TRS path), which the importer's
                # matrix path consumes for a parentless bone.
                if _need_mat:
                    if pb_parent:
                        baked[name]["mat"][fr] = local_bl.copy()
                    else:
                        baked[name]["mat"][fr] = dx_world.copy()

        scene.frame_set(orig_frame)

        # AnimationSet name: prefer the active action's name (which the
        # importer sets from the source file's AnimationSet name), falling
        # back to a generic "anim" if no action is currently bound.
        _animset_name = "anim"
        try:
            _ad = arm_obj.animation_data
            if _ad is not None and _ad.action is not None:
                _nm = _ad.action.name
                # Strip any Blender-added suffix like ".001" so the exported
                # name is closer to the original source. Blender appends
                # .001/.002/etc. to avoid name collisions when the same
                # file is re-imported, but the engine that consumes the
                # .x file expects the original AnimationSet name.
                if _nm:
                    import re as _re
                    _nm = _re.sub(r'\.\d{3}$', '', _nm)
                    _animset_name = _nm
        except Exception:
            pass
        w(f"AnimationSet {_names.get(_animset_name)} {{\n")
        # some biped/game biped exports name the AnimationKey
        # nodes 'R', 'S', 'T' for rotation/scale/translation. Plain DirectX
        # leaves them unnamed.
        _ak_rot   = "AnimationKey R" if pz_compat else "AnimationKey"
        _ak_scale = "AnimationKey S" if pz_compat else "AnimationKey"
        _ak_trans = "AnimationKey T" if pz_compat else "AnimationKey"
        # Helper: scale a Blender frame number to a file tick value.
        # int(round(...)) keeps the file's tick field DWORD-compatible.
        def _tick(fr):
            return max(0, int(round(fr * tick_scale_out)))
        for pb in arm_obj.pose.bones:
            name = pb.name
            rot_keys   = baked[name]["rot"]
            scale_keys = baked[name]["scale"]
            pos_keys   = baked[name]["pos"]
            mat_keys   = baked[name]["mat"]

            w(f"\tAnimation {{\n\t\t{{ {_names.get(name)} }}\n")

            if anim_key_format == "MATRIX":
                # Single 4x4 matrix track (keyType 4). DirectX stores the
                # matrix row-major; mathutils Matrix is row-indexed m[row][col]
                # with translation in the last COLUMN, and the parser does
                # reshape(4,4).T on read. To round-trip exactly we must emit
                # so that reshape(flat,(4,4)).T == M, i.e. flat = M transposed
                # in reading order: iterate columns outer, rows inner over
                # m[r][c]. (Verified: translation lands back in the last
                # column, not the last row.)
                def _mat16(M):
                    return ",".join(
                        f"{M[r][c]:.6f}" for c in range(4) for r in range(4))
                entries = [f"\t\t\t{_tick(fr)};16;{_mat16(M)};;"
                           for fr, M in sorted(mat_keys.items())]
                w(f"\t\t{_ak_rot} {{\n\t\t\t4;\n\t\t\t{len(mat_keys)};\n")
                w(",\n".join(entries) + ";\n\t\t}\n")
                w("\t}\n")
                continue

            # A track with zero keys is skipped entirely: emitting the
            # header with an empty array leaves a dangling separator that
            # strict tokenizers reject, and a keyless track carries no
            # information anyway (the per-bone fallback above guarantees
            # at least one non-empty track per bone).
            def _emit_rot(rot_keys=rot_keys):
                if not rot_keys:
                    return
                w(f"\t\t{_ak_rot} {{\n\t\t\t0;\n\t\t\t{len(rot_keys)};\n")
                entries = [f"\t\t\t{_tick(fr)};4;{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f};;"
                           for fr, (qw, qx, qy, qz) in sorted(rot_keys.items())]
                w(",\n".join(entries) + ";\n\t\t}\n")

            def _emit_scale(scale_keys=scale_keys):
                if not scale_keys:
                    return
                w(f"\t\t{_ak_scale} {{\n\t\t\t1;\n\t\t\t{len(scale_keys)};\n")
                entries = [f"\t\t\t{_tick(fr)};3;{sx:.6f},{sy:.6f},{sz:.6f};;"
                           for fr, (sx, sy, sz) in sorted(scale_keys.items())]
                w(",\n".join(entries) + ";\n\t\t}\n")

            def _emit_trans(pos_keys=pos_keys):
                if not pos_keys:
                    return
                w(f"\t\t{_ak_trans} {{\n\t\t\t2;\n\t\t\t{len(pos_keys)};\n")
                entries = [f"\t\t\t{_tick(fr)};3;{px:.6f},{py:.6f},{pz:.6f};;"
                           for fr, (px, py, pz) in sorted(pos_keys.items())]
                w(",\n".join(entries) + ";\n\t\t}\n")

            # PZ high-precision originals order the tracks S, R, T per
            # bone; plain exports keep the historical R, S, T. Type-
            # tagged loaders accept either — matching the native order
            # just keeps diffs against original files clean.
            if pz_compat:
                _emit_scale(); _emit_rot(); _emit_trans()
            else:
                _emit_rot(); _emit_scale(); _emit_trans()

            w("\t}\n")

        # Object-level animation tracks: rigid meshes whose container
        # frames never became bones carry their placement and animation
        # as OBJECT F-curves (the importer's exact object-level chain
        # path — sb.X's x3ds pieces, HUD's crowbar). Without these
        # tracks the re-imported file loses both the animation and, for
        # FPSC-style files where keys replace FTMs, the placement.
        # Keys are written in the same conjugated file space as the
        # standalone frame FTMs, with the file-form conjugate
        # quaternion (w, -x, -y, -z) the importer's composer reads.
        def _obj_key_frames(o):
            frames = set()
            try:
                _ad = o.animation_data
                _act = _ad.action if _ad else None
            except Exception:
                _act = None
            if _act is None:
                return []
            _fc_iters = []
            if getattr(_act, "fcurves", None):
                _fc_iters.append(_act.fcurves)
            if hasattr(_act, "layers"):
                for _ly in _act.layers:
                    for _st in _ly.strips:
                        for _cb in _st.channelbags:
                            _fc_iters.append(_cb.fcurves)
            for _fcs in _fc_iters:
                for _fc in _fcs:
                    for _kp in _fc.keyframe_points:
                        _f = int(round(_kp.co[0]))
                        if anim_frame_start <= _f <= anim_frame_end:
                            frames.add(_f)
            return sorted(frames)

        _anim_obj_ids = set()
        for _aobj in mesh_objs:
            if id(_aobj) in _bone_nested_ids:
                continue
            _ofrs = _obj_key_frames(_aobj)
            if not _ofrs:
                continue
            _anim_obj_ids.add(id(_aobj))
            _o_rot, _o_scl, _o_pos = {}, {}, {}
            for _fr in _ofrs:
                scene.frame_set(_fr)
                _L = _bl_world_to_dx_frame(_aobj.matrix_world,
                                           bl_to_dx_3, inv_scale)
                _q = _L.to_quaternion()
                _o_rot[_fr] = (_q.w, -_q.x, -_q.y, -_q.z)
                _sv = _L.to_scale()
                _o_scl[_fr] = (_sv[0], _sv[1], _sv[2])
                _tv = _L.to_translation()
                _o_pos[_fr] = (_tv[0], _tv[1], _tv[2])
            scene.frame_set(orig_frame)
            _ofn = _aobj.get("_x_frame_name") or _aobj.name
            w(f"\tAnimation {{\n\t\t{{ {_names.get(_ofn)} }}\n")
            w(f"\t\t{_ak_rot} {{\n\t\t\t0;\n\t\t\t{len(_o_rot)};\n")
            entries = [f"\t\t\t{_tick(fr)};4;{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f};;"
                       for fr, (qw, qx, qy, qz) in sorted(_o_rot.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")
            w(f"\t\t{_ak_scale} {{\n\t\t\t1;\n\t\t\t{len(_o_scl)};\n")
            entries = [f"\t\t\t{_tick(fr)};3;{sx:.6f},{sy:.6f},{sz:.6f};;"
                       for fr, (sx, sy, sz) in sorted(_o_scl.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")
            w(f"\t\t{_ak_trans} {{\n\t\t\t2;\n\t\t\t{len(_o_pos)};\n")
            entries = [f"\t\t\t{_tick(fr)};3;{px:.6f},{py:.6f},{pz:.6f};;"
                       for fr, (px, py, pz) in sorted(_o_pos.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")
            w("\t}\n")

        # Auxiliary-frame animations (e.g. Translation_Data on high-precision
        # files): re-emit the source-file text verbatim so the
        # AnimationSet has a track for every Frame in the file, not
        # just the bones we baked from the armature.
        if _passthrough_aux_animations and not binary_format:
            _aa = str(_passthrough_aux_animations).strip()
            # Indent each line to sit one level inside the AnimationSet.
            _aa = "\n".join(f"\t{ln.lstrip()}" if ln.strip() else ln
                            for ln in _aa.split("\n"))
            w("\n" + _aa + "\n")

        w("}\n")

    if binary_format:
        payload = _ser.getvalue()
        if compress:
            with open(filepath, "wb") as fh:
                # When compressing the binary form, the format magic on
                fh.write(_mszip_wrap_x(payload, base_format="bin"))
        else:
            with open(filepath, "wb") as fh:
                fh.write(_bin_header)
                fh.write(payload)
    else:
        text_blob = "".join(out)
        if compress:
            # Strip the 'xof 0303txt 0032' line we wrote at the top — the
            text_bytes = text_blob.encode("utf-8")
            # Find and skip the leading magic line (always starts with
            # "xof " and is 16 bytes plus a newline).
            if text_bytes.startswith(b"xof "):
                # Drop the 16-byte magic and any blank lines following it
                inner = text_bytes[16:].lstrip(b"\r\n")
            else:
                inner = text_bytes
            with open(filepath, "wb") as fh:
                fh.write(_mszip_wrap_x(inner, base_format="txt"))
        else:
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(text_blob)

    return {"FINISHED"}, all_warnings


def _arm_world_or_none(arm_obj):
    """arm_obj.matrix_world when it meaningfully differs from identity,
    else None. Gating on None keeps every existing identity-armature
    export byte-identical while making non-identity armature transforms
    (e.g. the uniform-rig object scale the importer now applies, or a
    user-moved armature) flow consistently into root FTMs, root
    animation keys, skin offsets and parented-mesh vertices — all four
    must agree on the space or the skin detaches."""
    if arm_obj is None:
        return None
    try:
        M = arm_obj.matrix_world
        ident = Matrix.Identity(4)
        for r in range(4):
            for c in range(4):
                if abs(M[r][c] - ident[r][c]) > 1e-9:
                    return M.copy()
    except Exception:
        return None
    return None


def _write_material_block(w, mat, ind, name_or_none,
                          export_textures, filepath):
    """Emit one Material block. `name_or_none` carries the (sanitized)
    identifier for a named root-level declaration, or None for an
    anonymous inline block inside a MeshMaterialList (the style every
    native Virtual Sailor text generator uses). Shared by the root
    writer and the inline-materials compat path so both stay in sync."""
    bsdf = _get_principled(mat)

    face_color = mat.get("_x_face_color")
    power      = mat.get("_x_power")
    specular   = mat.get("_x_specular")
    emissive   = mat.get("_x_emissive")
    x_tex_name = mat.get("_x_texture_filename")

    if face_color is not None and len(face_color) >= 4:
        r, g, b, a = face_color[0], face_color[1], face_color[2], face_color[3]
    elif bsdf:
        col = bsdf.inputs["Base Color"].default_value
        r, g, b, a = col[0], col[1], col[2], col[3]
    else:
        r, g, b, a = 0.8, 0.8, 0.8, 1.0

    if power is not None:
        shininess = power
    elif bsdf:
        roughness = bsdf.inputs["Roughness"].default_value
        shininess = max(1.0, 128.0 ** (1.0 - roughness))
    else:
        shininess = 32.0

    if specular is not None and len(specular) >= 3:
        sr, sg, sb = specular[0], specular[1], specular[2]
    elif bsdf:
        spec_inp = (bsdf.inputs.get("Specular IOR Level")
                    or bsdf.inputs.get("Specular"))
        sv = spec_inp.default_value if spec_inp else 0.5
        sr = sg = sb = sv
    else:
        sr = sg = sb = 0.5

    if emissive is not None and len(emissive) >= 3:
        er, eg, eb = emissive[0], emissive[1], emissive[2]
    else:
        er = eg = eb = 0.0

    head = f"Material {name_or_none} {{" if name_or_none else "Material {"
    w(f"{ind}{head}\n")
    w(f"{ind}\t {r:.6f}; {g:.6f}; {b:.6f}; {a:.6f};;\n")
    w(f"{ind}\t {shininess:.6f};\n")
    w(f"{ind}\t {sr:.6f}; {sg:.6f}; {sb:.6f};;\n")
    w(f"{ind}\t {er:.6f}; {eg:.6f}; {eb:.6f};;\n")
    if export_textures:
        # Standard spelling is 'TextureFilename'. Some engines
        # (Bugsnax/Horsepower) author 'TextureFileName'; the importer
        # stashes the source spelling so round-trips keep it, while
        # fresh exports stay spec-standard — a declared-template/name
        # mismatch is exactly what strict .NET-style parsers fall
        # over on.
        _texfn_tag = str(mat.get("_x_texfn_case") or "TextureFilename")
        if _texfn_tag not in ("TextureFilename", "TextureFileName"):
            _texfn_tag = "TextureFilename"
        if x_tex_name:
            esc = x_tex_name.replace("\\", "\\\\")
            w(f'{ind}\t{_texfn_tag} {{"{esc}";}}\n')
        else:
            tp = _tex_path(mat)
            if tp:
                try:
                    rel = os.path.relpath(tp, os.path.dirname(filepath))
                except ValueError:
                    rel = os.path.basename(tp)
                rel = rel.replace("\\", "\\\\")
                w(f'{ind}\t{_texfn_tag} {{"{rel}";}}\n')
    w(f"{ind}}}\n")


def _mszip_wrap_x(payload: bytes, base_format: str = "bin") -> bytes:
    """Wrap a raw .x payload in the MSZIP-compressed container
    (xof 0303bzip / xof 0303tzip).

    Framing verified byte-level against native FPSC bzip files
    (Colonel-X.X and nine others):

        magic(16) | u32 totalUncompressedSize | chunk*

        chunk := u16 uncompressedSize (<= 32768)   -- on EVERY chunk
                 u16 compressedSize               -- INCLUDES the 2-byte 'CK'
                 'CK'
                 raw deflate

    totalUncompressedSize counts the 16-byte magic that is NOT stored in
    the stream (i.e. payload + 16). Native encoders chain the deflate
    dictionary across chunks; compressing each chunk independently is
    fully compatible (chained decoders simply see no back-references),
    so that is what we do.
    """
    out = bytearray()
    if base_format == "bin":
        out += b"xof 0303bzip0032"
    else:
        out += b"xof 0303tzip0032"

    out += (len(payload) + 16).to_bytes(4, "little")

    CHUNK_SIZE = 32768
    pos = 0
    while pos < len(payload) or (pos == 0 and not payload):
        chunk = payload[pos:pos + CHUNK_SIZE]
        co = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
        compressed = co.compress(chunk) + co.flush(zlib.Z_FINISH)
        out += len(chunk).to_bytes(2, "little")            # uncompressed size
        out += (len(compressed) + 2).to_bytes(2, "little") # incl. 'CK'
        out += b"CK"
        out += compressed
        pos += CHUNK_SIZE
        if not payload:
            break
    return bytes(out)
def _bl_world_to_dx_frame(bl_mat, bl_to_dx_3, inv_scale):
    """Conjugated Blender→DX world conversion (C @ M @ C^-1) for
    standalone mesh Frame FTMs. The importer rebuilds frame worlds by
    conjugation, and the mesh vertices are one-side-converted, so this
    is the form that composes back to the identical Blender world.
    Maps an Identity Blender world to an Identity FTM."""
    conv_4 = Matrix.Identity(4)
    conv_inv_4 = Matrix.Identity(4)
    inv3 = bl_to_dx_3.inverted()
    for r in range(3):
        for c in range(3):
            conv_4[r][c] = bl_to_dx_3[r][c]
            conv_inv_4[r][c] = inv3[r][c]
    dx_mat = conv_4 @ bl_mat @ conv_inv_4
    dx_mat[0][3] *= inv_scale
    dx_mat[1][3] *= inv_scale
    dx_mat[2][3] *= inv_scale
    # Conjugation turns exact zeros into ±1e-9 epsilons that format as
    # '-0.000000' under %.6f; snap anything below the format's precision
    # so an Identity world writes the same clean '0.000000' bytes the
    # source files carry (keeps Bugsnax round-trip exports byte-identical).
    for _r in range(4):
        for _c in range(4):
            if abs(dx_mat[_r][_c]) < 5e-7:
                dx_mat[_r][_c] = 0.0
    return dx_mat


def _bl_bone_to_dx_world(bl_mat, bl_to_dx_3, inv_scale):

    conv_4 = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            conv_4[r][c] = bl_to_dx_3[r][c]
    dx_mat = conv_4 @ bl_mat

    dx_mat[0][3] *= inv_scale
    dx_mat[1][3] *= inv_scale
    dx_mat[2][3] *= inv_scale
    return dx_mat

def _merge_split_group_for_export(group_objs, source_mesh_name):
    """Merge a group of Blender mesh objects (split from one source
    Mesh on import) into a single temporary object suitable for .x
    export as one Mesh node with a multi-material MeshMaterialList.

    The objects in `group_objs` are expected to be sorted by
    _x_split_group_idx so the resulting material slot order matches
    the original .x file's material declaration order.

    Returns a NEW Blender object linked to the active collection. The
    caller is responsible for unlinking + deleting it after use.
    """
    if not group_objs:
        raise ValueError("Empty group")

    # bmesh-based merge: concatenate verts/faces from all source meshes
    # into one bmesh, tracking which source-mesh-index each face came
    # from (for per-face material assignment) and which material to put
    # in each slot.
    merged_bm = bmesh.new()

    # Result data layers
    uv_layer = None
    deform_layer = None    # vertex weights

    # We'll build the material list in source-object order.
    merged_materials: list = []

    # Vertex-group name registry — the merged bmesh's deform layer
    # uses integer indices; we map bone names → indices here, in the
    # order they're first encountered. The output object's
    # vertex_groups must be created in the SAME order.
    vg_registry: list = []
    def _vg_idx_for(bone_name: str) -> int:
        try:
            return vg_registry.index(bone_name)
        except ValueError:
            idx = len(vg_registry)
            vg_registry.append(bone_name)
            return idx

    # Per-source-vert→merged-vert index map (re-keyed per source)
    for src_idx, src_obj in enumerate(group_objs):
        me_src = src_obj.data
        if not me_src:
            continue

        # Pull this source object's material into the merged list
        src_mats = _effective_materials(src_obj)
        # The split-import produces one material per object, but be
        # defensive: take the first non-None material.
        chosen_mat = next((m for m in src_mats if m is not None), None)
        merged_mat_idx = len(merged_materials)
        merged_materials.append(chosen_mat)

        # Build a temporary bmesh of THIS source mesh so we can iterate
        # its verts/faces/UVs/groups cleanly with bmesh semantics.
        src_bm = bmesh.new()
        src_bm.from_mesh(me_src)
        src_bm.faces.ensure_lookup_table()
        src_bm.verts.ensure_lookup_table()

        # Initialize the merged UV layer the first time we see one
        if src_bm.loops.layers.uv:
            src_uv = src_bm.loops.layers.uv.active or src_bm.loops.layers.uv[0]
            if uv_layer is None:
                uv_layer = merged_bm.loops.layers.uv.new(src_uv.name)
        else:
            src_uv = None

        # Initialize the merged deform (vertex-group weights) layer
        if src_bm.verts.layers.deform:
            src_deform = src_bm.verts.layers.deform.active or src_bm.verts.layers.deform[0]
            if deform_layer is None:
                deform_layer = merged_bm.verts.layers.deform.new()
        else:
            src_deform = None

        # Vertex group index → vertex group NAME mapping. The merged
        # object inherits a fresh vertex_groups list (built in
        # vg_registry order); we translate each source weight's
        # integer vg-index through the source object's name list to
        # find the matching merged index.
        src_vg_index_to_name = {vg.index: vg.name for vg in src_obj.vertex_groups}

        # Copy verts. Track the remap from src bmesh vert index → merged
        # bmesh BMVert.
        vert_remap: dict = {}
        for src_v in src_bm.verts:
            new_v = merged_bm.verts.new(src_v.co)
            new_v.normal = src_v.normal
            vert_remap[src_v.index] = new_v
            # Copy vertex group weights.
            if src_deform is not None and deform_layer is not None:
                src_weights = src_v[src_deform]
                for src_vg_idx, weight in src_weights.items():
                    bone_name = src_vg_index_to_name.get(src_vg_idx)
                    if bone_name is None:
                        continue
                    merged_vg_idx = _vg_idx_for(bone_name)
                    new_v[deform_layer][merged_vg_idx] = weight
        merged_bm.verts.ensure_lookup_table()
        merged_bm.verts.index_update()

        # Copy faces. Apply per-face material slot index pointing to
        # this source object's material in the merged slot list.
        for src_f in src_bm.faces:
            try:
                new_f = merged_bm.faces.new(
                    tuple(vert_remap[v.index] for v in src_f.verts)
                )
            except ValueError:
                # Duplicate face — bmesh refuses. Skip.
                continue
            new_f.smooth = src_f.smooth
            new_f.material_index = merged_mat_idx
            # Copy UVs (per-loop)
            if src_uv is not None and uv_layer is not None:
                for src_loop, new_loop in zip(src_f.loops, new_f.loops):
                    new_loop[uv_layer].uv = src_loop[src_uv].uv

        src_bm.free()

    merged_bm.verts.ensure_lookup_table()
    merged_bm.verts.index_update()
    merged_bm.faces.ensure_lookup_table()
    merged_bm.faces.index_update()

    # Build the merged mesh/object.
    merged_me = bpy.data.meshes.new(f"_xexp_merge_{source_mesh_name}")
    merged_bm.to_mesh(merged_me)
    merged_bm.free()
    merged_me.update()

    merged_obj = bpy.data.objects.new(f"_xexp_merge_{source_mesh_name}",
                                       merged_me)

    # Add materials to the merged object's data block in slot order
    for m in merged_materials:
        merged_me.materials.append(m)

    # Add vertex groups by name, in the SAME order as vg_registry so
    # the deform-layer integer indices match the object's vg list.
    for vg_name in vg_registry:
        merged_obj.vertex_groups.new(name=str(vg_name))

    # Copy provenance + parenting + armature modifier from the FIRST
    # source object so the exporter sees a fully-bound mesh.
    first = group_objs[0]
    # The frame/mesh names should be the ORIGINAL pre-split source.
    merged_obj["_x_mesh_name"] = source_mesh_name
    if first.get("_x_frame_name"):
        merged_obj["_x_frame_name"] = first["_x_frame_name"]
    if first.get("_x_container_bone"):
        # Split pieces of a container-frame mesh: the merged whole is
        # the container mesh, so the wrapperless bone-nested write
        # applies to it exactly as it would to the unsplit original.
        merged_obj["_x_container_bone"] = first["_x_container_bone"]
    if first.data.get("_x_vcolors_name"):
        merged_me["_x_vcolors_name"] = first.data["_x_vcolors_name"]

    # Per-vertex colors: the bmesh concatenation above does not carry
    # POINT-domain color attributes, silently dropping baked lighting /
    # kwXport vertex colours from any merged multi-material mesh. The
    # merged vertex order is the plain concatenation of the sources'
    # vertex orders, so rebuild the attribute sequentially.
    _src_vc = []
    _any_vc = False
    for _sobj in group_objs:
        _attr = None
        for _ca in getattr(_sobj.data, "color_attributes", ()) or ():
            if getattr(_ca, "domain", "") == 'POINT':
                _attr = _ca
                break
        _src_vc.append(_attr)
        _any_vc = _any_vc or (_attr is not None)
    if _any_vc:
        try:
            _vc_name = next((a.name for a in _src_vc if a is not None), "Col")
            _mattr = merged_me.color_attributes.new(
                name=_vc_name, type='FLOAT_COLOR', domain='POINT')
            _mi = 0
            for _sobj, _attr in zip(group_objs, _src_vc):
                for _vi in range(len(_sobj.data.vertices)):
                    if _attr is not None:
                        _mattr.data[_mi].color = _attr.data[_vi].color
                    else:
                        _mattr.data[_mi].color = (1.0, 1.0, 1.0, 1.0)
                    _mi += 1
        except Exception:
            pass

    # Match the first object's transform/parent/modifiers.
    merged_obj.matrix_world = first.matrix_world.copy()
    merged_obj.parent       = first.parent
    merged_obj.parent_type  = first.parent_type
    if first.parent_bone:
        merged_obj.parent_bone = first.parent_bone
    for mod_src in first.modifiers:
        if mod_src.type == 'ARMATURE':
            mod_dst = merged_obj.modifiers.new(name=mod_src.name, type='ARMATURE')
            mod_dst.object = mod_src.object
            mod_dst.use_vertex_groups = mod_src.use_vertex_groups

    # Link into the source's collection (or scene if no collection)
    placed = False
    for col in first.users_collection:
        try:
            col.objects.link(merged_obj)
            placed = True
            break
        except Exception:
            pass
    if not placed:
        try:
            bpy.context.scene.collection.objects.link(merged_obj)
        except Exception:
            pass

    return merged_obj



def _write_mesh_frame(obj, out, indent,
                      depsgraph, use_mesh_modifiers,
                      export_normals, export_uvs,
                      export_materials, export_weights,
                      arm_obj, bl_to_dx_3, inv_scale,
                      triangulate, written_mats,
                      unweld_on_export=True,
                      names=None,
                      frame_matrix_override=None,
                      suppress_frame_wrapper=False,
                      inline_materials=False,
                      fill_missing_uvs=False,
                      export_textures=True,
                      filepath=""):
    if names is None:
        names = _NameSanitizer()
    w        = out.feed if isinstance(out, _BinarySerializer) else out.append
    ind      = "\t" * indent
    warnings = []

    me_src = obj.data

    bm = bmesh.new()
    bm.from_mesh(me_src)
    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces)
    me_work = bpy.data.meshes.new("_x_export_tmp")
    bm.to_mesh(me_work)
    bm.free()

    me_work.update()

    conv_inv_3 = bl_to_dx_3

    bl_verts = me_work.vertices

    # Check whether we're going to re-emit a stashed DeclData block. If
    # so, the output vertex order must match the original file's order
    # exactly — DeclData is indexed by vertex position. Use a 1-to-1
    # vert mapping (no dedup) in that case so bl_verts[i] lines up with
    # the stashed DeclData's i-th vertex entry. This is the same
    # condition the writer below uses; computing it here lets us pick
    # the right vertex strategy upfront.
    _check_stashed_decldata = obj.get("_x_decldata")
    _check_stashed_orig_vc = obj.get("_x_orig_vert_count")
    _preserve_vert_order = bool(
        _check_stashed_decldata
        and _check_stashed_orig_vc is not None
        and int(_check_stashed_orig_vc) == len(bl_verts)
    )

    # Collect per-loop normal and UV so we can decide which loops can
    # share a vertex. Loops sharing the same Blender vertex index AND the
    # same normal AND the same UV collapse to a single output vertex;
    # any disagreement forces a split. This keeps clean meshes clean on
    # export while still producing valid DirectX geometry on meshes with
    # UV/normal seams.
    raw_loop_normals = _get_corner_normals(me_work) if export_normals else None
    uv_layer_active = me_work.uv_layers.active if (export_uvs and me_work.uv_layers) else None

    new_to_src = []
    new_loop_normal = []   # per output vertex
    new_uv          = []   # per output vertex
    faces = []
    loop_to_new_vi = [0] * len(me_work.loops)
    key_to_new_vi = {}

    def _round(v, q=1e-6):
        # Quantise floats so near-equal normals/UVs collapse to the same
        # key without falling foul of float rounding noise.
        return round(v / q) * q

    if _preserve_vert_order:
        # Identity mapping: every Blender vertex becomes one output
        # vertex at the same index. Each loop refers to its underlying
        # vertex directly. This guarantees the output's vertex ordering
        # matches the stashed DeclData's expected ordering.
        for vi in range(len(bl_verts)):
            new_to_src.append(vi)
            new_loop_normal.append(None)
            new_uv.append(None)
        for poly in me_work.polygons:
            face_indices = []
            for li in poly.loop_indices:
                vi = me_work.loops[li].vertex_index
                loop_to_new_vi[li] = vi
                face_indices.append(vi)
            faces.append(face_indices)
    elif unweld_on_export:
        # Unweld path: every loop becomes its own output vertex. No
        # dedup is done — the dedup key is effectively the loop index
        # itself. This restores the original file's vert count when
        # the mesh was imported with welding enabled (the .x format
        # naturally splits verts at UV/normal/smoothing-group seams,
        # so what gets written here is one vert per face-corner, just
        # like the original).
        for poly in me_work.polygons:
            face_indices = []
            for li in poly.loop_indices:
                src_vi = me_work.loops[li].vertex_index
                new_vi = len(new_to_src)
                new_to_src.append(src_vi)
                new_loop_normal.append(raw_loop_normals[li] if raw_loop_normals is not None else None)
                new_uv.append(tuple(uv_layer_active.data[li].uv) if uv_layer_active is not None else None)
                loop_to_new_vi[li] = new_vi
                face_indices.append(new_vi)
            faces.append(face_indices)
    else:
        for poly in me_work.polygons:
            face_indices = []
            for li in poly.loop_indices:
                src_vi = me_work.loops[li].vertex_index

                if raw_loop_normals is not None:
                    n = raw_loop_normals[li]
                    norm_key = (_round(n[0]), _round(n[1]), _round(n[2]))
                else:
                    norm_key = None

                if uv_layer_active is not None:
                    uv = uv_layer_active.data[li].uv
                    uv_key = (_round(uv[0]), _round(uv[1]))
                else:
                    uv_key = None

                key = (src_vi, norm_key, uv_key)
                new_vi = key_to_new_vi.get(key)
                if new_vi is None:
                    new_vi = len(new_to_src)
                    key_to_new_vi[key] = new_vi
                    new_to_src.append(src_vi)
                    new_loop_normal.append(raw_loop_normals[li] if raw_loop_normals is not None else None)
                    new_uv.append(tuple(uv_layer_active.data[li].uv) if uv_layer_active is not None else None)
                loop_to_new_vi[li] = new_vi
                face_indices.append(new_vi)
            faces.append(face_indices)

    n_verts = len(new_to_src)
    n_faces = len(faces)

    # A mesh parented to the armature is written with an Identity frame,
    # so its vertices must already BE in the composed space the offsets
    # and skeleton describe. With the armature at identity (every legacy
    # case) obj.matrix_world is identity too and this is a no-op; with a
    # transformed armature (uniform-rig object scale, user-moved rig)
    # the world transform is baked into the vertices here. Bone-nested
    # meshes (frame_matrix_override) keep object-local verts — their
    # frame carries the placement.
    _vert_world_M = None
    if (frame_matrix_override is None and arm_obj is not None
            and obj.parent == arm_obj
            and getattr(obj, "parent_type", "OBJECT") != "BONE"):
        _vert_world_M = _arm_world_or_none(obj)  # non-identity world only
        if _vert_world_M is not None:
            _vert_world_M = obj.matrix_world.copy()
    if _vert_world_M is not None:
        verts_dx = [(conv_inv_3 @ (_vert_world_M @ Vector(bl_verts[src_vi].co)))
                    * inv_scale
                    for src_vi in new_to_src]
    else:
        verts_dx = [(conv_inv_3 @ Vector(bl_verts[src_vi].co)) * inv_scale
                    for src_vi in new_to_src]

    # Wrapperless container write (suppress_frame_wrapper): the mesh's
    # container frame on import became the bone whose Frame the caller
    # is currently writing, and the importer never transformed the mesh
    # data itself (placement lives on the object). So the vertices are
    # written verbatim in object-local space — they ARE the original
    # frame-local data — and no holder Frame or transform is emitted.
    # (Verified: applying the Blender parent-relative local here instead
    # re-introduced the exact drift this path exists to fix.)

    # Build a loop-aligned normal list for the MeshNormals writer below.
    # The downstream code expects one entry per loop in poly-iteration
    # order; since we no longer have a 1-to-1 loop→new_vert mapping, we
    # rebuild this view explicitly from raw_loop_normals.
    if raw_loop_normals is not None:
        loop_normals = list(raw_loop_normals)
    else:
        loop_normals = []

    frame_name = obj.get("_x_frame_name") or obj.name
    _mn_stash = obj.get("_x_mesh_name")
    if _mn_stash is not None:
        # Stash present: reproduce the source name exactly. An empty
        # string means the source Mesh node was UNNAMED — write it
        # unnamed rather than inventing a 'Geo' suffix that compounds
        # on every round trip.
        mesh_name = str(_mn_stash)
    else:
        mesh_name = (f"{obj.name}Geo" if not obj.name.endswith("Geo")
                     else obj.name)
    stashed_frame_ftm = obj.get("_x_frame_ftm")
    # The stashed source FTM is only valid for a standalone frame write:
    # a bone-nested emission needs the caller's bone-RELATIVE local, and
    # world-baked vertices (transformed armature) have moved out of the
    # space the stash described.
    if frame_matrix_override is not None or _vert_world_M is not None:
        stashed_frame_ftm = None

    if not suppress_frame_wrapper:
        w(f"{ind}Frame {names.get(frame_name)} {{\n")
        w(f"{ind}\tFrameTransformMatrix {{\n")
        if stashed_frame_ftm is not None and len(stashed_frame_ftm) >= 16:
            # Snap ±1e-9 epsilons left by the flatten pass's matrix math:
            # they format as '-0.000000' and needlessly dirty the diff
            # against the source file's clean zeros.
            ftm_str = ",".join(
                f"{(0.0 if abs(float(v)) < 5e-7 else float(v)):.6f}"
                for v in stashed_frame_ftm[:16])
        else:
            if frame_matrix_override is not None:
                # Nested emission (bone-parented mesh): the caller supplies
                # the bone-relative local transform.
                frame_mat = frame_matrix_override
            elif arm_obj and obj.parent == arm_obj:
                # Skinned meshes sit at Identity world (their verts are
                # in composed model space), and when the ARMATURE itself
                # is transformed the vertices were baked through
                # matrix_world above — both cases need an Identity
                # frame. But a RIGID mesh parented to the armature at
                # object level has a non-identity world that an Identity
                # frame silently discards, scattering the pieces on
                # re-import. Write its world as the frame FTM.
                if _vert_world_M is not None:
                    frame_mat = Matrix.Identity(4)
                else:
                    frame_mat = _bl_world_to_dx_frame(obj.matrix_world,
                                                      bl_to_dx_3, inv_scale)
            else:
                # Standalone mesh frame. The FTM must be the CONJUGATED
                # world (C @ W @ C^-1): the mesh vertices below are
                # written one-side-converted (C @ v), and the importer
                # rebuilds frame worlds by conjugation, so a one-sided
                # FTM here composes to C@W@C@v in file space — verts
                # rotated by C^2 whenever W carries rotation or
                # translation (sb.X's x3ds pieces sat 2142 units off).
                # For the common Identity world this now writes an
                # Identity FTM, matching the source files byte-for-byte
                # instead of writing the bare conversion matrix.
                frame_mat = _bl_world_to_dx_frame(obj.matrix_world,
                                                  bl_to_dx_3, inv_scale)
            ftm_str = _mat4_to_dx(frame_mat)
        w(f"{ind}\t\t{ftm_str};;\n")
        w(f"{ind}\t}}\n")

    if n_verts > 65535:
        warnings.append(
            f"Mesh '{obj.name}' exports with {n_verts} vertices — above "
            f"the 65535 limit of 16-bit-index legacy tools (fragMOTION "
            f"and some older engines may truncate or reject it). The "
            f".x format itself is fine with it.")

    if mesh_name:
        w(f"{ind}\tMesh {names.get(mesh_name)} {{\n")
    else:
        w(f"{ind}\tMesh {{\n")

    w(f"{ind}\t\t{n_verts};\n")
    vert_lines = [f"{ind}\t\t {v.x:.6f}; {v.y:.6f}; {v.z:.6f};" for v in verts_dx]
    w(",\n".join(vert_lines) + ";\n")

    w(f"\n{ind}\t\t{n_faces};\n")
    face_lines = [f"{ind}\t\t{len(f)};{','.join(str(i) for i in f)};" for f in faces]
    w(",\n".join(face_lines) + ";\n")

    # Round-trip preservation for high-precision-style meshes: if this object was
    # imported from a file that used a DeclData block instead of the
    # standard MeshNormals + MeshTextureCoords blocks, re-emit DeclData
    # verbatim and skip the standard blocks. Falls back to standard
    # blocks if the user has edited the mesh and the vertex count no
    # longer matches the stashed data.
    _stashed_decldata = obj.get("_x_decldata")
    _stashed_xskinheader = obj.get("_x_xskinheader")
    _stashed_orig_vc = obj.get("_x_orig_vert_count")
    _use_stashed_decl = bool(
        _stashed_decldata
        and _stashed_orig_vc is not None
        and int(_stashed_orig_vc) == len(bl_verts)
    )

    # Row-remapped DeclData: when welding changed the vertex count the
    # verbatim stash above no longer lines up, and falling back to
    # standard MeshNormals/MeshTextureCoords silently drops tangents and
    # any second TEXCOORD set the DeclData carried. The import-time
    # weld map (_x_declmap: original row -> post-weld vert) lets us pick,
    # for every OUTPUT vertex, the original row whose UV/normal match
    # that vertex — re-emitting the source data bit-exactly in the new
    # vertex order. Resolution happens up front; any unmatched vertex
    # aborts the mode and the standard blocks below emit as before.
    _decl_remap_rows = None
    _decl_stash_parsed = None
    if (_stashed_decldata and not _use_stashed_decl
            and obj.get("_x_declmap") is not None):
        try:
            import re as _re
            import struct as _st
            _dnums = [int(x) for x in
                      _re.findall(r'-?\d+', str(_stashed_decldata))]
            _ne = _dnums[0]
            _els = []
            _di = 1
            for _ in range(_ne):
                _els.append(tuple(_dnums[_di:_di + 4]))
                _di += 4
            _nd = _dnums[_di]; _di += 1
            _dwords = _dnums[_di:_di + _nd]
            _tf = {1: 2, 2: 3, 3: 4}
            _stride = sum(_tf.get(t, 0) for t, _m, _u, _ui in _els)
            if (len(_dwords) == _nd and _stride > 0
                    and _nd % _stride == 0
                    and all(t in _tf for t, _m, _u, _ui in _els)):
                _rows = _nd // _stride
                _declmap = [int(v) for v in obj["_x_declmap"]]
                if _rows == len(_declmap):
                    _offs = {}
                    _off = 0
                    for t, _m, _u, _ui in _els:
                        _offs.setdefault((_u, _ui), _off)
                        _off += _tf[t]
                    _floats = [
                        _st.unpack('<f', _st.pack('<I', d & 0xFFFFFFFF))[0]
                        for d in _dwords]
                    _uv0o = _offs.get((5, 0))
                    _nrmo = _offs.get((3, 0))
                    _groups = {}
                    for _ri, _post in enumerate(_declmap):
                        _groups.setdefault(_post, []).append(_ri)
                    _resolved = []
                    _ok = True
                    for _oi, _svi in enumerate(new_to_src):
                        _cands = _groups.get(_svi)
                        if not _cands:
                            _ok = False
                            break
                        if len(_cands) == 1 or (_uv0o is None
                                                and _nrmo is None):
                            _resolved.append(_cands[0])
                            continue
                        _tuv = new_uv[_oi] if _oi < len(new_uv) else None
                        _tn = (new_loop_normal[_oi]
                               if _oi < len(new_loop_normal) else None)
                        if _tn is not None:
                            _tnd = conv_inv_3 @ Vector(_tn)
                        _best = None
                        _bscore = None
                        for _ri in _cands:
                            _b = _ri * _stride
                            _uverr = 0.0
                            if _uv0o is not None and _tuv is not None:
                                _uverr = (abs(_floats[_b + _uv0o]
                                              - _tuv[0])
                                          + abs(_floats[_b + _uv0o + 1]
                                                - (1.0 - _tuv[1])))
                            _ndot = 0.0
                            if _nrmo is not None and _tn is not None:
                                _ndot = (_floats[_b + _nrmo] * _tnd.x
                                         + _floats[_b + _nrmo + 1] * _tnd.y
                                         + _floats[_b + _nrmo + 2] * _tnd.z)
                            _score = (_uverr, -_ndot)
                            if _bscore is None or _score < _bscore:
                                _bscore = _score
                                _best = _ri
                        if (_uv0o is not None and _tuv is not None
                                and _bscore[0] > 2e-3):
                            _ok = False
                            break
                        _resolved.append(_best)
                    if _ok and len(_resolved) == n_verts:
                        _decl_remap_rows = _resolved
                        _decl_stash_parsed = (_els, _stride, _dwords)
        except Exception:
            _decl_remap_rows = None
    _use_decl_remap = _decl_remap_rows is not None

    if (export_normals and loop_normals and not _use_stashed_decl
            and not _use_decl_remap):

        if _vert_world_M is not None:
            # Vertices were baked through the object's world transform
            # above, so normals must rotate through the same basis (its
            # normalized 3x3; uniform scale drops out, rotation is kept)
            # or lighting detaches from the geometry.
            _nrm_R = _vert_world_M.to_3x3()
            _ns = _nrm_R.to_scale()
            _nrm_R = Matrix([[_nrm_R[r][c] / (_ns[c] or 1.0)
                              for c in range(3)] for r in range(3)])
            dx_loop_normals = [conv_inv_3 @ (_nrm_R @ Vector(n))
                               for n in loop_normals]
        else:
            dx_loop_normals = [conv_inv_3 @ Vector(n) for n in loop_normals]

        # Blender zeroes corner normals on zero-area faces — including
        # explicitly-set custom split normals — so meshes with degenerate
        # triangles (57 PZ weapon/prop files carry them) read back
        # (0,0,0) for those loops. A literal 0;0;0 in MeshNormals breaks
        # loaders that normalize on load, so borrow a nonzero normal
        # from another loop of the same source vertex; the degenerate
        # face always shares its verts with real geometry, and the
        # neighbour's value is what the original exporters wrote there.
        if any(n.length_squared < 1e-12 for n in dx_loop_normals):
            _vert_nrm = {}
            _li = 0
            for _poly in me_work.polygons:
                for _pl in _poly.loop_indices:
                    _n = dx_loop_normals[_li]
                    if _n.length_squared >= 1e-12:
                        _vert_nrm.setdefault(
                            me_work.loops[_pl].vertex_index, _n)
                    _li += 1
            _li = 0
            for _poly in me_work.polygons:
                for _pl in _poly.loop_indices:
                    if dx_loop_normals[_li].length_squared < 1e-12:
                        _sub = _vert_nrm.get(me_work.loops[_pl].vertex_index)
                        dx_loop_normals[_li] = (Vector(_sub) if _sub is not None
                                                else Vector((0.0, 1.0, 0.0)))
                    _li += 1

        unique_norms   = []
        norm_key_to_idx = {}

        face_norm_idx  = []

        loop_index = 0
        for face in faces:
            corners = []
            for _ in face:
                n = dx_loop_normals[loop_index]
                key = (round(n.x, 6), round(n.y, 6), round(n.z, 6))
                idx = norm_key_to_idx.get(key)
                if idx is None:
                    idx = len(unique_norms)
                    norm_key_to_idx[key] = idx
                    unique_norms.append(key)
                corners.append(idx)
                loop_index += 1
            face_norm_idx.append(corners)

        w(f"\n{ind}\t\tMeshNormals {{\n{ind}\t\t\t{len(unique_norms)};\n")
        w(",\n".join(f"{ind}\t\t\t {nx:.6f};{ny:.6f};{nz:.6f};"
                     for nx, ny, nz in unique_norms) + ";\n")
        w(f"\n{ind}\t\t\t{n_faces};\n")
        face_norm_lines = [f"{ind}\t\t\t{len(fni)};{','.join(str(i) for i in fni)};"
                           for fni in face_norm_idx]
        w(",\n".join(face_norm_lines) + ";\n")
        w(f"{ind}\t\t}}\n")

    if (export_uvs and fill_missing_uvs and not me_work.uv_layers
            and not _use_stashed_decl and not _use_decl_remap):
        # Compat fill: every native Virtual Sailor mesh carries a
        # MeshTextureCoords block, and some strict consumers treat its
        # absence as malformed. Zeroed coordinates keep them loading
        # untextured rather than failing.
        w(f"\n{ind}\t\tMeshTextureCoords {{\n{ind}\t\t\t{n_verts};\n")
        w(",\n".join(f"{ind}\t\t\t 0.000000;0.000000;"
                     for _ in range(n_verts)) + ";\n")
        w(f"{ind}\t\t}}\n")

    if (export_uvs and me_work.uv_layers and not _use_stashed_decl
            and not _use_decl_remap):
        uv_layer = me_work.uv_layers.active

        new_uvs = [(0.0, 0.0)] * n_verts
        for poly in me_work.polygons:
            for li in poly.loop_indices:
                nv = loop_to_new_vi[li]
                uv = uv_layer.data[li].uv
                new_uvs[nv] = (uv[0], 1.0 - uv[1])

        w(f"\n{ind}\t\tMeshTextureCoords {{\n{ind}\t\t\t{n_verts};\n")
        uv_lines = [f"{ind}\t\t\t {u:.6f};{v_:.6f};" for u, v_ in new_uvs]
        w(",\n".join(uv_lines) + ";\n")
        w(f"{ind}\t\t}}\n")

    # Per-vertex colors (kwXport-style; FPSC segment meshes use them for
    # baked lighting). Emitted from the active POINT-domain color
    # attribute, one indexed entry per OUTPUT vertex via the same
    # new_to_src mapping the position writer uses, so colors stay glued
    # to their vertices across the unweld/preserve-order paths. Format
    # matches native files: nVertexColors; then index;r;g;b;a;; entries.
    _vcol_attr = None
    try:
        for _ca in getattr(me_work, "color_attributes", ()) or ():
            if getattr(_ca, "domain", "") == 'POINT':
                _vcol_attr = _ca
                break
    except Exception:
        _vcol_attr = None
    if _vcol_attr is not None:
        try:
            _vc_name = str(me_work.get("_x_vcolors_name")
                           or obj.data.get("_x_vcolors_name") or "")
            _vc_head = (f"MeshVertexColors {_vc_name} {{" if _vc_name
                        else "MeshVertexColors {")
            w(f"\n{ind}\t\t{_vc_head}\n{ind}\t\t\t{n_verts};\n")
            _vc_lines = []
            for _ovi, _svi in enumerate(new_to_src):
                try:
                    _c = _vcol_attr.data[_svi].color
                    _r, _g, _b = float(_c[0]), float(_c[1]), float(_c[2])
                    _a = float(_c[3]) if len(_c) > 3 else 1.0
                except Exception:
                    _r = _g = _b = _a = 1.0
                _vc_lines.append(
                    f"{ind}\t\t\t{_ovi};"
                    f"{_r:.6f};{_g:.6f};{_b:.6f};{_a:.6f};;")
            w(",\n".join(_vc_lines) + ";\n")
            w(f"{ind}\t\t}}\n")
        except Exception:
            pass

    if export_materials:
        mats = _effective_materials(obj)
        if mats:
            w(f"\n{ind}\t\tMeshMaterialList {{\n{ind}\t\t\t{len(mats)};\n{ind}\t\t\t{n_faces};\n")
            w(",\n".join(f"{ind}\t\t\t{p.material_index}" for p in me_work.polygons) + ";\n")
            for mat in mats:
                if inline_materials:
                    # Inline blocks inside the list — the style native
                    # Virtual Sailor generators and kwXport both use, and
                    # the safest shape for template-driven parsers that
                    # choke on cross-object references. kwXport keeps the
                    # material NAMED inline ("Material bottom { ... }"),
                    # which stays readable and round-trips, so do the same.
                    _write_material_block(w, mat, f"{ind}\t\t\t",
                                          names.get(mat.name),
                                          export_textures, filepath)
                else:
                    w(f"{ind}\t\t\t{{{names.get(mat.name)}}}\n")
            w(f"{ind}\t\t}}\n")

    # Re-emit the original DeclData block verbatim when this mesh was
    # imported from a file that used one and the vertex count is
    # unchanged. Sits between MeshMaterialList and XSkinMeshHeader to
    # match the original high-precision biped layout.
    if _use_stashed_decl:
        _dd = str(_stashed_decldata).strip()
        _dd = "\n".join(f"{ind}\t\t{ln.lstrip()}" if ln.strip() else ln
                        for ln in _dd.split("\n"))
        w("\n" + _dd + "\n")
    elif _use_decl_remap:
        _els, _stride, _dwords = _decl_stash_parsed
        w(f"\n{ind}\t\tDeclData {{\n{ind}\t\t\t{len(_els)};\n")
        _el_lines = [f"{ind}\t\t\t{t};{m};{u};{ui};"
                     for t, m, u, ui in _els]
        w(",\n".join(_el_lines) + ";\n")
        w(f"{ind}\t\t\t{n_verts * _stride};\n")
        _out_dw = []
        for _ri in _decl_remap_rows:
            _b = _ri * _stride
            _out_dw.extend(_dwords[_b:_b + _stride])
        w(",\n".join(f"{ind}\t\t\t{d & 0xFFFFFFFF}" for d in _out_dw)
          + ";\n")
        w(f"{ind}\t\t}}\n")

    if export_weights and arm_obj and obj.vertex_groups:
        bone_names = {b.name for b in arm_obj.data.bones}

        orig_me = obj.data
        vgroups = [vg for vg in obj.vertex_groups if vg.name in bone_names]

        if vgroups:

            ref_me = orig_me if not use_mesh_modifiers or len(orig_me.vertices) == n_verts else me_work

            src_to_new = {}
            for new_vi, src_vi in enumerate(new_to_src):
                src_to_new.setdefault(src_vi, []).append(new_vi)

            group_influences = []
            for vg in vgroups:
                influences = []
                gi = vg.index
                for v in ref_me.vertices:
                    for g in v.groups:
                        if g.group == gi and g.weight > 0.0:

                            for nv in src_to_new.get(v.index, []):
                                influences.append((nv, g.weight))
                            break
                # Include the group even if it has no influences: high-precision
                # / biped exports declare a SkinWeights entry
                # for every bone in the skin set, including ones the
                # artist hasn't painted any weights onto. Skipping them
                # changes the XSkinMeshHeader bone count seen by the
                # engine and can mis-index downstream lookups.
                group_influences.append((vg, influences))

            if group_influences:
                n_groups = len(group_influences)

                if _stashed_xskinheader:
                    # Re-emit the original XSkinMeshHeader text verbatim
                    # whenever we have it stashed — this is independent
                    # of whether DeclData round-trips, since the header
                    # is just metadata (max-weights-per-vertex/face and
                    # bone count) not vertex-indexed data.
                    _xs = str(_stashed_xskinheader).strip()
                    _xs = "\n".join(f"{ind}\t\t{ln.lstrip()}" if ln.strip() else ln
                                    for ln in _xs.split("\n"))
                    w("\n" + _xs + "\n")
                else:
                    # Compute proper max-weights-per-vertex and
                    # max-weights-per-face for the XSkinMeshHeader.
                    # Per-vertex: count how many vertex groups each
                    # vertex belongs to (capped by what we actually
                    # exported, which is the same set as group_influences).
                    influence_set_by_vert = {}
                    for vg_idx, (_vg, infs) in enumerate(group_influences):
                        for nv, _w in infs:
                            influence_set_by_vert.setdefault(nv, set()).add(vg_idx)
                    max_per_vert = max((len(s) for s in influence_set_by_vert.values()),
                                       default=0)
                    # Per-face: union of vertex influences across the
                    # face's corners. Bounded above by 3 * max_per_vert.
                    max_per_face = 0
                    for face_corners in faces:
                        face_infs = set()
                        for nv in face_corners:
                            face_infs |= influence_set_by_vert.get(nv, set())
                        if len(face_infs) > max_per_face:
                            max_per_face = len(face_infs)
                    w(f"\n{ind}\t\tXSkinMeshHeader {{\n")
                    w(f"{ind}\t\t\t{max_per_vert};\n{ind}\t\t\t{max_per_face};\n{ind}\t\t\t{n_groups};\n")
                    w(f"{ind}\t\t}}\n")

                for vg, influences in group_influences:
                    bone = arm_obj.data.bones.get(vg.name)

                    if bone:
                        # Bind world includes the armature object transform
                        # (when non-identity) — must match the space the
                        # frame hierarchy and vertices are written in.
                        _arm_M = _arm_world_or_none(arm_obj)
                        _bind_rest = (_arm_M @ bone.matrix_local
                                      if _arm_M is not None
                                      else bone.matrix_local)
                        bind_dx = _bl_bone_to_dx_world(_bind_rest, bl_to_dx_3, inv_scale)
                        try:
                            offset = bind_dx.inverted()
                        except ValueError:
                            offset = Matrix.Identity(4)
                    else:
                        offset = Matrix.Identity(4)

                    w(f"\n{ind}\t\tSkinWeights {{\n")
                    w(f"{ind}\t\t\t\"{names.get(vg.name)}\";\n")
                    w(f"{ind}\t\t\t{len(influences)};\n")
                    w(",\n".join(f"{ind}\t\t\t{vi}" for vi, _ in influences) + ";\n")
                    w(",\n".join(f"{ind}\t\t\t{wt:.6f}" for _, wt in influences) + ";\n")
                    w(f"{ind}\t\t\t{_mat4_to_dx(offset)};;\n")
                    w(f"{ind}\t\t}}\n")

    w(f"{ind}\t}}\n")
    if not suppress_frame_wrapper:
        w(f"{ind}}}\n")

    bpy.data.meshes.remove(me_work)
    return warnings
