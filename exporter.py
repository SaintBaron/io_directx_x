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
from typing import List, Optional, Tuple
from mathutils import Matrix, Vector, Quaternion
from .parser import XNode, _is_mesh_bone_name

_BT_NAME      = 0x0001
_BT_STRING    = 0x0002
_BT_INT_LIST  = 0x0006
_BT_FLT_LIST  = 0x0007
_BT_OBRACE    = 0x000a
_BT_CBRACE    = 0x000b
_BT_COMMA     = 0x0013
_BT_SEMICOLON = 0x0014

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

    def feed(self, text: str):
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

                is_plain_int = ('.' not in num and 'e' not in num.lower()
                                and not num.startswith('-'))
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
        self._buf += b'\x00\x00'

    def _flush_ints(self):
        if not self._pending_ints:
            return
        self._emit_u16(_BT_INT_LIST)
        self._emit_u32(len(self._pending_ints))
        for v in self._pending_ints:
            self._emit_u32(v)
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
             anim_fps=30.0,
             anim_frame_start=1,
             anim_frame_end=250,
             triangulate=False,
             binary_format=False,
             compress=False,
             **_):

    scene     = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    objects   = context.selected_objects if use_selection else list(context.scene.objects)

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

    if binary_format:

        _bin_header = b"xof 0303bin 0032"

        w(f"AnimTicksPerSecond {{ {int(anim_fps)}; }}\n")
    else:
        w("xof 0303txt 0032\n\n")
        w(f"AnimTicksPerSecond {{\n\t{int(anim_fps)};\n}}\n")

    mesh_objs     = [o for o in objects if o.type == "MESH"]
    armature_objs = [o for o in objects if o.type == "ARMATURE"]
    arm_obj       = armature_objs[0] if armature_objs else None

    written_mats = {}
    if export_materials:
        for obj in mesh_objs:
            for slot in obj.material_slots:
                mat = slot.material
                if not mat or mat.name in written_mats:
                    continue
                written_mats[mat.name] = mat
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

                w(f"Material {mat.name} {{\n")
                w(f"\t {r:.6f}; {g:.6f}; {b:.6f}; {a:.6f};;\n")
                w(f"\t {shininess:.6f};\n")
                w(f"\t {sr:.6f}; {sg:.6f}; {sb:.6f};;\n")
                w(f"\t {er:.6f}; {eg:.6f}; {eb:.6f};;\n")
                if export_textures:
                    if x_tex_name:

                        esc = x_tex_name.replace("\\", "\\\\")
                        w(f'\tTextureFileName {{"{esc}";}}\n')
                    else:
                        tp = _tex_path(mat)
                        if tp:
                            try:
                                rel = os.path.relpath(tp, os.path.dirname(filepath))
                            except ValueError:
                                rel = os.path.basename(tp)
                            rel = rel.replace("\\", "\\\\")
                            w(f'\tTextureFileName {{"{rel}";}}\n')
                w("}\n")

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

                    local_mat = _bl_bone_to_dx_world(bone.matrix_local, bl_to_dx_3, inv_scale)
                ftm_string = _mat4_to_dx(local_mat)

            w(f"{ind}Frame {bone.name} {{\n")
            w(f"{ind}\tFrameTransformMatrix {{\n")
            w(f"{ind}\t\t{ftm_string};;\n")
            w(f"{ind}\t}}\n")
            for child in bone.children:
                write_bone(child, indent + 1)
            w(f"{ind}}}\n")

        for rb in (b for b in arm_data.bones if not b.parent):
            write_bone(rb, 0)

    armature_set = set(armature_objs)
    all_warnings = []
    for obj in mesh_objs:
        all_warnings.extend(_write_mesh_frame(obj, out, 0,
                          depsgraph, use_mesh_modifiers,
                          export_normals, export_uvs,
                          export_materials, export_weights,
                          arm_obj, bl_to_dx_3, inv_scale,
                          triangulate, written_mats))

    if export_animation and arm_obj:
        orig_frame = scene.frame_current
        frame_count = anim_frame_end - anim_frame_start + 1

        rest_arm_bl = {b.name: b.matrix_local.copy() for b in arm_obj.data.bones}

        baked = {b.name: {"rot": {}, "scale": {}, "pos": {}} for b in arm_obj.pose.bones}

        conv_3 = Matrix.Identity(3)
        conv_inv_3 = bl_to_dx_3
        conv_3 = conv_inv_3.transposed()

        for fr in range(anim_frame_start, anim_frame_end + 1):
            scene.frame_set(fr)

            for pb in arm_obj.pose.bones:
                name = pb.name

                world_bl = pb.matrix.copy()

                if pb.parent:
                    parent_world_bl = pb.parent.matrix.copy()
                    local_bl = parent_world_bl.inverted() @ world_bl

                    dx_local = local_bl
                    dx_rot = dx_local.to_3x3()
                    dx_t   = dx_local.to_translation()
                    dx_s   = dx_local.to_scale()
                    q      = dx_rot.to_quaternion()
                else:
                    # ROOT bone: rotation is the pose-bone-local pose offset
                    # (matrix_basis), NOT the bone's world DX matrix.  The
                    # importer treats the file's stored quaternion as the
                    # pose offset for the skeleton root (local_rest_q is
                    # identity), so the round-trip requires q = pose_q.
                    # Position and scale still come from the world matrix.
                    dx_world = _bl_bone_to_dx_world(world_bl, bl_to_dx_3, inv_scale)
                    dx_t = dx_world.to_translation()
                    dx_s = dx_world.to_scale()
                    q    = pb.matrix_basis.to_3x3().to_quaternion()

                qw, qx, qy, qz = q.w, -q.x, -q.y, -q.z

                baked[name]["rot"]  [fr] = (qw, qx, qy, qz)
                baked[name]["scale"][fr] = (dx_s.x, dx_s.y, dx_s.z)
                baked[name]["pos"]  [fr] = (dx_t.x, dx_t.y, dx_t.z)

        scene.frame_set(orig_frame)

        w("AnimationSet anim {\n")
        for pb in arm_obj.pose.bones:
            name = pb.name
            rot_keys   = baked[name]["rot"]
            scale_keys = baked[name]["scale"]
            pos_keys   = baked[name]["pos"]

            w(f"\tAnimation {{\n\t\t{{ {name} }}\n")

            w(f"\t\tAnimationKey {{\n\t\t\t0;\n\t\t\t{len(rot_keys)};\n")
            entries = [f"\t\t\t{fr};4;{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f};;"
                       for fr, (qw, qx, qy, qz) in sorted(rot_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            w(f"\t\tAnimationKey {{\n\t\t\t1;\n\t\t\t{len(scale_keys)};\n")
            entries = [f"\t\t\t{fr};3;{sx:.6f},{sy:.6f},{sz:.6f};;"
                       for fr, (sx, sy, sz) in sorted(scale_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            w(f"\t\tAnimationKey {{\n\t\t\t2;\n\t\t\t{len(pos_keys)};\n")
            entries = [f"\t\t\t{fr};3;{px:.6f},{py:.6f},{pz:.6f};;"
                       for fr, (px, py, pz) in sorted(pos_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            w("\t}\n")

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


def _mszip_wrap_x(payload: bytes, base_format: str = "bin") -> bytes:
    """Wrap a raw .x payload (text or binary token stream, WITHOUT the"""
    out = bytearray()
    if base_format == "bin":
        out += b"xof 0303bzip0032"
    else:
        out += b"xof 0303tzip0032"

    # Size prefix.  The bundled parser skips this if the bytes
    total_size = len(payload)
    first_chunk_uncomp = min(total_size, 32768) if total_size else 0
    out += total_size.to_bytes(4, "little")
    out += first_chunk_uncomp.to_bytes(2, "little")

    CHUNK_SIZE = 32768
    pos = 0
    while pos < len(payload):
        chunk = payload[pos:pos + CHUNK_SIZE]
        # Raw deflate (no zlib header/trailer) — wbits = -15
        co = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
        compressed = co.compress(chunk) + co.flush(zlib.Z_FINISH)
        # Per-chunk header: u16 compressed-size, then 'CK' magic
        out += len(compressed).to_bytes(2, "little")
        out += b"CK"
        out += compressed
        pos += CHUNK_SIZE
    # Empty payload edge case — still emit one empty chunk so the parser
    # doesn't trip on an unexpected EOF in the middle of its read loop.
    if not payload:
        co = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
        compressed = co.compress(b"") + co.flush(zlib.Z_FINISH)
        out += len(compressed).to_bytes(2, "little")
        out += b"CK"
        out += compressed
    return bytes(out)

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

def _write_mesh_frame(obj, out, indent,
                      depsgraph, use_mesh_modifiers,
                      export_normals, export_uvs,
                      export_materials, export_weights,
                      arm_obj, bl_to_dx_3, inv_scale,
                      triangulate, written_mats):
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

    _bm_check = bmesh.new()
    _bm_check.from_mesh(me_work)
    _bm_check.edges.ensure_lookup_table()
    non_manifold = [e.index for e in _bm_check.edges if not e.is_manifold]
    _bm_check.free()
    if non_manifold:
        msg = (f"{obj.name}: {len(non_manifold)} non-manifold edge(s) — "
               f"may cause broken normals or bad skinning "
               f"(edges: {non_manifold[:10]}"
               f"{'...' if len(non_manifold) > 10 else ''})")
        warnings.append(msg)

    conv_inv_3 = bl_to_dx_3

    bl_verts = me_work.vertices
    new_to_src = []
    faces = []
    loop_to_new_vi = [0] * len(me_work.loops)

    for poly in me_work.polygons:
        face_indices = []
        for li in poly.loop_indices:
            src_vi = me_work.loops[li].vertex_index
            new_vi = len(new_to_src)
            new_to_src.append(src_vi)
            loop_to_new_vi[li] = new_vi
            face_indices.append(new_vi)
        faces.append(face_indices)

    n_verts = len(new_to_src)
    n_faces = len(faces)

    verts_dx = [(conv_inv_3 @ Vector(bl_verts[src_vi].co)) * inv_scale
                for src_vi in new_to_src]

    loop_normals = _get_corner_normals(me_work) if export_normals else []

    frame_name = obj.get("_x_frame_name") or obj.name
    mesh_name  = obj.get("_x_mesh_name")  or (f"{obj.name}Geo"
                                               if not obj.name.endswith("Geo")
                                               else obj.name)
    stashed_frame_ftm = obj.get("_x_frame_ftm")

    w(f"{ind}Frame {frame_name} {{\n")
    w(f"{ind}\tFrameTransformMatrix {{\n")
    if stashed_frame_ftm is not None and len(stashed_frame_ftm) >= 16:
        ftm_str = ",".join(f"{float(v):.6f}" for v in stashed_frame_ftm[:16])
    else:
        if arm_obj and obj.parent == arm_obj:
            frame_mat = Matrix.Identity(4)
        else:
            frame_mat = _bl_bone_to_dx_world(obj.matrix_world, bl_to_dx_3, inv_scale)
        ftm_str = _mat4_to_dx(frame_mat)
    w(f"{ind}\t\t{ftm_str};;\n")
    w(f"{ind}\t}}\n")

    w(f"{ind}\tMesh {mesh_name} {{\n")

    w(f"{ind}\t\t{n_verts};\n")
    vert_lines = [f"{ind}\t\t {v.x:.6f}; {v.y:.6f}; {v.z:.6f};" for v in verts_dx]
    w(",\n".join(vert_lines) + ";\n")

    w(f"\n{ind}\t\t{n_faces};\n")
    face_lines = [f"{ind}\t\t{len(f)};{','.join(str(i) for i in f)};" for f in faces]
    w(",\n".join(face_lines) + ";\n")

    if export_normals and loop_normals:

        dx_loop_normals = [conv_inv_3 @ Vector(n) for n in loop_normals]

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
        w(",\n".join(f"{ind}\t\t\t {nx:.6f},{ny:.6f},{nz:.6f};"
                     for nx, ny, nz in unique_norms) + ";\n")
        w(f"\n{ind}\t\t\t{n_faces};\n")
        face_norm_lines = [f"{ind}\t\t\t{len(fni)};{','.join(str(i) for i in fni)};"
                           for fni in face_norm_idx]
        w(",\n".join(face_norm_lines) + ";\n")
        w(f"{ind}\t\t}}\n")

    if export_uvs and me_work.uv_layers:
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

    if export_materials and obj.material_slots:
        mats = [s.material for s in obj.material_slots if s.material]
        if mats:
            w(f"\n{ind}\t\tMeshMaterialList {{\n{ind}\t\t\t{len(mats)};\n{ind}\t\t\t{n_faces};\n")
            w(",\n".join(f"{ind}\t\t\t{p.material_index}" for p in me_work.polygons) + ";\n")
            for mat in mats:

                w(f"{ind}\t\t\t{{{mat.name}}}\n")
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
                if influences:
                    group_influences.append((vg, influences))

            if group_influences:
                n_groups = len(group_influences)

                w(f"\n{ind}\t\tXSkinMeshHeader {{\n")
                w(f"{ind}\t\t\t{n_groups};\n{ind}\t\t\t{n_groups};\n{ind}\t\t\t{n_groups};\n")
                w(f"{ind}\t\t}}\n")

                for vg, influences in group_influences:
                    bone = arm_obj.data.bones.get(vg.name)

                    if bone:
                        bind_dx = _bl_bone_to_dx_world(bone.matrix_local, bl_to_dx_3, inv_scale)
                        try:
                            offset = bind_dx.inverted()
                        except ValueError:
                            offset = Matrix.Identity(4)
                    else:
                        offset = Matrix.Identity(4)

                    w(f"\n{ind}\t\tSkinWeights {{\n")
                    w(f"{ind}\t\t\t\"{vg.name}\";\n")
                    w(f"{ind}\t\t\t{len(influences)};\n")
                    w(",\n".join(f"{ind}\t\t\t{vi}" for vi, _ in influences) + ";\n")
                    w(",\n".join(f"{ind}\t\t\t{wt:.6f}" for _, wt in influences) + ";\n")
                    w(f"{ind}\t\t\t{_mat4_to_dx(offset)};;\n")
                    w(f"{ind}\t\t}}\n")

    w(f"{ind}\t}}\n")
    w(f"{ind}}}\n")

    bpy.data.meshes.remove(me_work)
    return warnings

# =============================================================================
# Bugsnax .xcache (SEMS) binary writer + Blender bridge
# =============================================================================

XCACHE_MAGIC = b'SEMS'
XCACHE_FPS = 25.0          # constant across all observed files
XCACHE_VERSION = 1
XCACHE_FLAGS = 0x10101


def _write_header(out: bytearray, anim_frame_count: float, bone_count: int) -> None:
    """Write the 32-byte file header at offset 0."""
    out += XCACHE_MAGIC
    out += struct.pack('<II', 0, 0)                       # bytes 0x04..0x0C: zero
    out += struct.pack('<f', float(anim_frame_count))     # bytes 0x0C: anim frame count
    out += struct.pack('<f', XCACHE_FPS)                  # bytes 0x10: 25.0
    out += struct.pack('<I', XCACHE_VERSION)              # bytes 0x14: version
    out += struct.pack('<I', XCACHE_FLAGS)                # bytes 0x18: flags
    out += struct.pack('<I', bone_count)                  # bytes 0x1C: bone count


# ---------- Per-bone data block writer ----------

def _write_4x4_dx(out: bytearray, m16: List[float]) -> None:
    """Write 16 floats (a 4x4 matrix in row-major DirectX order)."""
    out += struct.pack('<16f', *m16)


def _write_3x4_plus_translation(out: bytearray, m16: List[float]) -> None:
    """Write the 64-byte bone bind-pose layout: 3x4 rotation + translation."""
    out += struct.pack('<16f', *m16)


def _invert_dx_matrix(m16):
    """Invert a 4x4 matrix stored as 16 floats in DirectX row-major form,
    returning the inverse in the same row-major form. Used to compute the
    SkinWeights matrixOffset (= inverse of the bone's world bind pose).

    Falls back to identity if the matrix is singular (very rare; usually
    means a bone with zero scale baked into its bind, like the Lid* bones
    on Apple/Celery — in which case identity is a sensible default).
    """
    # Treat the row-major DX form as a column-vec matrix by transposing,
    # invert in column-vec form, then transpose back to row-major.
    try:
        # Build col-vec matrix: m_col[r][c] = m16[c*4 + r]
        m_col = Matrix([
            [m16[0],  m16[4],  m16[8],  m16[12]],
            [m16[1],  m16[5],  m16[9],  m16[13]],
            [m16[2],  m16[6],  m16[10], m16[14]],
            [m16[3],  m16[7],  m16[11], m16[15]],
        ])
        inv_col = m_col.inverted()
        # Transpose back to row-major DX form
        out = []
        for r in range(4):
            for c in range(4):
                out.append(float(inv_col[c][r]))
        return out
    except Exception:
        # Identity fallback if matrix is singular
        return [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]


def _write_decoration_block(out: bytearray,
                              skin_offset: Optional[List[float]] = None,
                              bind_translation: Optional[Tuple[float, float, float]] = None) -> None:
    """Write the 104-byte block that follows the 3 matrix copies in each
    bone's data section.

    Layout (verified against original game xcache files):
        +0..+39   (40 bytes): 9 floats of 0.0 + 1 float of 1.0
        +40..+103 (64 bytes): the bone's SkinWeights matrixOffset
                              (= inverse of the bone's world-space bind pose,
                               in DirectX row-major form)

    For bones where we have the proper skin_offset matrix, write that
    directly. For backwards compatibility (e.g. mesh frames that aren't
    real bones), accept a bind_translation and synthesize an offset that
    matches the historical hardcoded pattern (identity rotation + inverse
    of the bind translation), which is what the old version of this
    function produced.
    """
    # 40-byte prefix: 9 zero floats + 1.0
    out += b'\x00' * 36
    out += struct.pack('<f', 1.0)

    # 64-byte skin offset matrix
    if skin_offset is not None and len(skin_offset) == 16:
        out += struct.pack('<16f', *skin_offset)
    else:
        # Fallback: identity rotation + inverse bind translation. This
        # matches what the original 100-byte hardcoded decoration block
        # produced for ROOT bones (and is correct for any bone whose bind
        # has no rotation).
        if bind_translation is None:
            bx = by = bz = 0.0
        else:
            bx, by, bz = bind_translation
        out += struct.pack('<16f',
                            1.0, 0.0, -0.0, 0.0,
                           -0.0, 1.0, -0.0, 0.0,
                           -0.0, -0.0, 1.0, 0.0,
                           -float(bx), -float(by), -float(bz), 1.0)


def _write_position_stride16(out: bytearray, pos_keys: dict, max_tick: int) -> int:
    """Write the position stride-16 block (lag-1 encoded)."""
    if not pos_keys or max_tick < 1:
        return 0

    n_written = 0

    # Step 1: determine an extension value past max_tick to use as "lookahead"
    # for the final record.  Use the last tick's value.
    last_tick = max(pos_keys.keys())
    last_pos = pos_keys[last_tick]

    # Sort ticks
    ticks = sorted(pos_keys.keys())

    # Build per-tick (X, Y, Z) tuples.  Insert tick 1 if missing (use first known).
    pos_at = {}
    for t in range(1, max_tick + 1):
        if t in pos_keys:
            pos_at[t] = pos_keys[t]
        else:
            # Use nearest available
            nearest = min(ticks, key=lambda k: abs(k - t))
            pos_at[t] = pos_keys[nearest]

    # Step 2: emit records.  For tick=N, write (Y[N-1], Z[N-1], N, X[N]).
    for tick in range(1, max_tick + 1):
        if tick == 1:
            # Seed values — observed pattern
            f0 = 1.0
            f1 = 0.0
        else:
            # Y and Z of PREVIOUS tick
            prev_x, prev_y, prev_z = pos_at[tick - 1]
            f0 = prev_y
            f1 = prev_z
        # Current tick's X is at f3
        cur_x, cur_y, cur_z = pos_at[tick]
        out += struct.pack('<4f', f0, f1, float(tick), cur_x)
        n_written += 16

    return n_written


def _write_scale_stride16(out: bytearray, scale_keys: dict, max_tick: int,
                          last_pos_tick: int, last_pos_y: float, last_pos_z: float
                          ) -> Tuple[int, bytes]:
    """Write the scale stride-16 block (lag-1 encoded, separate from position)."""
    # First write the transition entry between pos and scale blocks
    transition = (
        struct.pack('<2f', last_pos_y, last_pos_z) +
        struct.pack('<I', last_pos_tick) +
        struct.pack('<f', 1.0)
    )
    out += transition

    if not scale_keys or max_tick < 1:
        # Just the transition + closing transition; no real scale entries
        return 16, transition

    n_written = 16

    # Sort ticks; for missing ticks use nearest
    ticks = sorted(scale_keys.keys())
    scale_at = {}
    for t in range(1, max_tick + 1):
        if t in scale_keys:
            scale_at[t] = scale_keys[t]
        else:
            nearest = min(ticks, key=lambda k: abs(k - t))
            scale_at[t] = scale_keys[nearest]

    # Lag-1 encoding: scale entry tick=N stores scale FOR tick=N-1.
    for entry_tick in range(2, max_tick + 1):
        sx, sy, sz = scale_at[entry_tick - 1]
        out += struct.pack('<4f', sx, sy, sz, float(entry_tick))
        n_written += 16

    return n_written, transition


def _write_rotation_stride20(out: bytearray, rot_keys: dict, max_tick: int) -> int:
    """Write the rotation stride-20 block."""
    if not rot_keys or max_tick < 1:
        return 0

    n_written = 0

    ticks = sorted(rot_keys.keys())
    rot_at = {}
    for t in range(1, max_tick + 1):
        if t in rot_keys:
            rot_at[t] = rot_keys[t]
        else:
            nearest = min(ticks, key=lambda k: abs(k - t))
            rot_at[t] = rot_keys[nearest]

    for tick in range(1, max_tick + 1):
        qx, qy, qz, qw = rot_at[tick]
        # Negate to match xcache convention (importer also negates on read)
        out += struct.pack('<5f', float(tick), -qx, -qy, -qz, -qw)
        n_written += 20

    return n_written


def _write_skin_weights(out: bytearray, skin: List[Tuple[int, float]]) -> int:
    """Write skin weights for one bone."""
    n = len(skin)
    out += struct.pack('<I', n)
    out += struct.pack('<H', 0)  # pad after count
    n_written = 6
    for vi, weight in skin:
        # 10 bytes per entry: u16 vi, u16 pad, f32 weight, u16 pad2
        out += struct.pack('<HHfH', int(vi) & 0xFFFF, 0, float(weight), 0)
        n_written += 10
    return n_written


def _write_bone_block_alignment_trailer(out: bytearray, block_start_byte_count: int,
                                         data_start_alignment: int) -> int:
    """Write the small alignment trailer at the end of each bone's data block."""
    # Compute current position relative to file start; the alignment we want is
    current_len = len(out)
    # We want (current_len + trailer_size) mod 4 == data_start_alignment mod 4
    # Solve for trailer_size in {0, 1, 2, 3}.
    target = data_start_alignment % 4
    trailer_size = (target - current_len) % 4
    # Empirically the format always writes exactly 2 trailer bytes for animated
    out += b'\x00' * trailer_size
    return trailer_size


def _write_no_anim_placeholder(out: bytearray) -> int:
    """Write the 16-byte 'anim placeholder' that appears in non-animated bones"""
    out += struct.pack('<4f', 1.0, 0.0, 0.0, 0.0)
    return 16


# ---------- Mesh-block writers ----------

# Constant 144-byte common material header from offset +0 of post-bone-header
_MESH_COMMON_HEADER_PROLOGUE = bytes.fromhex(
    # +0..+27: 1.0 + 27 zeros
    '0000803f' +
    '00' * 28 +
    # +32..+35: u32 = 1
    '01000000' +
    # +36..+39: u32 = 2
    '02000000' +
    # +40..+47: 8 zeros
    '00' * 8
)
assert len(_MESH_COMMON_HEADER_PROLOGUE) == 48

# Bytes after the bbox (+68..+143 = 76 bytes)
_MESH_COMMON_HEADER_EPILOGUE = bytes.fromhex(
    # +68..+71: f32 = 1.0
    '0000803f' +
    # +72..+75: zeros
    '00000000' +
    # +76..+91: 16 zeros
    '00' * 16 +
    # +92..+95: f32 = 1.0
    '0000803f' +
    # +96..+111: 16 zeros
    '00' * 16 +
    # +112..+115: f32 = 1.0
    '0000803f' +
    # +116..+127: 12 zeros
    '00' * 12 +
    # +128..+131: f32 = 1.0
    '0000803f' +
    # +132..+143: 12-byte color sentinel (white, white, RGBA(0,0,0,255))
    'ffffffff' 'ffffffff' '000000ff'
)
assert len(_MESH_COMMON_HEADER_EPILOGUE) == 76

# Color trailer: 0x808080FF (gray RGBA) — last 4 bytes of common header.
_MESH_COMMON_HEADER_COLOR_TAIL = bytes.fromhex('808080ff')

# Post-color header for textured meshes: 16 bytes before the first texture entry.
# (specular power 128.0 + 8 zeros + flag float 1.0)
_MESH_TEX_PROLOGUE = bytes.fromhex(
    '00000043' +    # f32 = 128.0
    '00' * 8 +
    '0000803f'      # f32 = 1.0
)
assert len(_MESH_TEX_PROLOGUE) == 16

# Constant 60-byte tail for textured meshes.  Includes the trailing separator
_MESH_TEX_TAIL = bytes.fromhex(
    # +0..+13: 14 zero bytes (incl. last-texture inter-separator absorbed into tail)
    '00' * 14 +
    # +14: 0x00
    '00' +
    # +15..+18: 01 01 01 01
    '01010101' +
    # +19..+22: 00 00 00 01
    '00000001' +
    # +23..+26: 01 0f 01 00
    '010f0100' +
    # +27..+30: 4 zeros
    '00000000' +
    # +31..+34: 0xFFFFFFFF sentinel
    'ffffffff' +
    # +35: 0x01 marker
    '01' +
    # +36..+39: f32 = 1.0
    '0000803f' +
    # +40..+43: f32 = 1.0
    '0000803f' +
    # +44..+59: 16 zeros
    '00' * 16
)
assert len(_MESH_TEX_TAIL) == 60


def _compute_bbox(verts):
    """Return ((min_x, min_y, min_z), (max_x, max_y, max_z)) for a list of"""
    if not verts:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _write_mesh_common_header(out: bytearray, bbox_min, bbox_max) -> None:
    """Write the 144-byte common material header that appears immediately"""
    out += _MESH_COMMON_HEADER_PROLOGUE
    out += struct.pack('<6f', *bbox_min, *bbox_max)
    out += _MESH_COMMON_HEADER_EPILOGUE
    out += _MESH_COMMON_HEADER_COLOR_TAIL


def _write_mesh_texture_entry(out: bytearray, path: str,
                               include_separator: bool) -> int:
    """Write one texture entry (presence flag + name_len + name + null) and"""
    name_bytes = path.encode('ascii')
    out += struct.pack('<B', 0x01)
    out += struct.pack('<I', len(name_bytes))
    out += name_bytes
    out += struct.pack('<B', 0x00)        # null terminator
    written = 1 + 4 + len(name_bytes) + 1
    if include_separator:
        out += struct.pack('<B', 0x00)    # alignment
        out += struct.pack('<I', 1)       # separator
        written += 5
    return written


def _write_mesh_block(out: bytearray, mesh: dict) -> None:
    """Write a complete mesh block (parent_idx + header + FTM + material/flags"""
    # Bone-style header
    name_bytes = mesh['name'].encode('ascii')
    out += struct.pack('<II', int(mesh.get('parent_idx', 0)), len(name_bytes))
    out += name_bytes
    out += struct.pack('<16f', *mesh['ftm'])

    # 296-byte standard bone-style pre-anim header: 3 matrix copies (192)
    # + 40-byte prefix + 64-byte skin offset matrix (104) = 296 bytes.
    ftm = mesh['ftm']
    bind_translation = (ftm[12], ftm[13], ftm[14])
    out += struct.pack('<16f', *ftm)   # bind copy 1
    out += struct.pack('<16f', *ftm)   # bind copy 2
    out += struct.pack('<16f', *ftm)   # bind copy 3 (= FTM)
    # Mesh frames are not real bones; they have no SkinWeights matrixOffset.
    # Use the bind_translation fallback path (= identity rotation + inverse
    # translation), which is what the original game files store here.
    _write_decoration_block(out, bind_translation=bind_translation)

    # 144-byte common material header
    bbox_min, bbox_max = _compute_bbox(mesh.get('verts', []))
    _write_mesh_common_header(out, bbox_min, bbox_max)

    # Texture section
    tex_paths = mesh.get('tex_paths') or []
    if tex_paths:
        out += _MESH_TEX_PROLOGUE
        # Each texture entry, all followed by the separator (the separator
        for i, path in enumerate(tex_paths):
            include_sep = (i < len(tex_paths) - 1)
            _write_mesh_texture_entry(out, path, include_separator=include_sep)
        # Fixed 60-byte tail (begins with the post-last-texture separator)
        out += _MESH_TEX_TAIL
    else:
        # No textures: just 2 bytes of zero alignment to round out the block
        out += b'\x00\x00'

    # Vertex stream
    verts = mesh.get('verts', [])
    normals = mesh.get('normals', [])
    uvs = mesh.get('uvs', [])
    n_verts = len(verts)
    out += struct.pack('<I', n_verts)

    nan_w = struct.unpack('<f', b'\xff\xff\xff\xff')[0]   # quiet NaN as in source files
    for vi in range(n_verts):
        vx, vy, vz = verts[vi]
        if vi < len(normals):
            nx, ny, nz = normals[vi]
        else:
            nx, ny, nz = 0.0, 1.0, 0.0
        if vi < len(uvs):
            u, v = uvs[vi]
        else:
            u, v = 0.0, 0.0
        # 16 floats per vertex
        out += struct.pack('<16f',
            float(vx), float(vy), float(vz),         # [0..2] position
            float(nx), float(ny), float(nz),         # [3..5] normal
            nan_w,                                    # [6]    tangent W (NaN)
            float(u), float(v),                       # [7..8] UV
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # [9..15] reserved
        )

    # Index buffer
    faces = mesh.get('faces', [])
    total_indices = sum(len(f) for f in faces)
    out += struct.pack('<II', 0, total_indices)
    for face in faces:
        for vi in face:
            out += struct.pack('<H', int(vi) & 0xFFFF)


# ---------- Bone-block top-level writer ----------

def _write_bone_block(out: bytearray, bone: dict, animated: bool, max_tick: int) -> None:
    """Write a complete bone block including header (parent_idx, name_len, name,"""
    # Bone header: parent_idx, name_len, name
    name_bytes = bone['name'].encode('ascii')
    out += struct.pack('<II', int(bone['parent_idx']), len(name_bytes))
    out += name_bytes

    # FrameTransformMatrix (parent-local, 64 bytes)
    _write_4x4_dx(out, bone['ftm'])

    # data_start position (for trailer alignment calculation)
    data_start_byte = len(out)
    data_start_align = data_start_byte % 4

    # Per-bone data block:
    bind = bone['bind_pose']
    bind_translation = (bind[12], bind[13], bind[14])

    # 3 copies: bind, bind, FTM (the third one is parent-local, matches frame-1
    # walk pose for animated bones; for non-animated bones it's same as bind).
    _write_3x4_plus_translation(out, bind)
    _write_3x4_plus_translation(out, bind)
    _write_3x4_plus_translation(out, bone['ftm'])

    # 104-byte block: 40-byte prefix + 64-byte SkinWeights matrixOffset
    # (= inverse of the bone's world-space bind pose). This is read back
    # by the parser at ftm_offset + 296 and used as the SkinWeights offset
    # matrix for skinning. Without the proper offset matrix, vertices
    # weighted to non-root bones get the bone's full world transform
    # applied without subtracting their bind position, producing scattered
    # bones and wrong mesh deformation during animation.
    skin_offset = bone.get('skin_offset')
    if skin_offset is None:
        skin_offset = _invert_dx_matrix(bind)
    _write_decoration_block(out, skin_offset=skin_offset,
                              bind_translation=bind_translation)

    # Animation channels (only if animated)
    if animated and max_tick >= 1:
        # Position block
        _write_position_stride16(out, bone.get('pos_keys', {}), max_tick)

        # Find last position record's f0,f1 for the transition entry between
        pos_keys = bone.get('pos_keys', {})
        if pos_keys and max_tick >= 2:
            ticks = sorted(pos_keys.keys())
            prev_t = max_tick - 1
            if prev_t in pos_keys:
                _, ly, lz = pos_keys[prev_t]
            else:
                nearest = min(ticks, key=lambda k: abs(k - prev_t))
                _, ly, lz = pos_keys[nearest]
        elif pos_keys:
            ticks = sorted(pos_keys.keys())
            _, ly, lz = pos_keys[ticks[0]]
        else:
            ly = 0.0
            lz = 0.0

        # Write scale block (preceded by the transition entry)
        _write_scale_stride16(out, bone.get('scale_keys', {}), max_tick,
                              last_pos_tick=max_tick,
                              last_pos_y=ly, last_pos_z=lz)

        # Closing transition entry between scale block and rotation block.
        # Empirically observed as 16 bytes of zeros.
        out += struct.pack('<4f', 0.0, 0.0, 0.0, 0.0)

        # Rotation block
        _write_rotation_stride20(out, bone.get('rot_keys', {}), max_tick)
    else:
        # Non-animated bone: write the 16-byte placeholder block instead of
        # animation records.
        _write_no_anim_placeholder(out)

    # Skin weights section (always present, may be empty)
    _write_skin_weights(out, bone.get('skin', []))

    # Bone block alignment trailer (variable size: typically 2 bytes for animated
    # bones, more for non-animated to maintain mod-4 alignment per bone).
    _write_bone_block_alignment_trailer(out, data_start_byte, data_start_align)


# ---------- Top-level export entry point ----------

def encode_xcache_bytes(bones: List[dict], anim_frame_count: int,
                         meshes: Optional[List[dict]] = None) -> bytes:
    """Build a complete .xcache file as bytes from a list of bone dicts and"""
    out = bytearray()

    n_meshes = len(meshes) if meshes else 0
    total_entries = len(bones) + n_meshes

    # Header — bone_count includes Geo "bones" (mesh entries)
    _write_header(out, anim_frame_count, total_entries)

    # Each bone
    animated = anim_frame_count > 0
    max_tick = int(anim_frame_count) if animated else 0
    for bone in bones:
        _write_bone_block(out, bone, animated=animated, max_tick=max_tick)

    # Each mesh (built from scratch, not spliced)
    if meshes:
        for mesh in meshes:
            _write_mesh_block(out, mesh)

    return bytes(out)


def export_xcache_to_file(filepath: str, bones: List[dict], anim_frame_count: int,
                          meshes: Optional[List[dict]] = None) -> int:
    """Write an .xcache file to disk.  Returns the number of bytes written."""
    data = encode_xcache_bytes(bones, anim_frame_count, meshes=meshes)
    with open(filepath, 'wb') as f:
        f.write(data)
    return len(data)


# Blender bridge

def _matrix_to_dx_floats(mat):
    """Convert a mathutils.Matrix (4x4) into a list of 16 floats in"""
    # Transpose: DX row-major = Blender column-major when read sequentially
    out = []
    for c in range(4):
        for r in range(4):
            out.append(float(mat[r][c]))
    return out


def _build_xcache_parent_idx(bones_in_order, idx, parent_name):
    """Compute the relative parent_idx field for a bone at index `idx`"""
    if idx == 0:
        return 2
    if idx == 1:
        return 28
    # Find parent's index
    if parent_name is None:
        return 0   # parent is root
    if idx >= 1 and bones_in_order[idx - 1].name == parent_name:
        return 1   # parent is previous bone in the list
    if bones_in_order[0].name == parent_name:
        return 0   # parent is root
    # Fall back to root sentinel — engine may complain but at least file parses
    return 0


def _topo_sort_bones(armature_bones):
    """Order Blender armature bones such that:"""
    # Find roots
    roots = [b for b in armature_bones if b.parent is None]
    if not roots:
        return list(armature_bones)
    # If there are multiple roots, we put the first as bone[0] and treat the
    # rest as bone[0]'s children semantically (parent_idx == 0).
    ordered = []
    visited = set()

    def visit(bone):
        if bone.name in visited:
            return
        visited.add(bone.name)
        ordered.append(bone)
        for child in bone.children:
            visit(child)

    for r in roots:
        visit(r)
    # Catch any orphans (shouldn't happen but be safe)
    for b in armature_bones:
        if b.name not in visited:
            ordered.append(b)
            visited.add(b.name)
    return ordered


def collect_bones_from_armature(arm_obj, bl_to_dx_3, inv_scale, scene=None,
                                 anim_frame_start=1, anim_frame_end=1,
                                 export_animation=False):
    """Extract bone dicts (suitable for encode_xcache_bytes) from a Blender"""

    arm_data = arm_obj.data
    bones = list(arm_data.bones)
    ordered = _topo_sort_bones(bones)
    name_to_idx = {b.name: i for i, b in enumerate(ordered)}

    # Build bone dicts (without animation first)
    bone_dicts = []
    for idx, bone in enumerate(ordered):
        # World bind pose
        if bone.parent is None:
            bind_world = _bl_bone_to_dx_world(bone.matrix_local, bl_to_dx_3, inv_scale)
        else:
            # World matrix in Blender, then convert
            bind_world = _bl_bone_to_dx_world(bone.matrix_local, bl_to_dx_3, inv_scale)
        bind_floats = _matrix_to_dx_floats(bind_world)

        # FrameTransformMatrix (parent-local).  For non-animated cases this is
        if bone.parent is None:
            ftm_local = bind_world
        else:
            parent_world = _bl_bone_to_dx_world(bone.parent.matrix_local, bl_to_dx_3, inv_scale)
            ftm_local = parent_world.inverted() @ bind_world
        ftm_floats = _matrix_to_dx_floats(ftm_local)

        parent_name = bone.parent.name if bone.parent else None
        parent_idx = _build_xcache_parent_idx(ordered, idx, parent_name)

        bone_dicts.append({
            'name': bone.name,
            'parent_idx': parent_idx,
            'ftm': ftm_floats,
            'bind_pose': bind_floats,
            'skin_offset': _invert_dx_matrix(bind_floats),
            'pos_keys': {},
            'scale_keys': {},
            'rot_keys': {},
            'skin': [],
        })

    # Animation baking
    anim_frame_count = 0
    if export_animation and scene is not None:
        orig_frame = scene.frame_current
        n_frames = anim_frame_end - anim_frame_start + 1
        if n_frames < 1:
            n_frames = 0
        anim_frame_count = n_frames

        if n_frames >= 1:
            for tick, fr in enumerate(range(anim_frame_start, anim_frame_end + 1), start=1):
                scene.frame_set(fr)
                for pb in arm_obj.pose.bones:
                    if pb.name not in name_to_idx:
                        continue
                    bone_idx = name_to_idx[pb.name]
                    world_bl = pb.matrix.copy()
                    if pb.parent:
                        parent_world = pb.parent.matrix.copy()
                        local_bl = parent_world.inverted() @ world_bl
                        dx_local = local_bl
                        rot = dx_local.to_3x3()
                        t   = dx_local.to_translation()
                        s   = dx_local.to_scale()
                        q   = rot.to_quaternion()
                    else:
                        # ROOT: rotation is the pose-bone-local offset
                        # (matrix_basis); position/scale come from world.
                        # The importer's local_rest_q for a skel-root is the
                        # identity quaternion, so it interprets the file's
                        # stored quat as the pose offset directly — round-trip
                        # therefore requires writing pose_q, not the bone's
                        # absolute world rotation.
                        dx_world = _bl_bone_to_dx_world(world_bl, bl_to_dx_3, inv_scale)
                        t = dx_world.to_translation()
                        s = dx_world.to_scale()
                        q = pb.matrix_basis.to_3x3().to_quaternion()
                    # The .xcache animation rotation convention stores the
                    bone_dicts[bone_idx]['rot_keys']  [tick] = (-q.x, -q.y, -q.z, q.w)
                    bone_dicts[bone_idx]['scale_keys'][tick] = (s.x, s.y, s.z)
                    bone_dicts[bone_idx]['pos_keys']  [tick] = (t.x, t.y, t.z)
            scene.frame_set(orig_frame)

    return bone_dicts, anim_frame_count


def collect_skin_weights(mesh_dicts, arm_obj, bone_dicts):
    """Populate the 'skin' field of each bone dict from the mesh dicts"""
    name_to_idx = {bd['name']: i for i, bd in enumerate(bone_dicts)}
    vertex_offset = 0

    for mesh_dict in mesh_dicts:
        mesh_obj = mesh_dict.get('_blender_obj')
        bl_to_export = mesh_dict.get('_bl_to_export')
        if mesh_obj is None or bl_to_export is None:
            # Mesh dict wasn't produced by our blender-side helper — skip
            vertex_offset += len(mesh_dict.get('verts', []))
            continue

        # Check this mesh is bound to arm_obj (skip otherwise — verts won't
        # have meaningful weights against this armature).
        bound = False
        if mesh_obj.parent is arm_obj:
            bound = True
        for mod in mesh_obj.modifiers:
            if mod.type == 'ARMATURE' and getattr(mod, 'object', None) is arm_obj:
                bound = True
                break
        if not bound:
            vertex_offset += len(mesh_dict.get('verts', []))
            continue

        me = mesh_obj.data
        # Map vertex_group index → bone name
        vg_to_bone = {}
        for vg in mesh_obj.vertex_groups:
            if vg.name in name_to_idx:
                vg_to_bone[vg.index] = vg.name

        # Iterate Blender source vertices.  For each weight, emit one skin
        for bl_vi, v in enumerate(me.vertices):
            if bl_vi >= len(bl_to_export):
                continue
            export_vis = bl_to_export[bl_vi]
            if not export_vis:
                continue
            for grp in v.groups:
                bone_name = vg_to_bone.get(grp.group)
                if bone_name is None:
                    continue
                w = float(grp.weight)
                if w <= 0.0:
                    continue
                bone_skin = bone_dicts[name_to_idx[bone_name]]['skin']
                for export_vi in export_vis:
                    bone_skin.append((vertex_offset + export_vi, w))

        vertex_offset += len(mesh_dict.get('verts', []))


def collect_meshes_from_blender(mesh_objs, arm_obj, bl_to_dx_3, inv_scale,
                                  depsgraph=None, use_modifiers=True):
    """Walk Blender mesh objects and produce mesh dicts ready for"""
    from mathutils import Matrix, Vector
    import bmesh
    import bpy

    meshes = []
    for obj in mesh_objs:
        # IMPORTANT: use the raw obj.data (bind-pose vertices), NOT the
        me_src = obj.data
        obj_eval = None

        # Triangulate via bmesh into a temp mesh so we have triangles only.
        bm = bmesh.new()
        bm.from_mesh(me_src)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        me_work = bpy.data.meshes.new('_xcache_export_tmp')
        bm.to_mesh(me_work)
        bm.free()
        me_work.update()
        me_work.calc_loop_triangles()

        n_verts_blender = len(me_work.vertices)

        # ---- Vertex unrolling at UV seams ----
        uv_layer = me_work.uv_layers.active.data if me_work.uv_layers.active else None

        # Per-vert distinct UVs.  Each entry: list of (uv_tuple, [loop_idx, ...]).
        UV_ROUND = 6
        per_vert_uv_groups = [[] for _ in range(n_verts_blender)]
        # loop -> (blender_vi, uv_group_idx_within_that_vert)
        loop_uv_assignment = [None] * len(me_work.loops)

        if uv_layer is not None:
            for loop in me_work.loops:
                vi = loop.vertex_index
                raw_uv = uv_layer[loop.index].uv
                uv_key = (round(float(raw_uv[0]), UV_ROUND),
                          round(float(raw_uv[1]), UV_ROUND))
                groups = per_vert_uv_groups[vi]
                # Find existing group with matching UV
                gi = -1
                for k, (existing_key, _loops) in enumerate(groups):
                    if existing_key == uv_key:
                        gi = k
                        break
                if gi < 0:
                    gi = len(groups)
                    groups.append((uv_key, [loop.index]))
                else:
                    groups[gi][1].append(loop.index)
                loop_uv_assignment[loop.index] = (vi, gi)
        else:
            # No UVs: each vert gets one (synthetic) group with no loops
            for vi in range(n_verts_blender):
                per_vert_uv_groups[vi].append(((0.0, 0.0), []))
            for loop in me_work.loops:
                loop_uv_assignment[loop.index] = (loop.vertex_index, 0)

        # Ensure every Blender vert is represented even if no loop refers to
        # it (e.g. an isolated vert).  This keeps skin-weight indexing stable.
        for vi in range(n_verts_blender):
            if not per_vert_uv_groups[vi]:
                per_vert_uv_groups[vi].append(((0.0, 0.0), []))

        # Build the export vertex array and the bl_vi -> [export_vi, ...] map.
        verts = []
        normals = []
        uvs = []
        bl_to_export = [[] for _ in range(n_verts_blender)]
        # (bl_vi, group_idx) -> export_vi
        group_to_export = {}
        for bl_vi in range(n_verts_blender):
            v = me_work.vertices[bl_vi]
            co = v.co
            n = v.normal
            dx_co = (bl_to_dx_3 @ Vector(co)) * inv_scale
            dx_n = bl_to_dx_3 @ Vector(n)
            for gi, (uv_key, _loops) in enumerate(per_vert_uv_groups[bl_vi]):
                export_vi = len(verts)
                verts.append((dx_co.x, dx_co.y, dx_co.z))
                normals.append((dx_n.x, dx_n.y, dx_n.z))
                # Importer applies (u, 1.0 - v) when reading UV from the
                # XNode tree; invert that here so the round-trip matches.
                uvs.append((uv_key[0], 1.0 - uv_key[1]))
                bl_to_export[bl_vi].append(export_vi)
                group_to_export[(bl_vi, gi)] = export_vi

        # Per-vertex normals.  Use the mesh's vertex normals (smooth shading
        try:
            me_work.calc_normals_split()
        except Exception:
            pass

        # Faces: for each loop in the triangle, pick the export vert that
        # matches the loop's UV.
        faces = []
        for tri in me_work.loop_triangles:
            tri_verts = []
            for li in tri.loops:
                assignment = loop_uv_assignment[li]
                if assignment is None:
                    # Fall back to the first export vert for this Blender vi
                    bl_vi = me_work.loops[li].vertex_index
                    tri_verts.append(bl_to_export[bl_vi][0])
                else:
                    bl_vi, gi = assignment
                    tri_verts.append(group_to_export[(bl_vi, gi)])
            faces.append(tuple(tri_verts))

        # FTM matrix: identity for mesh frames in observed files (the mesh
        # geometry is already in armature/world space when emitted).
        ftm = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

        # Texture paths: pull from material slots if any of them have a
        tex_paths = []
        seen_paths = set()
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            stashed = mat.get('_x_texture_filename')
            if isinstance(stashed, str) and stashed and stashed not in seen_paths:
                tex_paths.append(stashed)
                seen_paths.add(stashed)
                continue
            try:
                tree = mat.node_tree
                if tree is not None:
                    picked = None
                    for node in tree.nodes:
                        if node.type != 'TEX_IMAGE':
                            continue
                        # Per-node engine-format path takes priority
                        node_stash = node.get('_x_texture_filename')
                        if isinstance(node_stash, str) and node_stash:
                            picked = node_stash
                            break
                        if node.image is not None:
                            p = node.image.filepath_raw or node.image.filepath or ''
                            if p:
                                picked = p
                                break
                    if picked and picked not in seen_paths:
                        tex_paths.append(picked)
                        seen_paths.add(picked)
            except Exception:
                pass

        # Mesh name: prefer the importer-stashed _x_mesh_name; else derive
        # from object name (append 'Geo' so it matches the engine convention).
        mesh_name = (obj.data.get('_x_mesh_name')
                     or obj.get('_x_mesh_name')
                     or obj.name)
        if not mesh_name.endswith('Geo'):
            mesh_name = mesh_name + 'Geo'

        meshes.append({
            'name': mesh_name,
            'parent_idx': 0,
            'ftm': ftm,
            'verts': verts,
            'normals': normals,
            'uvs': uvs,
            'faces': faces,
            'tex_paths': tex_paths,
            # Internal helpers consumed by collect_skin_weights so it can
            '_bl_to_export': bl_to_export,
            '_blender_obj': obj,
        })

        # Cleanup
        bpy.data.meshes.remove(me_work)
        if obj_eval is not None:
            obj_eval.to_mesh_clear()

    return meshes


def export_xcache_from_blender(context, filepath,
                                use_selection=False,
                                use_mesh_modifiers=True,
                                global_scale=1.0,
                                axis_forward="-Z",
                                axis_up="Y",
                                export_armature=True,
                                export_weights=True,
                                export_animation=True,
                                anim_frame_start=1,
                                anim_frame_end=250,
                                **_):
    """Top-level entry point for exporting the current Blender scene state to"""
    import math
    from .exporter import _axis_matrix

    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    objects = (context.selected_objects
               if use_selection else list(context.scene.objects))

    # The importer applies M_imp = axis_fix @ axis_matrix_IMPORTER to incoming
    axis_base = _axis_matrix(axis_forward, axis_up)
    axis_fix = Matrix.Rotation(math.pi, 4, 'Z')
    conv_mat_4 = axis_base @ axis_fix
    bl_to_dx_3 = conv_mat_4.to_3x3()
    inv_scale = 1.0 / global_scale if global_scale != 0.0 else 1.0

    armature_objs = [o for o in objects if o.type == 'ARMATURE']
    mesh_objs = [o for o in objects if o.type == 'MESH']

    if not armature_objs and not mesh_objs:
        return {'CANCELLED'}, ['No armature or mesh found in scene; nothing to export.']

    warnings = []

    # Skeleton + animation
    if armature_objs:
        arm_obj = armature_objs[0]
        bone_dicts, anim_frame_count = collect_bones_from_armature(
            arm_obj, bl_to_dx_3, inv_scale, scene=scene,
            anim_frame_start=anim_frame_start,
            anim_frame_end=anim_frame_end,
            export_animation=export_animation,
        )

        # Drop any "mesh-bone" entries from the skeletal bone list — those
        bone_dicts = [bd for bd in bone_dicts
                      if not _is_mesh_bone_name(bd['name'])]
    else:
        # No armature: write an empty bone list with a single sentinel root.
        arm_obj = None
        bone_dicts = []
        anim_frame_count = 0
        warnings.append(
            "No armature found — exported file has no skeleton.  Most "
            "Bugsnax assets expect a skeleton; the engine may reject this."
        )

    # Mesh entries built from current scene (NOT spliced from any source file).
    mesh_dicts = []
    if mesh_objs:
        try:
            mesh_dicts = collect_meshes_from_blender(
                mesh_objs, arm_obj, bl_to_dx_3, inv_scale,
                depsgraph=depsgraph, use_modifiers=use_mesh_modifiers,
            )
        except Exception as e:
            warnings.append(
                f"Mesh export failed: {e}.  Exported file has skeleton + "
                f"animation only (no geometry)."
            )
            mesh_dicts = []

    # Skin weights need both bone_dicts and mesh_dicts to exist
    if armature_objs and export_weights and mesh_dicts:
        collect_skin_weights(mesh_dicts, arm_obj, bone_dicts)

    # Strip internal-only keys before encoding so the encoder doesn't trip
    # on Blender objects that have no place in the output stream.
    for md in mesh_dicts:
        md.pop('_bl_to_export', None)
        md.pop('_blender_obj', None)

    n_bytes = export_xcache_to_file(filepath, bone_dicts, anim_frame_count,
                                     meshes=mesh_dicts or None)

    if mesh_dicts:
        total_verts = sum(len(m['verts']) for m in mesh_dicts)
        total_tris = sum(len(m['faces']) for m in mesh_dicts)
        warnings.append(
            f"Wrote {len(bone_dicts)} bone(s), {anim_frame_count} animation "
            f"frame(s), {len(mesh_dicts)} mesh(es) ({total_verts} vertices, "
            f"{total_tris} triangles). Material/flags block uses default "
            f"values — texture paths preserved from material custom "
            f"properties / image nodes if present."
        )

    return {'FINISHED'}, warnings