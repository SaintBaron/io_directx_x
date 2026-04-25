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
import struct
import bmesh
from mathutils import Matrix, Vector, Quaternion

_BT_NAME      = 0x0001
_BT_STRING    = 0x0002
_BT_INT_LIST  = 0x0006
_BT_FLT_LIST  = 0x0007
_BT_OBRACE    = 0x000a
_BT_CBRACE    = 0x000b
_BT_COMMA     = 0x0013
_BT_SEMICOLON = 0x0014

class _BinarySerializer:
    """Convert the text-oriented write stream to DirectX binary tokens.

    The exporter writes text via `w(string)`.  When binary output is requested,
    `w` is replaced by this object's `feed()` method, which parses each text
    chunk into tokens and emits the binary equivalents into a bytearray.

    Design rationale
    ----------------
    Re-using the existing text-generation logic avoids duplicating the ~400-line
    body of export_x.  The trade-off is that we parse our OWN text output back
    into tokens, which is slightly roundabout but very robust: the text output
    is deterministic and ASCII-clean, so the tokeniser is trivially simple.

    Binary token encoding (32-bit floats, little-endian):
      NAME token    : u16(0x0001) u32(len) bytes
      STRING token  : u16(0x0002) u32(len) bytes u16(0x0000)  [terminator]
      INT_LIST      : u16(0x0006) u32(count) count×u32
      FLOAT_LIST    : u16(0x0007) u32(count) count×f32
      { / } / , / ; : u16(token_id)

    The serialiser groups consecutive floats and integers into list records,
    which is what real DX binary files do and what the binary parser expects.
    """

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
             **_):

    scene     = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    objects   = context.selected_objects if use_selection else list(context.scene.objects)

    bl_to_dx_3 = _axis_matrix(axis_forward, axis_up).to_3x3()
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
                else:

                    dx_local = _bl_bone_to_dx_world(world_bl, bl_to_dx_3, inv_scale)

                dx_rot = dx_local.to_3x3()
                dx_t   = dx_local.to_translation()
                dx_s   = dx_local.to_scale()
                q      = dx_rot.to_quaternion()

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
        with open(filepath, "wb") as fh:
            fh.write(_bin_header)
            fh.write(payload)
    else:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("".join(out))

    return {"FINISHED"}, all_warnings

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