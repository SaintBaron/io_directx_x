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

template Frame {
 <3d82ab46-62da-11cf-ab39-0020af71e433>
 [...]
}

template Matrix4x4 {
 <f6f23f45-7686-11cf-8f52-0040333594a3>
 array FLOAT matrix[16];
}

template FrameTransformMatrix {
 <f6f23f41-7686-11cf-8f52-0040333594a3>
 Matrix4x4 frameMatrix;
}

template Vector {
 <3d82ab5e-62da-11cf-ab39-0020af71e433>
 FLOAT x;
 FLOAT y;
 FLOAT z;
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

template MeshFaceWraps {
 <ed1ec5c0-c0a8-11d0-941c-0080c80cfa7b>
 DWORD nFaceWrapValues;
 array Boolean2d faceWrapValues[nFaceWrapValues];
}

template MeshTextureCoords {
 <f6f23f40-7686-11cf-8f52-0040333594a3>
 DWORD nTextureCoords;
 array Coords2d textureCoords[nTextureCoords];
}

template Coords2d {
 <f6f23f44-7686-11cf-8f52-0040333594a3>
 FLOAT u;
 FLOAT v;
}

template MeshNormals {
 <f6f23f43-7686-11cf-8f52-0040333594a3>
 DWORD nNormals;
 array Vector normals[nNormals];
 DWORD nFaceNormals;
 array MeshFace faceNormals[nFaceNormals];
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

template Animation {
 <3d82ab4f-62da-11cf-ab39-0020af71e433>
 [...]
}

template AnimationSet {
 <3d82ab50-62da-11cf-ab39-0020af71e433>
 [Animation <3d82ab4f-62da-11cf-ab39-0020af71e433>]
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
}"""

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
             anim_key_format="TRS",
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

    if binary_format:

        _bin_header = b"xof 0303bin 0032"

        w(f"AnimTicksPerSecond {{ {file_ticks_per_second}; }}\n")
    else:
        w("xof 0303txt 0032\n\n")
        # Emit DirectX templates. Prefer the stashed source-file copy
        # for byte-identical round-trips; fall back to the canonical
        # set when there's nothing stashed (e.g. a from-scratch model
        # in Blender) so that high-precision-style engines that expect templates
        # at the top of the file still get them.
        if _passthrough_templates:
            w(_passthrough_templates)
            w("\n\n")
        elif pz_compat:
            w(_DEFAULT_TEMPLATES)
            w("\n\n")
        w(f"AnimTicksPerSecond {{\n\t{file_ticks_per_second};\n}}\n")

    mesh_objs     = [o for o in objects if o.type == "MESH"]
    armature_objs = [o for o in objects if o.type == "ARMATURE"]
    arm_obj       = armature_objs[0] if armature_objs else None

    written_mats = {}
    if export_materials:
        for obj in mesh_objs:
            for mat in _effective_materials(obj):
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

    # Re-emit any auxiliary top-level frames (e.g. Translation_Data)
    # captured from the source .x file. These sit between the skeleton
    # and the mesh in the original layout, so that's where we put them.
    if _passthrough_aux_frames and not binary_format:
        w(_passthrough_aux_frames)
        w("\n\n")

    all_warnings = []

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
                              unweld_on_export=unweld_on_export))
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
        for fr in sorted(all_sample_frames):
            scene.frame_set(fr)

            for pb in arm_obj.pose.bones:
                name = pb.name
                rot_set, sc_set, tr_set = per_bone_frames[name]
                if fr not in rot_set and fr not in sc_set and fr not in tr_set:
                    continue

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
                    # ROOT: write absolute rotation in un-conv'd file space,
                    # not just the pose offset. The importer reconstructs
                    # via local_rest_q = (conv.inv @ matrix_local).to_quat()
                    # and pose_q = local_rest_q.inv @ abs_q, so we must
                    # write abs_q = rotation of (conv.inv @ pb.matrix).
                    # Translation/scale come from the world matrix.
                    dx_world = _bl_bone_to_dx_world(world_bl, bl_to_dx_3, inv_scale)
                    dx_t = dx_world.to_translation()
                    dx_s = dx_world.to_scale()
                    un_conv_pose = bl_to_dx_3 @ pb.matrix.to_3x3()
                    q = un_conv_pose.to_quaternion()

                qw, qx, qy, qz = q.w, -q.x, -q.y, -q.z

                if fr in rot_set:
                    baked[name]["rot"]  [fr] = (qw, qx, qy, qz)
                if fr in sc_set:
                    baked[name]["scale"][fr] = (dx_s.x, dx_s.y, dx_s.z)
                if fr in tr_set:
                    baked[name]["pos"]  [fr] = (dx_t.x, dx_t.y, dx_t.z)

                # Matrix-key export: store the full DX-space local 4x4 for
                # this frame. The importer reads matrix keys as dx_local and
                # rebuilds the pose via rest^-1 @ parent_rest @ dx_local, so we
                # store the same parent-relative local transform the TRS path
                # encodes. For the root we store the DX-space world transform
                # (already computed for the TRS path), which the importer's
                # matrix path consumes for a parentless bone.
                if anim_key_format == "MATRIX":
                    if pb.parent:
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
        w(f"AnimationSet {_animset_name} {{\n")
        # some biped/game biped exports name the AnimationKey
        # nodes 'R', 'S', 'T' for rotation/scale/translation. Plain DirectX
        # leaves them unnamed.
        _ak_rot   = "AnimationKey R" if pz_compat else "AnimationKey"
        _ak_scale = "AnimationKey S" if pz_compat else "AnimationKey"
        _ak_trans = "AnimationKey T" if pz_compat else "AnimationKey"
        # Helper: scale a Blender frame number to a file tick value.
        # int(round(...)) keeps the file's tick field DWORD-compatible.
        def _tick(fr):
            return int(round(fr * tick_scale_out))
        for pb in arm_obj.pose.bones:
            name = pb.name
            rot_keys   = baked[name]["rot"]
            scale_keys = baked[name]["scale"]
            pos_keys   = baked[name]["pos"]
            mat_keys   = baked[name]["mat"]

            w(f"\tAnimation {{\n\t\t{{ {name} }}\n")

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

            w(f"\t\t{_ak_rot} {{\n\t\t\t0;\n\t\t\t{len(rot_keys)};\n")
            entries = [f"\t\t\t{_tick(fr)};4;{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f};;"
                       for fr, (qw, qx, qy, qz) in sorted(rot_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            w(f"\t\t{_ak_scale} {{\n\t\t\t1;\n\t\t\t{len(scale_keys)};\n")
            entries = [f"\t\t\t{_tick(fr)};3;{sx:.6f},{sy:.6f},{sz:.6f};;"
                       for fr, (sx, sy, sz) in sorted(scale_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            w(f"\t\t{_ak_trans} {{\n\t\t\t2;\n\t\t\t{len(pos_keys)};\n")
            entries = [f"\t\t\t{_tick(fr)};3;{px:.6f},{py:.6f},{pz:.6f};;"
                       for fr, (px, py, pz) in sorted(pos_keys.items())]
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


def _mszip_wrap_x(payload: bytes, base_format: str = "bin") -> bytes:
    """Wrap a raw .x payload in the MSZIP-compressed container format
    (xof 0303bzip / xof 0303tzip), splitting the payload into 32KB
    chunks each prefixed by uncompressed/compressed size headers."""
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
                      unweld_on_export=True):
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

    verts_dx = [(conv_inv_3 @ Vector(bl_verts[src_vi].co)) * inv_scale
                for src_vi in new_to_src]

    # Build a loop-aligned normal list for the MeshNormals writer below.
    # The downstream code expects one entry per loop in poly-iteration
    # order; since we no longer have a 1-to-1 loop→new_vert mapping, we
    # rebuild this view explicitly from raw_loop_normals.
    if raw_loop_normals is not None:
        loop_normals = list(raw_loop_normals)
    else:
        loop_normals = []

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

    if export_normals and loop_normals and not _use_stashed_decl:

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

    if export_uvs and me_work.uv_layers and not _use_stashed_decl:
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

    if export_materials:
        mats = _effective_materials(obj)
        if mats:
            w(f"\n{ind}\t\tMeshMaterialList {{\n{ind}\t\t\t{len(mats)};\n{ind}\t\t\t{n_faces};\n")
            w(",\n".join(f"{ind}\t\t\t{p.material_index}" for p in me_work.polygons) + ";\n")
            for mat in mats:

                w(f"{ind}\t\t\t{{{mat.name}}}\n")
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
