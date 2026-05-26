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
from .parser import XNode, _is_mesh_bone_name

_BT_NAME      = 0x0001
_BT_STRING    = 0x0002
_BT_INT_LIST  = 0x0006
_BT_FLT_LIST  = 0x0007
_BT_OBRACE    = 0x000a
_BT_CBRACE    = 0x000b
_BT_COMMA     = 0x0013
_BT_SEMICOLON = 0x0014

# Standard DirectX .x template declarations. Many DirectX consumers
# (including Project Zomboid's engine and 3DS Max's loader) expect these
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
    # When pz_compat is on, anim_fps is forced to 4800 (PZ convention)
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
        # in Blender) so that PZ-style engines that expect templates
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

        baked = {b.name: {"rot": {}, "scale": {}, "pos": {}} for b in arm_obj.pose.bones}

        # Collect per-bone keyframe times directly from the F-curves so
        # that a sparse imported animation stays sparse on export — the
        # original PZ file has e.g. 41 rotation keys but only 2 scale
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
        # Project Zomboid / 3DS Max biped exports name the AnimationKey
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

            w(f"\tAnimation {{\n\t\t{{ {name} }}\n")

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

        # Auxiliary-frame animations (e.g. Translation_Data on PZ
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

    # Round-trip preservation for PZ-style meshes: if this object was
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
    # match the original PZ / 3DS Max biped layout.
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
                # Include the group even if it has no influences: PZ
                # / 3DS Max biped exports declare a SkinWeights entry
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
        return [float(inv_col[c][r]) for r in range(4) for c in range(4)]
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


def _write_skin_weights(out: bytearray, skin: List[Tuple]) -> int:
    """Write skin weights for one bone.

    Entries can be 2-tuples (vi, weight) for single-mesh exports, or
    3-tuples (vi, weight, chunk_idx) for multi-mesh exports. Vi is
    CHUNK-LOCAL — indexes into the sub-mesh's vert array, not a
    combined one.

    Chunk encoding (the "lookback" rule, matches _read_skin_weights):
    --------------------------------------------------------------
    The trailer u16 at byte offset 8 of each 10-byte entry stores
    the chunk index of the NEXT entry, NOT the current one. The
    current entry's chunk is given by the PREVIOUS entry's trailer
    (or the bone's `pad` u16 header for entry [0]). This was
    verified against Journal.xcache where the trailer values
    transition at chunk boundaries.

    So when writing:
      * Header `pad` = first entry's chunk (entry [0] is in chunk
        `pad`).
      * Entry [N]'s trailer = entry [N+1]'s chunk.
      * The LAST entry's trailer has no successor — we write its
        own chunk index, which gives stable round-trip behaviour
        when re-imported.
    """
    n = len(skin)
    out += struct.pack('<I', n)

    # Extract per-entry (vi, weight, chunk) — defaulting chunk to 0
    # for 2-tuple legacy entries.
    parsed = []
    for entry in skin:
        if len(entry) == 3:
            parsed.append((int(entry[0]), float(entry[1]),
                           int(entry[2]) & 0xFFFF))
        else:
            parsed.append((int(entry[0]), float(entry[1]), 0))

    # Pad field = primary chunk = chunk of the FIRST entry (per the
    # lookback rule: entry [0]'s chunk is read from `pad`).
    primary_chunk = parsed[0][2] if parsed else 0
    out += struct.pack('<H', primary_chunk)
    n_written = 6

    for i, (vi, weight, chunk) in enumerate(parsed):
        # The trailer of entry i is the chunk of entry i+1 (or this
        # entry's own chunk if this is the last entry).
        if i + 1 < len(parsed):
            next_chunk = parsed[i + 1][2]
        else:
            next_chunk = chunk
        # 10 bytes per entry: u16 vi, u16 pad, f32 weight, u16 trailer
        out += struct.pack('<HHfH',
                           vi & 0xFFFF,
                           0,
                           weight,
                           next_chunk & 0xFFFF)
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
    """Write the 16-byte 'anim placeholder' (identity quaternion) that
    appears in non-animated bones. Returns 16."""
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
    """Return ((min_x, min_y, min_z), (max_x, max_y, max_z)) for a list of 3-tuples."""
    if not verts:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _write_mesh_common_header(out: bytearray, bbox_min, bbox_max) -> None:
    """Write the 144-byte common material header that appears immediately
    after the mesh FTM: a fixed prologue, the bbox (6 floats), a fixed
    epilogue, and the color tail."""
    out += _MESH_COMMON_HEADER_PROLOGUE
    out += struct.pack('<6f', *bbox_min, *bbox_max)
    out += _MESH_COMMON_HEADER_EPILOGUE
    out += _MESH_COMMON_HEADER_COLOR_TAIL


def _write_mesh_texture_entry(out: bytearray, path: str,
                               include_separator: bool) -> int:
    """Write one texture entry (presence flag + name_len + name + null) and
    optionally a 1-byte alignment separator. Returns the number of bytes
    written."""
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
    """Write a complete mesh block (parent_idx + header + FTM + material/flags
    block + verts/normals/uvs/faces) in the .xcache binary format."""
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
    """Write a complete bone block including header (parent_idx, name_len, name),
    FTM (64 bytes), the 296-byte pre-anim decoration, and animation/skin data
    if present."""
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
    """Build a complete .xcache file as bytes from a list of bone dicts and
    optional mesh dicts. Writes the SEMS header, all bone blocks (with
    animation/skin data), and any mesh blocks."""
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
    """Convert a mathutils.Matrix (4x4) into a list of 16 floats in DirectX
    row-major order (Blender's column-major matrix, transposed)."""
    # Transpose: DX row-major = Blender column-major when read sequentially
    return [float(mat[r][c]) for c in range(4) for r in range(4)]


def _build_xcache_parent_idx(bones_in_order, idx, parent_name):
    """Compute the relative parent_idx field for a bone at index `idx`
    in the topologically sorted bone list. Returns the xcache sentinel
    encoding: 2 for the first bone, 28 for the second, otherwise 1 if
    the parent is the previous bone, 0 if it's the root."""
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
    """Order Blender armature bones in depth-first parent-before-child order
    suitable for .xcache emission. Multiple roots are flattened with the
    first treated as bone[0] and the rest as its semantic children."""
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
    """Extract bone dicts (suitable for encode_xcache_bytes) from a Blender
    armature. Each dict carries name, parent_idx, FTM (parent-local),
    bind_pose (world), skin_offset, and optional animation channels."""

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
    """Populate the 'skin' field of each bone dict from the mesh dicts'
    Blender vertex groups.

    For single-mesh exports: each bone gets a flat list of (vi, weight)
    pairs, with vi indexing directly into the mesh's vert array.

    For multi-mesh exports (multiple mesh_dicts representing sub-meshes
    of one source xcache): each bone gets a list of (vi, weight,
    chunk_idx) triples. Vi values are CHUNK-LOCAL, not offset into a
    combined array; the chunk_idx tells the reader which sub-mesh the
    vi indexes into. This matches the original Bugsnax xcache format,
    so multi-mesh files round-trip with their sub-mesh structure
    intact.
    """
    name_to_idx = {bd['name']: i for i, bd in enumerate(bone_dicts)}
    is_multi_mesh = len(mesh_dicts) > 1

    for chunk_idx, mesh_dict in enumerate(mesh_dicts):
        mesh_obj = mesh_dict.get('_blender_obj')
        bl_to_export = mesh_dict.get('_bl_to_export')
        if mesh_obj is None or bl_to_export is None:
            # Mesh dict wasn't produced by our blender-side helper — skip
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
            continue

        me = mesh_obj.data
        # Map vertex_group index → bone name
        vg_to_bone = {}
        for vg in mesh_obj.vertex_groups:
            if vg.name in name_to_idx:
                vg_to_bone[vg.index] = vg.name

        # Iterate Blender source vertices. Vi values are CHUNK-LOCAL —
        # they index into THIS mesh's vert array, not a combined one.
        # When multi-mesh, we tag each entry with chunk_idx so the
        # writer can set the trailer u16 to identify which sub-mesh
        # the entry belongs to.
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
                    if is_multi_mesh:
                        bone_skin.append((export_vi, w, chunk_idx))
                    else:
                        bone_skin.append((export_vi, w))


def collect_meshes_from_blender(mesh_objs, arm_obj, bl_to_dx_3, inv_scale,
                                  depsgraph=None, use_modifiers=True):
    """Walk Blender mesh objects and produce mesh dicts ready for
    encode_xcache_bytes (name, ftm, verts, normals, uvs, faces). Mesh
    data uses the raw bind-pose vertices (not depsgraph-evaluated, so
    armature deformation doesn't bake into the export)."""
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

        def _normalize_tex_path(p: str) -> str:
            """Strip Blender's relative-path prefixes and backslash-encode.
            Examples:
              '//BefficaBody_D.dds' -> 'BefficaBody_D.dds'
              '.\\BefficaBody_D.dds' -> 'BefficaBody_D.dds'
              './foo/bar.dds' -> 'foo/bar.dds'
              'Content/Models/.../X.dds' -> 'Content/Models/.../X.dds'
            Engine-style 'Content/...' paths are kept as-is.
            """
            if not p:
                return p
            p = p.replace('\\', '/')
            while p.startswith('./') or p.startswith('//'):
                p = p[2:]
            return p

        for mat in _effective_materials(obj):
            if mat is None:
                continue
            stashed = mat.get('_x_texture_filename')
            if isinstance(stashed, str) and stashed:
                stashed = _normalize_tex_path(stashed)
                if stashed and stashed not in seen_paths:
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
                    if picked:
                        picked = _normalize_tex_path(picked)
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
    """Top-level entry point for exporting the current Blender scene state to
    a .xcache binary file. Collects bones (with animation), meshes, and
    skin weights, then writes the SEMS-format binary container."""
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

    # Sort mesh objects by _x_submesh_idx if available, so split-mesh
    # imports round-trip with sub-meshes in their original order. Objects
    # without the property (e.g. user-created meshes) sort to the end,
    # preserving their relative scene order.
    def _submesh_key(o):
        idx = o.get('_x_submesh_idx')
        if idx is None:
            return (1, 0, o.name)  # bucket 1: no stash, after stashed
        return (0, int(idx), o.name)
    mesh_objs.sort(key=_submesh_key)

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

    export_xcache_to_file(filepath, bone_dicts, anim_frame_count,
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