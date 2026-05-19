"""
DirectX .x  ──►  Blender  importer

Key design decisions
--------------------
BONE REST POSE comes from inv(SkinWeights offset matrix), NOT from the
FrameTransformMatrix hierarchy.  The FTM hierarchy in Burger.x encodes an
animated pose (frame 1 of the walk cycle), not the bind pose.  The offset
matrix is defined as: offset = inv(bone_world_bind_pose), so
bone_world_bind_pose = inv(offset).  Using these matrices for edit_bone.matrix
gives correct per-bone local axes and correct Blender skinning.

Bones that have NO SkinWeights entry (i.e. never influence any vertex) still
need to exist in the armature for the animation tracks to work.  For those
bones we fall back to the FTM-derived global matrix.

Mesh-container frames (frames whose only child is a Mesh, with no sub-Frames)
are NOT added to the armature as bones.
"""

import os
import math
import bpy
from mathutils import Matrix, Vector, Quaternion

from .parser import parse_x_file

_KEY_TYPE_VALUES = {
    0: 4,
    1: 3,
    2: 3,
    4: 16,
}

def _fill_unweighted_from_neighbours(obj):
    """For each vertex with no skin weights, copy the bone assignment of
    the geometrically closest weighted vertex. Used to repair meshes
    where a SkinWeights block went missing in the source file (e.g.
    BurgerB's right-tusk tip, where Burger_r_Tusk_01_05SHJnt is declared
    as a bone but has no SkinWeights entry, leaving 12 mirrored verts
    floating in rest pose while the rest of the tusk animates).
    """
    me = obj.data
    n_verts = len(me.vertices)
    if n_verts == 0 or not obj.vertex_groups:
        return

    weighted = []
    unweighted = []
    for v in me.vertices:
        if v.groups:
            weighted.append(v.index)
        else:
            unweighted.append(v.index)
    if not unweighted or not weighted:
        return

    # Look up coords once; vertices.foreach_get is overkill for this size.
    coords = [me.vertices[i].co for i in range(n_verts)]

    for u_vi in unweighted:
        up = coords[u_vi]
        # Closest weighted vert by squared distance.
        best_vi = weighted[0]
        best_d2 = (up - coords[best_vi]).length_squared
        for w_vi in weighted:
            d2 = (up - coords[w_vi]).length_squared
            if d2 < best_d2:
                best_d2 = d2
                best_vi = w_vi
        # Copy each (group, weight) pair from the donor onto the orphan.
        for g in me.vertices[best_vi].groups:
            vg = obj.vertex_groups[g.group]
            vg.add([u_vi], g.weight, "REPLACE")


def _mat4_from_list(vals):
    if len(vals) < 16:
        return Matrix.Identity(4)
    return Matrix([
        [vals[0],  vals[1],  vals[2],  vals[3]],
        [vals[4],  vals[5],  vals[6],  vals[7]],
        [vals[8],  vals[9],  vals[10], vals[11]],
        [vals[12], vals[13], vals[14], vals[15]],
    ]).transposed()

class _SyntheticKeyNode:
    """Minimal object mimicking an XNode AnimationKey for synthetic tracks."""
    __slots__ = ("_nums",)

    def __init__(self, nums_list):
        self._nums = nums_list

    def nums(self):
        return self._nums

def _compose_type4_from_trs(rot_node, scale_node, trans_node):
    """Compose per-channel TRS animation keys into a single type-4 (matrix)
    key sequence.

    This is an optimization for files where rotation, scale, and translation
    tracks share identical tick sequences (e.g. Bugsnax xcache exports).
    For files where the tracks have different key densities — e.g. Project
    Zomboid, where scale is often just 2 keys while rotation has 21 — the
    composition cannot be done by simple index-pairing without distorting
    the animation. In that case we return None and let the caller fall back
    to per-track processing, which handles arbitrary key densities correctly
    via Blender's separate F-curve channels.
    """
    def _parse(node, expected):
        nums = node.nums()
        if len(nums) < 2:
            return None
        count = int(nums[1])
        out = []
        i = 2
        while i < len(nums) and len(out) < count:
            if i + 1 >= len(nums):
                break
            tick = nums[i]; i += 1
            i += 1
            if i + expected > len(nums):
                break
            out.append((tick, nums[i:i + expected]))
            i += expected
        return out

    rot_frames   = _parse(rot_node,   4)
    scale_frames = _parse(scale_node, 3)
    trans_frames = _parse(trans_node, 3)
    if not rot_frames or not scale_frames or not trans_frames:
        return None

    # Bail out unless all three tracks have the SAME tick sequence.  This
    # is the only case where simple index-pairing produces a correct
    # composition. Track-count mismatch (or matching counts with diverging
    # ticks) means the file stores TRS at different sample densities and
    # the per-track path must handle them independently.
    if len(rot_frames) != len(scale_frames) or len(rot_frames) != len(trans_frames):
        return None
    for (rt, _), (st, _), (tt, _) in zip(rot_frames, scale_frames, trans_frames):
        if rt != st or rt != tt:
            return None

    n = len(rot_frames)
    if n == 0:
        return None

    out_nums = [4.0, float(n)]
    for i in range(n):
        tick, rot_vals   = rot_frames[i]
        _,    scale_vals = scale_frames[i]
        _,    trans_vals = trans_frames[i]

        w, x, y, z = rot_vals[:4]
        sx, sy, sz = scale_vals[:3]
        tx, ty, tz = trans_vals[:3]

        r00 = 1.0 - 2.0*(y*y + z*z);  r01 =       2.0*(x*y - z*w);  r02 =       2.0*(x*z + y*w)
        r10 =       2.0*(x*y + z*w);  r11 = 1.0 - 2.0*(x*x + z*z);  r12 =       2.0*(y*z - x*w)
        r20 =       2.0*(x*z - y*w);  r21 =       2.0*(y*z + x*w);  r22 = 1.0 - 2.0*(x*x + y*y)

        r00 *= sx; r01 *= sx; r02 *= sx
        r10 *= sy; r11 *= sy; r12 *= sy
        r20 *= sz; r21 *= sz; r22 *= sz

        out_nums.extend([
            tick, 16.0,
            r00, r01, r02, 0.0,
            r10, r11, r12, 0.0,
            r20, r21, r22, 0.0,
            tx,  ty,  tz,  1.0,
        ])

    return _SyntheticKeyNode(out_nums)

def _axis_matrix(axis_forward, axis_up):
    _AXES = {'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0),'Z':(0,0,1),'-Z':(0,0,-1)}
    import numpy as np
    fwd = np.array(_AXES[axis_forward], float)
    upv = np.array(_AXES[axis_up],      float)
    rgt = np.cross(fwd, upv)
    F   = np.column_stack([rgt, upv, fwd])
    B   = np.column_stack([(1,0,0),(0,0,1),(0,1,0)])
    M3  = (B @ np.linalg.inv(F)).astype(float)
    return Matrix([[float(M3[r,c]) for c in range(3)] + [0] for r in range(3)] + [[0,0,0,1]])

def _collect_offset_matrices(root_node):
    bind_poses = {}

    def walk(node):
        if node.kind == "SkinWeights":
            bone_name = next((v for t, v in node.values if t == "STR"), None)
            if bone_name is None:
                bone_name = next((v for t, v in node.values if t == "WORD"), None)
            nums = node.nums()
            if bone_name and nums:
                n = int(nums[0])
                offset_vals = nums[1 + n + n : 1 + n + n + 16]
                if len(offset_vals) == 16:
                    offset_mat = _mat4_from_list(offset_vals)
                    try:
                        bind_poses[bone_name] = offset_mat.inverted()
                    except ValueError:
                        bind_poses[bone_name] = Matrix.Identity(4)
        for child in node.children:
            walk(child)

    walk(root_node)
    return bind_poses

def _collect_ftm_globals(frame_node, parent_mat=None):
    if parent_mat is None:
        parent_mat = Matrix.Identity(4)
    ftm       = frame_node.child("FrameTransformMatrix")
    local_mat = _mat4_from_list(ftm.nums()) if ftm else Matrix.Identity(4)
    global_mat = parent_mat @ local_mat
    result = {frame_node.name: global_mat}
    for child in frame_node.children:
        if child.kind == "Frame":
            result.update(_collect_ftm_globals(child, global_mat))
    return result

def _compute_animation_frame_range(root):
    min_f = None
    max_f = None
    for anim_set in [n for n in root.children if n.kind == "AnimationSet"]:
        for anim_node in anim_set.children_of("Animation"):
            for key_node in anim_node.children_of("AnimationKey"):
                nums = key_node.nums()
                if len(nums) < 2:
                    continue
                key_type  = int(nums[0])
                key_count = int(nums[1])
                expected  = _KEY_TYPE_VALUES.get(key_type)
                if expected is None:
                    continue

                i = 2
                for _ in range(key_count):
                    if i + 1 >= len(nums):
                        break
                    tick = nums[i]
                    i += 2
                    if i + expected > len(nums):
                        break
                    i += expected
                    if min_f is None or tick < min_f:
                        min_f = tick
                    if max_f is None or tick > max_f:
                        max_f = tick
    if min_f is None:
        return None
    return int(round(min_f)), int(round(max_f))

def import_x(context, filepath,
             use_apply_transform=True,
             global_scale=1.0,
             axis_forward="-Z",
             axis_up="Y",
             import_normals=True,
             import_uvs=True,
             import_materials=True,
             import_textures=True,
             import_armature=True,
             import_weights=True,
             import_animation=True,
             anim_fps=0.0,
             set_frame_range=True,
             rest_pose_source='FRAME_TRANSFORM',
             infer_sharps=True,
             sharp_angle_deg=75.0,
             lock_root_translation=False,
             lock_leaf_translation=False,
             smooth_shade_from_faces=False,
             **_):

    root     = parse_x_file(filepath)
    base_dir = os.path.dirname(filepath)

    # ----- Passthrough text capture -----
    # The exporter doesn't natively re-emit template declarations or
    # "auxiliary" frames like PZ's Translation_Data (which exists at the
    # top level of the file but isn't part of the skeleton or mesh
    # hierarchy). To make round-trip lossless, capture those blocks as
    # raw source text now and stash them so the exporter can spit them
    # back out verbatim. This only works for the text .x format; binary
    # .x and .xcache have no template equivalent.
    passthrough_templates = []
    passthrough_frames = {}
    passthrough_decldata = {}        # mesh_name -> raw DeclData block text
    passthrough_xskinheader = {}     # mesh_name -> raw XSkinMeshHeader block text
    passthrough_animations = {}      # target_name -> raw Animation block text
    try:
        with open(filepath, 'rb') as _fh:
            _raw = _fh.read(64)
        if _raw[:16] == b"xof 0303txt 0032":
            with open(filepath, 'r', encoding='latin-1', errors='replace') as _fh:
                _full_text = _fh.read()

            def _extract_braced_block(text, start_idx):
                # Given index pointing at a `{`, walk the matching brace.
                depth = 0
                j = start_idx
                while j < len(text):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            return j + 1
                    j += 1
                return j

            # All `template Foo { ... }` declarations
            import re as _re
            for _m in _re.finditer(r'\btemplate\s+\w+\s*\{', _full_text):
                _start = _m.start()
                _open_brace = _full_text.index('{', _start)
                _end = _extract_braced_block(_full_text, _open_brace)
                passthrough_templates.append(_full_text[_start:_end].rstrip())

            # Any top-level Frame in the source file
            top_frame_names = {c.name for c in root.children if c.kind == "Frame"}
            for _m in _re.finditer(r'(?m)^\s*Frame\s+(\w+)\s*\{', _full_text):
                _name = _m.group(1)
                if _name not in top_frame_names:
                    continue
                _start = _full_text.find('Frame', _m.start())
                _open_brace = _full_text.index('{', _start)
                _end = _extract_braced_block(_full_text, _open_brace)
                passthrough_frames[_name] = _full_text[_start:_end].rstrip()

            # Per-mesh DeclData and XSkinMeshHeader
            for _m in _re.finditer(r'(?m)^\s*Mesh\s+(\w+)\s*\{', _full_text):
                _mesh_name = _m.group(1)
                _mesh_start = _full_text.find('Mesh', _m.start())
                _mesh_open  = _full_text.index('{', _mesh_start)
                _mesh_end   = _extract_braced_block(_full_text, _mesh_open)
                _mesh_text  = _full_text[_mesh_start:_mesh_end]
                for _kind, _stash in [
                    ('DeclData', passthrough_decldata),
                    ('XSkinMeshHeader', passthrough_xskinheader),
                ]:
                    _km = _re.search(r'\b' + _kind + r'\s*\{', _mesh_text)
                    if _km is None:
                        continue
                    _ks = _km.start()
                    _ko = _mesh_text.index('{', _ks)
                    _ke = _extract_braced_block(_mesh_text, _ko)
                    _stash[_mesh_name] = _mesh_text[_ks:_ke].rstrip()

            # Per-target Animation blocks (so non-bone frames like
            # Translation_Data still get a track on round-trip)
            for _m in _re.finditer(r'AnimationSet\s+\w+\s*\{', _full_text):
                _line_start = _full_text.rfind('\n', 0, _m.start()) + 1
                if 'template' in _full_text[_line_start:_m.start()]:
                    continue
                _as_start = _m.start()
                _as_open  = _full_text.index('{', _as_start)
                _as_end   = _extract_braced_block(_full_text, _as_open)
                _as_text  = _full_text[_as_start:_as_end]
                for _am in _re.finditer(r'Animation\s*\{', _as_text):
                    _aline_start = _as_text.rfind('\n', 0, _am.start()) + 1
                    if 'template' in _as_text[_aline_start:_am.start()]:
                        continue
                    _a_start = _am.start()
                    _a_open  = _as_text.index('{', _a_start)
                    _a_end   = _extract_braced_block(_as_text, _a_open)
                    _a_text  = _as_text[_a_start:_a_end]
                    _ref = _re.search(r'\{\s*(\w+)\s*\}', _a_text)
                    if _ref:
                        passthrough_animations[_ref.group(1)] = _a_text.rstrip()
                break
    except Exception:
        passthrough_templates = []
        passthrough_frames = {}
        passthrough_decldata = {}
        passthrough_xskinheader = {}
        passthrough_animations = {}

    state = _ImportState(
        base_dir=base_dir,
        global_scale=global_scale,
        import_normals=import_normals,
        import_uvs=import_uvs,
        import_materials=import_materials,
        import_textures=import_textures,
        import_armature=import_armature,
        import_weights=import_weights,
        import_animation=import_animation,
        anim_fps=anim_fps,
        rest_pose_source=rest_pose_source,
        infer_sharps=infer_sharps,
        sharp_angle_deg=sharp_angle_deg,
        lock_root_translation=lock_root_translation,
        lock_leaf_translation=lock_leaf_translation,
        smooth_shade_from_faces=smooth_shade_from_faces,
    )
    state._passthrough_decldata = passthrough_decldata
    state._passthrough_xskinheader = passthrough_xskinheader

    file_ticks_per_second = None
    for node in root.children:
        if node.kind == "AnimTicksPerSecond":
            nums = node.nums()
            if nums:
                file_ticks_per_second = nums[0]
                state.ticks_per_second = nums[0]

    if anim_fps and anim_fps > 0:
        # User-specified FPS: use that as the target Blender scene FPS,
        # and scale tick values from the file's resolution to match.
        target_fps = anim_fps
        if file_ticks_per_second and file_ticks_per_second > 0:
            state.tick_scale = float(target_fps) / float(file_ticks_per_second)
        state.ticks_per_second = target_fps
    elif file_ticks_per_second is not None and file_ticks_per_second > 240:
        # High-precision tick rate (e.g. Project Zomboid uses 4800) —
        # Blender's scene FPS field is integer with effective max around
        # 240 in normal use, and using 4800 as the frame-number unit makes
        # the timeline confusing (animations that are <1 second land at
        # frames 0..3200 instead of 0..20).
        # Normalize to 30 fps: scale tick values down so animation length
        # in frames matches duration-in-seconds × 30.
        target_fps = 30.0
        state.tick_scale = target_fps / float(file_ticks_per_second)
        state.ticks_per_second = target_fps

    mat_nodes = [n for n in root.children if n.kind == "Material"]
    for node in mat_nodes:
        state.parse_material(node, base_dir)

    frame_nodes = [n for n in root.children if n.kind == "Frame"]

    if import_armature and frame_nodes:
        bind_poses = _collect_offset_matrices(root)
        ftm_globals = {}
        for fn in frame_nodes:
            ftm_globals.update(_collect_ftm_globals(fn))

        conv_mat = _axis_matrix(axis_forward, axis_up)
        axis_fix = Matrix.Rotation(math.pi, 4, 'Z')

        conv_mat = axis_fix @ conv_mat

        _cm3 = conv_mat.to_3x3()

        state.build_armature(frame_nodes, context, bind_poses, ftm_globals, conv_mat)

        # Stash passthrough text on the armature so the exporter can
        # re-emit templates and auxiliary frames verbatim on round-trip.
        if state.armature_obj is not None:
            arm_data = state.armature_obj.data
            if passthrough_templates:
                arm_data["_x_templates"] = "\n\n".join(passthrough_templates)

            def _frame_owns_skeleton_or_mesh(node):
                if node.name in state._skel_root_names:
                    return True
                for ch in node.children:
                    if ch.kind == "Mesh":
                        return True
                    if ch.kind == "Frame" and _frame_owns_skeleton_or_mesh(ch):
                        return True
                return False

            aux_blocks = []
            aux_anim_blocks = []
            for fn in frame_nodes:
                if _frame_owns_skeleton_or_mesh(fn):
                    continue
                txt = passthrough_frames.get(fn.name)
                if txt:
                    aux_blocks.append(txt)
                anim_txt = passthrough_animations.get(fn.name)
                if anim_txt:
                    aux_anim_blocks.append(anim_txt)
            if aux_blocks:
                arm_data["_x_aux_frames"] = "\n\n".join(aux_blocks)
            if aux_anim_blocks:
                arm_data["_x_aux_animations"] = "\n\n".join(aux_anim_blocks)

    for anim_set in [n for n in root.children if n.kind == "AnimationSet"]:
        for anim_node in anim_set.children_of("Animation"):
            ref = anim_node.child("REF")
            if not ref: continue
            bname = next((v for t,v in ref.values if t=="WORD"), None)
            if not bname: continue
            for key_node in anim_node.children_of("AnimationKey"):
                knums = key_node.nums()
                if len(knums) < 2 or int(knums[0]) != 1: continue
                scale_vals = []
                i = 2
                while i < len(knums):
                    if i + 1 >= len(knums): break
                    i += 1; i += 1
                    if i + 3 > len(knums): break
                    scale_vals.append(knums[i]); i += 3
                if scale_vals and all(abs(v) < 0.01 for v in scale_vals):
                    state._always_hidden_bones.add(bname)
    if state._always_hidden_bones:
        pass

    for node in root.children:
        if node.kind == "Frame":
            state.import_frame_meshes(node, context)

    if import_animation:
        anim_sets = [n for n in root.children if n.kind == "AnimationSet"]
        for node in anim_sets:
            state.import_animation_set(node, context)

    if import_animation and set_frame_range:
        frame_range = _compute_animation_frame_range(root)
        if frame_range:
            fstart, fend = frame_range
            # Apply the same tick → frame scale used for animation keys, so
            # the scene frame range matches where the keys actually land.
            if state.tick_scale != 1.0:
                fstart = int(round(fstart * state.tick_scale))
                fend   = int(round(fend   * state.tick_scale))

            if fend < fstart:
                fend = fstart
            context.scene.frame_start = fstart
            context.scene.frame_end   = fend
            context.scene.frame_set(fstart)
        else:
            pass

    return {"FINISHED"}

def _iter_action_fcurves(action, anim_data=None):
    if hasattr(action, "fcurves"):

        yield from action.fcurves
        return

    slot = None
    if anim_data is not None and hasattr(anim_data, "action_slot"):
        slot = anim_data.action_slot
    for layer in action.layers:
        for strip in layer.strips:
            if slot is not None:
                cb = strip.channelbag(slot)
                if cb is not None:
                    yield from cb.fcurves
            else:

                for cb in strip.channelbags:
                    yield from cb.fcurves

def _get_or_create_fcurve(action, anim_data, data_path, index, group_name):
    if hasattr(action, "fcurves"):
        fc = action.fcurves.find(data_path, index=index)
        if fc is None:
            fc = action.fcurves.new(data_path, index=index, action_group=group_name)
        return fc

    slot = getattr(anim_data, "action_slot", None) if anim_data is not None else None
    if not action.layers:
        action.layers.new("Layer")
    layer = action.layers[0]
    if not layer.strips:
        layer.strips.new(type="KEYFRAME")
    strip = layer.strips[0]
    cb = strip.channelbag(slot, ensure=True) if slot is not None else (
        strip.channelbags[0] if strip.channelbags else strip.channelbags.new()
    )
    fc = cb.fcurves.find(data_path, index=index)
    if fc is None:
        fc = cb.fcurves.new(data_path, index=index, group_name=group_name)
    return fc

def _decode_decl_data(decl_node, expected_vert_count):
    """Decode a DeclData node into per-vertex normals and UVs.

    DirectX's DeclData packs an arbitrary list of per-vertex elements
    (positions, normals, tangents, UVs, etc.) into a single DWORD
    stream, with a small element table at the start describing the
    layout. PZ / 3DS Max biped exports use this instead of separate
    MeshNormals and MeshTextureCoords blocks.

    Layout per the DirectX template:
        DWORD nElements;
        VertexElement Elements[nElements];   # 4 DWORDs each (Type, Method, Usage, UsageIndex)
        DWORD nDWords;
        DWORD data[nDWords];                 # raw stream, reinterpret as floats per the type table

    Element Type codes used here:
        1 = D3DDECLTYPE_FLOAT2 (2 floats)
        2 = D3DDECLTYPE_FLOAT3 (3 floats)
        3 = D3DDECLTYPE_FLOAT4 (4 floats)

    Element Usage codes used here:
        0 = POSITION   3 = NORMAL   5 = TEXCOORD   6 = TANGENT

    Returns (per_vert_normals, per_vert_uvs) where each is a list of
    tuples or None if not present in the layout. Returns None overall
    if the data is malformed.
    """
    import struct
    nums = decl_node.nums()
    if not nums:
        return None
    try:
        n_elements = int(nums[0])
        if n_elements <= 0 or n_elements > 32:
            return None
        # Element table: 4 ints per element
        if 1 + n_elements * 4 + 1 > len(nums):
            return None
        elements = []
        i = 1
        for _ in range(n_elements):
            t  = int(nums[i]);     i += 1
            m  = int(nums[i]);     i += 1
            u  = int(nums[i]);     i += 1
            ui = int(nums[i]);     i += 1
            elements.append((t, m, u, ui))
        n_dwords = int(nums[i]); i += 1
        if i + n_dwords > len(nums):
            return None
        # Reinterpret each DWORD's 32 bits as a float.
        floats = []
        for j in range(n_dwords):
            raw = int(nums[i + j]) & 0xFFFFFFFF
            floats.append(struct.unpack('<f', struct.pack('<I', raw))[0])

        # Compute per-vertex stride in floats and locate normal / UV
        # offsets within each vertex.
        type_floats = {1: 2, 2: 3, 3: 4}
        stride = 0
        norm_off = None
        uv_off   = None
        for t, _m, u, _ui in elements:
            n_f = type_floats.get(t)
            if n_f is None:
                # Unknown / unsupported type — bail
                return None
            if u == 3 and norm_off is None and n_f == 3:
                norm_off = (stride, 3)
            elif u == 5 and uv_off is None and n_f >= 2:
                uv_off = (stride, 2)
            stride += n_f
        if stride == 0:
            return None
        if n_dwords % stride != 0:
            return None
        n_verts = n_dwords // stride
        if expected_vert_count and n_verts != expected_vert_count:
            # Layout doesn't match the mesh vertex count — refuse to
            # apply rather than risk corrupting unrelated data.
            return None

        per_vert_normals = None
        if norm_off is not None:
            off, count = norm_off
            per_vert_normals = []
            for v in range(n_verts):
                base = v * stride + off
                per_vert_normals.append((floats[base], floats[base + 1], floats[base + 2]))
        per_vert_uvs = None
        if uv_off is not None:
            off, _ = uv_off
            per_vert_uvs = []
            for v in range(n_verts):
                base = v * stride + off
                per_vert_uvs.append((floats[base], floats[base + 1]))
        return (per_vert_normals, per_vert_uvs)
    except Exception:
        return None


def _apply_custom_normals(me, loop_normals):
    if hasattr(me, "normals_split_custom_set"):

        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True
        me.normals_split_custom_set(loop_normals)
    else:

        attr = me.attributes.get("custom_normal")
        if attr is None:
            attr = me.attributes.new("custom_normal", "FLOAT_VECTOR", "CORNER")
        for i, n in enumerate(loop_normals):
            attr.data[i].vector = n


def _clear_custom_split_normals(me):
    """Remove any custom per-loop normals on the mesh and disable
    auto-smooth, so Blender renders using face-averaged normals (the
    standard "shade smooth" appearance)."""
    try:
        # Pre-4.x: there's an explicit "free custom split normals" call,
        # plus a use_auto_smooth toggle. Disabling auto-smooth makes the
        # poly use_smooth flag drive the result directly.
        if hasattr(me, "free_normals_split"):
            me.free_normals_split()
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = False
    except Exception:
        pass
    try:
        # 4.1+: custom normals live in a CORNER FLOAT_VECTOR attribute
        # (or sometimes the mesh attribute "custom_normal"). Drop them
        # so geometry-derived normals win.
        attr = me.attributes.get("custom_normal")
        if attr is not None:
            me.attributes.remove(attr)
    except Exception:
        pass
    try:
        # Belt-and-braces: apply identity-ish normals via the split-normals
        # API to nuke any cached overrides Blender may have already
        # baked. Using me.calc_normals() / me.update() forces a recompute.
        if hasattr(me, "calc_normals"):
            me.calc_normals()
        me.update()
    except Exception:
        pass

class _ImportState:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.materials:          dict  = {}
        self.armature_obj:       object = None
        self.ticks_per_second:   float  = 30.0
        self.tick_scale:         float  = 1.0   # multiplier from file ticks → Blender frames
        self.created_objects:    list   = []
        self._conv_mat           = Matrix.Identity(4)
        self._always_hidden_bones: set = set()
        self._skel_root_names:   set    = set()
        self._bone_rebind:       dict   = {}
        # Per-mesh source-text passthroughs (filled in by import_x);
        # default to empty so non-import code paths don't crash.
        self._passthrough_decldata: dict = {}
        self._passthrough_xskinheader: dict = {}

    def parse_material(self, node, base_dir):
        name = node.name or "Material"
        existing = bpy.data.materials.get(name)
        if existing:
            self.materials[name] = existing
            return existing
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True

        bsdf = next(
            (n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"),
            None,
        )
        out_node = next(
            (n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"),
            None,
        )
        if bsdf is None:
            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        if out_node is None:
            out_node = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")

        if not any(
            lnk.from_node == bsdf and lnk.to_node == out_node
            for lnk in mat.node_tree.links
        ):
            mat.node_tree.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

        nums = node.nums()
        if len(nums) >= 4:
            r, g, b, a = nums[0], nums[1], nums[2], nums[3]
            bsdf.inputs["Base Color"].default_value = (r, g, b, a)

            mat["_x_face_color"] = (r, g, b, a)
        if len(nums) >= 5:
            shininess = nums[4]
            # The .x format pre-dates PBR.  Most exporters (Blender's old
            raw_roughness = max(0.0, min(1.0,
                1.0 - math.log(max(shininess, 1e-6)) / math.log(128.0)))
            ROUGHNESS_FLOOR = 0.5
            roughness = ROUGHNESS_FLOOR + raw_roughness * (1.0 - ROUGHNESS_FLOOR)
            bsdf.inputs["Roughness"].default_value = roughness
            mat["_x_power"] = shininess
        if len(nums) >= 8:
            sr, sg, sb = nums[5], nums[6], nums[7]
            spec_val   = (sr + sg + sb) / 3.0
            # Same rationale as above — the raw spec value in .x files is
            spec_val = min(spec_val, 0.2)
            spec_input = (bsdf.inputs.get("Specular IOR Level")
                          or bsdf.inputs.get("Specular"))
            if spec_input:
                spec_input.default_value = spec_val
            else:
                pass
            mat["_x_specular"] = (sr, sg, sb)
        if len(nums) >= 11:
            mat["_x_emissive"] = (nums[8], nums[9], nums[10])

        # Stash the texture filename on the material *unconditionally*
        tex_node = node.child("TextureFileName") or node.child("TextureFilename")
        original_tex_name = None
        if tex_node:
            strs = tex_node.strings()
            if strs:
                original_tex_name = strs[0].replace('\\\\', '\\')
                mat["_x_texture_filename"] = original_tex_name

        if self.import_textures and original_tex_name:
            tex_path  = original_tex_name.replace("\\", os.sep).replace("/", os.sep)
            full_path = os.path.join(base_dir, tex_path)

            # Build a list of plausible directories to look in.  xcache
            tex_basename = os.path.basename(tex_path)
            tex_subparts = tex_path.split(os.sep)

            search_paths = []

            # 1. Exact path from xcache, anchored at base_dir
            search_paths.append(full_path)

            # 2. Progressively shorter tails of the engine path under
            #    base_dir (e.g. "Apple/Apple_D.dds", "Apple_D.dds")
            for cut in range(1, len(tex_subparts)):
                search_paths.append(os.path.join(base_dir, *tex_subparts[cut:]))

            # 3. Common texture subfolders next to the xcache
            for sub in ("Textures", "textures", "Tex", "tex"):
                search_paths.append(os.path.join(base_dir, sub, tex_basename))

            # 4. Walk up to 3 directory levels above base_dir, also with
            ancestor = base_dir
            for _ in range(3):
                parent = os.path.dirname(ancestor)
                if not parent or parent == ancestor:
                    break
                ancestor = parent
                search_paths.append(os.path.join(ancestor, tex_path))
                search_paths.append(os.path.join(ancestor, tex_basename))

            # For each candidate location, try the requested extension
            alt_exts = [".png", ".jpg", ".jpeg", ".tga", ".dds", ".bmp", ".tif", ".tiff", ".webp"]

            def _expand_candidates(p):
                yield p
                stem, _ext = os.path.splitext(p)
                for ext in alt_exts:
                    yield stem + ext

            img = None
            tried = set()
            for sp in search_paths:
                for candidate in _expand_candidates(sp):
                    if candidate in tried:
                        continue
                    tried.add(candidate)
                    if os.path.exists(candidate):
                        try:
                            img = bpy.data.images.load(candidate, check_existing=True)
                        except Exception:
                            img = None
                        if img is not None:
                            break
                if img is not None:
                    break

            # Always create a TEX_IMAGE node and connect it to the
            if img is None:
                try:
                    placeholder_name = os.path.basename(original_tex_name) or name
                    img = bpy.data.images.get(placeholder_name)
                    if img is None:
                        img = bpy.data.images.new(
                            placeholder_name, width=1, height=1, alpha=False,
                        )
                    # Point the placeholder at the path we wanted —
                    try:
                        img.filepath = full_path
                        img.source = 'FILE'
                    except Exception:
                        pass
                except Exception:
                    img = None

            if img is not None:
                tex_img_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
                tex_img_node.image = img
                # Stash the original (engine-format) path on the
                tex_img_node["_x_texture_filename"] = original_tex_name
                try:
                    mat.node_tree.links.new(
                        tex_img_node.outputs["Color"],
                        bsdf.inputs["Base Color"],
                    )
                except Exception:
                    pass

        self.materials[name] = mat
        return mat

    def build_armature(self, frame_nodes, context, bind_poses, ftm_globals, conv_mat):
        self._conv_mat = conv_mat

        use_ftm_rest = (self.rest_pose_source == 'FRAME_TRANSFORM')
        self._bone_rebind = {}

        # A Frame is a skeleton root if it isn't carrying a Mesh. The
        # previous test ("has at least one Frame child") accidentally
        # excluded leaf bones — fine when bones formed a deep chain
        # (each bone had the next as a Frame child), broken for flat
        # hierarchies where every bone sits at top level with no Frame
        # children of its own. In Bugsnax xcaches the corrected parser
        # now emits the spine bones as top-level siblings, which under
        # the old rule were all silently dropped.
        skel_roots = [f for f in frame_nodes
                      if not any(c.kind == "Mesh" for c in f.children)]

        self._skel_root_names = {f.name for f in skel_roots}
        skipped = [f.name for f in frame_nodes if f not in skel_roots]
        if skipped:
            pass

        arm_data = bpy.data.armatures.new("Armature")
        arm_data.display_type = "STICK"
        arm_obj  = bpy.data.objects.new("Armature", arm_data)
        context.collection.objects.link(arm_obj)
        self.armature_obj = arm_obj
        self.created_objects.append(arm_obj)
        arm_obj.matrix_world = Matrix.Identity(4)

        context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")

        bone_count     = [0]
        fallback_count = [0]

        def add_bone(frame_node, parent_edit_bone):
            name = frame_node.name or "Bone"

            if use_ftm_rest:

                # FTM-rest path (canonical .x convention):
                # bone.matrix_local = FTM_global. Animation keys are absolute
                # local TRS replacements for FTM.
                #
                # The xcache parser (see parser._resolve_parent_indices) now
                # detects top-level bones via `ftm == bind_pose` and writes
                # the world bind matrix as their FrameTransformMatrix, so for
                # Bugsnax skeletons FTM_global == engine bind world directly
                # — matching the dev-supplied .x files exactly.
                #
                # The `_bone_rebind` matrix below is kept for safety: if a
                # file's FTM and SkinWeights still disagree after parsing, it
                # rewrites vertex positions so binding still works. For the
                # common case where they agree, this matrix evaluates to
                # identity and the rebind step is a harmless no-op.

                new_rest_bl = conv_mat @ ftm_globals[name] if name in ftm_globals else None
                old_bind_bl = conv_mat @ bind_poses[name]  if name in bind_poses  else None
                if new_rest_bl is None:
                    new_rest_bl = old_bind_bl
                if new_rest_bl is None:
                    rest_mat = Matrix.Identity(4)
                else:
                    rest_mat = new_rest_bl

                if old_bind_bl is not None and new_rest_bl is not old_bind_bl:
                    try:
                        self._bone_rebind[name] = new_rest_bl @ old_bind_bl.inverted()
                    except ValueError:
                        pass
                if name not in bind_poses:
                    fallback_count[0] += 1
            else:

                if name in bind_poses:
                    rest_mat = conv_mat @ bind_poses[name]
                elif name in ftm_globals:
                    rest_mat = conv_mat @ ftm_globals[name]
                    fallback_count[0] += 1
                else:
                    rest_mat = Matrix.Identity(4)

            eb = arm_data.edit_bones.new(name)
            if self.global_scale != 1.0:
                scaled = rest_mat.copy()
                scaled.translation *= self.global_scale
                rest_mat = scaled
            eb.matrix = rest_mat
            if (eb.tail - eb.head).length < 1e-4:
                eb.tail = eb.head + rest_mat.to_3x3() @ Vector((0, 0.1, 0))
            eb.use_connect = False
            if parent_edit_bone:
                eb.parent = parent_edit_bone

            bone_count[0] += 1

            for child in frame_node.children:
                if child.kind == "Frame":
                    add_bone(child, eb)

        for fn in skel_roots:
            add_bone(fn, None)

        bpy.ops.object.mode_set(mode="OBJECT")

    def import_frame_meshes(self, frame_node, context, parent_matrix=None):
        if parent_matrix is None:
            parent_matrix = Matrix.Identity(4)
        ftm       = frame_node.child("FrameTransformMatrix")
        local_mat = _mat4_from_list(ftm.nums()) if ftm else Matrix.Identity(4)
        world_mat = parent_matrix @ local_mat
        for child in frame_node.children:
            if child.kind == "Mesh":
                self._build_mesh(child, context, world_mat, frame_node.name)
            elif child.kind == "Frame":
                self.import_frame_meshes(child, context, world_mat)

    def _build_mesh(self, mesh_node, context, world_mat, frame_name):
        name_hint = mesh_node.name or frame_name or "Mesh"

        nums = mesh_node.nums()
        if not nums:
            return

        vcount = int(nums[0])
        verts  = []
        idx    = 1
        for _ in range(vcount):
            if idx + 3 > len(nums):
                break
            verts.append((nums[idx], nums[idx + 1], nums[idx + 2]))
            idx += 3

        if idx >= len(nums):
            return
        fcount = int(nums[idx]); idx += 1
        faces  = []
        for _ in range(fcount):
            if idx >= len(nums):
                break
            n    = int(nums[idx]); idx += 1
            face = [int(nums[idx + j]) for j in range(n)]; idx += n
            faces.append(face)

        from collections import Counter

        me  = bpy.data.meshes.new(name_hint)
        obj = bpy.data.objects.new(name_hint, me)
        context.collection.objects.link(obj)
        self.created_objects.append(obj)

        # Stash the original DeclData and XSkinMeshHeader source-text
        # blobs (if this file has them) so the exporter can re-emit the
        # exact original bytes on round-trip.  We index by the mesh
        # node's name in the source file.
        _src_mesh_name = mesh_node.name or ""
        if _src_mesh_name:
            decl_text = self._passthrough_decldata.get(_src_mesh_name)
            if decl_text:
                obj["_x_decldata"] = decl_text
            xs_text = self._passthrough_xskinheader.get(_src_mesh_name)
            if xs_text:
                obj["_x_xskinheader"] = xs_text
            # Record the original file's vertex count too: if the user
            # later edits the mesh and the count changes, the exporter
            # will know the stashed DeclData no longer applies.
            obj["_x_orig_vert_count"] = vcount

        M = self._conv_mat
        s = self.global_scale
        if M != Matrix.Identity(4) or s != 1.0:
            verts = [((M @ Vector(v)) * s).to_tuple() for v in verts]
        me.from_pydata(verts, [], faces)
        me.update()

        _pending_loop_normals = None
        normals_node = mesh_node.child("MeshNormals")
        if self.import_normals and normals_node:
            norms = normals_node.nums()
            if norms:
                ncount       = int(norms[0])
                normals_list = []
                ni = 1
                conv_rot = self._conv_mat.to_3x3()
                for _ in range(ncount):
                    if ni + 3 > len(norms):
                        break
                    n_raw = Vector((norms[ni], norms[ni + 1], norms[ni + 2]))
                    normals_list.append((conv_rot @ n_raw).normalized().to_tuple())
                    ni += 3

                if ni < len(norms):
                    ni += 1

                face_norm_indices = []
                for face in faces:
                    cc = len(face)
                    if ni >= len(norms):
                        face_norm_indices.append([0] * cc); continue
                    ni += 1
                    idxs = [int(norms[ni + k]) if ni + k < len(norms) else 0 for k in range(cc)]
                    ni += cc
                    face_norm_indices.append(idxs)

                oob = 0
                loop_normals = []
                for poly, norm_idxs in zip(me.polygons, face_norm_indices):
                    for corner, _ in enumerate(poly.loop_indices):
                        ni_val = norm_idxs[corner] if corner < len(norm_idxs) else 0
                        loop_normals.append(
                            normals_list[ni_val] if ni_val < len(normals_list) else (0.0, 0.0, 1.0)
                        )
                        if ni_val >= len(normals_list):
                            oob += 1
                if oob:
                    pass
                _pending_loop_normals = loop_normals

        uv_node = mesh_node.child("MeshTextureCoords")
        if self.import_uvs and uv_node:
            uvnums  = uv_node.nums()
            uvcount = int(uvnums[0]) if uvnums else 0
            uvs     = []
            ui      = 1
            for _ in range(uvcount):
                if ui + 2 > len(uvnums):
                    break
                uvs.append((uvnums[ui], 1.0 - uvnums[ui + 1]))
                ui += 2
            uv_layer = me.uv_layers.new(name="UVMap")
            uv_miss  = 0
            for poly in me.polygons:
                for loop_idx in poly.loop_indices:
                    vi = me.loops[loop_idx].vertex_index
                    if vi < len(uvs):
                        uv_layer.data[loop_idx].uv = uvs[vi]
                    else:
                        uv_miss += 1
            if uv_miss:
                pass

        # PZ / 3DS Max biped exports often skip MeshNormals and
        # MeshTextureCoords entirely, packing per-vertex normals,
        # tangents, and UVs into a DeclData block instead. If we have a
        # DeclData node and didn't already get normals/UVs from the
        # standard blocks, decode it.
        decl_node = mesh_node.child("DeclData")
        if decl_node is not None and (
            (_pending_loop_normals is None and self.import_normals) or
            (not me.uv_layers and self.import_uvs)
        ):
            decoded = _decode_decl_data(decl_node, vcount)
            if decoded is not None:
                per_vert_normals, per_vert_uvs = decoded
                if (per_vert_normals is not None
                        and self.import_normals
                        and _pending_loop_normals is None):
                    conv_rot = self._conv_mat.to_3x3()
                    converted = [
                        (conv_rot @ Vector(n)).normalized().to_tuple()
                        for n in per_vert_normals
                    ]
                    loop_normals = []
                    for poly in me.polygons:
                        for loop_idx in poly.loop_indices:
                            vi = me.loops[loop_idx].vertex_index
                            if vi < len(converted):
                                loop_normals.append(converted[vi])
                            else:
                                loop_normals.append((0.0, 0.0, 1.0))
                    _pending_loop_normals = loop_normals
                if (per_vert_uvs is not None
                        and self.import_uvs
                        and not me.uv_layers):
                    uv_layer = me.uv_layers.new(name="UVMap")
                    for poly in me.polygons:
                        for loop_idx in poly.loop_indices:
                            vi = me.loops[loop_idx].vertex_index
                            if vi < len(per_vert_uvs):
                                u, v = per_vert_uvs[vi]
                                uv_layer.data[loop_idx].uv = (u, 1.0 - v)

        mat_list_node = mesh_node.child("MeshMaterialList")
        if self.import_materials and mat_list_node:
            mat_nums = mat_list_node.nums()
            if mat_nums:
                face_mat_count   = int(mat_nums[1]) if len(mat_nums) > 1 else 0
                face_mat_indices = [int(mat_nums[2 + i])
                                    for i in range(face_mat_count)
                                    if 2 + i < len(mat_nums)]

                inline_mats = []
                for child in mat_list_node.children:
                    if child.kind == "Material":
                        inline_mats.append(self.parse_material(child, self.base_dir))

                ref_mats = []
                for child in mat_list_node.children:
                    if child.kind == "REF":
                        ref_name = next((v for t, v in child.values if t == "WORD"), None)
                        if ref_name and ref_name in self.materials:
                            ref_mats.append(self.materials[ref_name])
                        elif ref_name:
                            pass

                source = inline_mats or ref_mats or list(self.materials.values())
                seen, used_mats = set(), []
                for m in source:
                    if m.name not in seen:
                        seen.add(m.name); used_mats.append(m)

                for m in used_mats:
                    me.materials.append(m)
                for i, poly in enumerate(me.polygons):
                    if i < len(face_mat_indices):
                        poly.material_index = face_mat_indices[i]

        if self.import_weights and self.armature_obj:
            sw_nodes = mesh_node.children_of("SkinWeights")

            pre_weld_skin = []
            for sw in sw_nodes:
                bone_name = next((v for t, v in sw.values if t == "STR"), None)
                if bone_name is None:
                    bone_name = next((v for t, v in sw.values if t == "WORD"), None)
                sw_nums = sw.nums()
                if not bone_name:
                    continue
                if not sw_nums:
                    continue
                influence_count = int(sw_nums[0])
                indices = [int(sw_nums[1 + i])             for i in range(influence_count)]
                weights = [sw_nums[1 + influence_count + i] for i in range(influence_count)]
                pre_weld_skin.append((bone_name, list(zip(indices, weights))))

            obj.matrix_world  = Matrix.Identity(4)
            obj.parent        = self.armature_obj
            arm_mod           = obj.modifiers.new("Armature", "ARMATURE")
            arm_mod.object    = self.armature_obj
            arm_mod.use_vertex_groups = True

            arm_mod.use_deform_preserve_volume = True

            if self._bone_rebind:
                vert_accum      = [None] * len(me.vertices)
                vert_weight_sum = [0.0]  * len(me.vertices)
                for bone_name, influences in pre_weld_skin:
                    rebind = self._bone_rebind.get(bone_name)
                    if rebind is None:
                        continue
                    for vi, w in influences:
                        if vi >= len(vert_accum):
                            continue
                        contrib = rebind * w
                        if vert_accum[vi] is None:
                            vert_accum[vi] = contrib
                        else:
                            for r in range(4):
                                for c in range(4):
                                    vert_accum[vi][r][c] += contrib[r][c]
                        vert_weight_sum[vi] += w
                rebind_applied = 0
                for vi, accum in enumerate(vert_accum):
                    if accum is None or vert_weight_sum[vi] <= 0.0:
                        continue
                    v_old = me.vertices[vi].co.copy()
                    v_h   = Vector((v_old.x, v_old.y, v_old.z, 1.0))
                    v_new = accum @ v_h
                    ws    = vert_weight_sum[vi]
                    me.vertices[vi].co = Vector((v_new.x / ws,
                                                 v_new.y / ws,
                                                 v_new.z / ws))
                    rebind_applied += 1
                me.update()

            import bmesh as _bmesh
            pre_weld_positions = [tuple(v.co) for v in me.vertices]
            _pre_count = len(pre_weld_positions)

            EPS = 5

            vi_to_weights: dict = {}
            for bn, infls in pre_weld_skin:
                for vi, w in infls:
                    vi_to_weights.setdefault(vi, []).append((bn, round(w, EPS)))
            vi_weight_sig = [
                tuple(sorted(vi_to_weights.get(vi, [])))
                for vi in range(_pre_count)
            ]

            key_to_vis: dict = {}
            for vi in range(_pre_count):
                p = pre_weld_positions[vi]
                key = (round(p[0], EPS), round(p[1], EPS), round(p[2], EPS),
                       vi_weight_sig[vi])
                key_to_vis.setdefault(key, []).append(vi)

            pre_to_post: dict = {}
            keeper_pre_vis = []
            for vis in key_to_vis.values():
                keep = min(vis)
                keeper_pre_vis.append(keep)
            keeper_pre_vis.sort()
            keep_to_post: dict = {keep: i for i, keep in enumerate(keeper_pre_vis)}

            pre_to_keep: dict = {}
            for vis in key_to_vis.values():
                keep = min(vis)
                for vi in vis:
                    pre_to_keep[vi] = keep
            for vi in range(_pre_count):
                pre_to_post[vi] = keep_to_post[pre_to_keep[vi]]
            _post_count = len(keeper_pre_vis)

            _bm = _bmesh.new()
            _bm.from_mesh(obj.data)
            _bm.verts.ensure_lookup_table()

            targetmap = {}
            for vis in key_to_vis.values():
                if len(vis) < 2:
                    continue
                keep = min(vis)
                keep_v = _bm.verts[keep]
                for other in vis:
                    if other != keep:
                        targetmap[_bm.verts[other]] = keep_v

            if targetmap:
                _bmesh.ops.weld_verts(_bm, targetmap=targetmap)

            _bm.verts.ensure_lookup_table()
            _bm.to_mesh(obj.data)
            _bm.free()
            obj.data.update()

            remap_miss = _pre_count - len(pre_to_post)
            if remap_miss:
                pass

            post_vert_pre_vis: dict = {}

            for bone_name, influences in pre_weld_skin:
                vg = obj.vertex_groups.get(bone_name) or obj.vertex_groups.new(name=bone_name)
                for pre_vi, w in influences:
                    post_vi = pre_to_post.get(pre_vi)
                    if post_vi is None:
                        continue
                    seen_pre = post_vert_pre_vis.get(post_vi)
                    if seen_pre is None:

                        post_vert_pre_vis[post_vi] = {pre_vi}
                        vg.add([post_vi], w, "REPLACE")
                    elif pre_vi in seen_pre:

                        vg.add([post_vi], w, "REPLACE")
                    else:

                        pass

            # Some source meshes have verts that no SkinWeights block
            # touches — usually a SkinWeights entry was lost when the
            # cache was baked, leaving (often mirrored) tip verts with
            # zero total weight. They'd stay at rest pose while the rest
            # of the mesh animates, which manifests as a small cluster
            # hanging in space (e.g. the right-tusk tip on BurgerB).
            #
            # Fill them in by copying the weight assignment from the
            # geometrically closest weighted vert. This is what 3DS Max
            # and Maya do when displaying partially-weighted skins; the
            # engine appears to do the same implicitly at load time.
            _fill_unweighted_from_neighbours(obj)

            for poly in obj.data.polygons:
                poly.use_smooth = True
            obj.data.update()
            if getattr(self, "smooth_shade_from_faces", False):
                # Smooth-shade from faces: every poly is marked smooth so
                # Blender averages face normals at shared vertices. Any
                # custom split normals on the mesh are explicitly cleared
                # so they don't override the auto-smoothed result.
                _clear_custom_split_normals(obj.data)
            elif _pending_loop_normals is not None:
                # File-authored normals path: apply the exact per-loop
                # normals decoded from MeshNormals or DeclData.
                _apply_custom_normals(obj.data, _pending_loop_normals)
                if self.infer_sharps:
                    self._infer_sharps_from_normal_list(obj.data, _pending_loop_normals)

        else:

            obj.matrix_world = Matrix.Identity(4)

            for poly in obj.data.polygons:
                poly.use_smooth = True
            obj.data.update()
            if getattr(self, "smooth_shade_from_faces", False):
                _clear_custom_split_normals(obj.data)
            elif _pending_loop_normals is not None:
                _apply_custom_normals(obj.data, _pending_loop_normals)
                if self.infer_sharps:
                    self._infer_sharps_from_normal_list(obj.data, _pending_loop_normals)

    def _infer_sharps_from_normal_list(self, me, loop_normals):
        import math as _math

        threshold_cos = _math.cos(_math.radians(self.sharp_angle_deg))

        if hasattr(me, "calc_normals"):
            me.calc_normals()
        else:
            me.update()

        edge_to_polys = {}
        for poly in me.polygons:
            for i in range(poly.loop_total):
                li = poly.loop_start + i
                ei = me.loops[li].edge_index
                edge_to_polys.setdefault(ei, []).append(poly.index)

        sharp_count = 0
        for edge in me.edges:
            polys = edge_to_polys.get(edge.index, [])
            if len(polys) != 2:
                continue
            p0 = me.polygons[polys[0]]; p1 = me.polygons[polys[1]]
            n0 = p0.normal; n1 = p1.normal
            cos_a = max(-1.0, min(1.0, n0.dot(n1)))
            if cos_a < threshold_cos:
                edge.use_edge_sharp = True
                sharp_count += 1

        # Replace the file's per-loop normals with Blender-computed smooth
        # shading that breaks at the sharp edges we just marked. This gives
        # a clean smooth-shaded appearance while preserving the hard creases
        # the geometry asked for (the file's loop normals are typically
        # already smoothed, but go through them verbatim and you get the
        # mesh's flat-shading character baked in).
        _clear_custom_split_normals(me)
        for poly in me.polygons:
            poly.use_smooth = True
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True
        me.update()

    def import_animation_set(self, anim_set_node, context):
        if not self.armature_obj:
            return

        arm_obj = self.armature_obj
        scene   = context.scene

        target_fps = int(round(self.ticks_per_second))
        if scene.render.fps != target_fps:
            scene.render.fps = target_fps
        else:
            pass

        arm_obj.animation_data_create()
        anim_data = arm_obj.animation_data

        if anim_data.action is not None:
            track = anim_data.nla_tracks.new()
            track.name = anim_data.action.name
            start_frame = int(anim_data.action.frame_range[0])
            track.strips.new(anim_data.action.name, start_frame, anim_data.action)
            anim_data.action = None

        action = bpy.data.actions.new(anim_set_node.name or "Action")

        if hasattr(action, "slots"):

            slot = action.slots.new(id_type="OBJECT", name=arm_obj.name)
            anim_data.action = action
            anim_data.action_slot = slot
        else:
            anim_data.action = action

        context.view_layer.objects.active = arm_obj

        bpy.ops.object.mode_set(mode="POSE")

        anim_nodes = anim_set_node.children_of("Animation")
        keyframe_errors = 0

        for anim_node in anim_nodes:
            ref = anim_node.child("REF")
            if not ref:
                continue
            bone_name = next((v for t, v in ref.values if t == "WORD"), None)
            if not bone_name:
                continue
            if bone_name not in arm_obj.pose.bones:
                continue

            pose_bone = arm_obj.pose.bones[bone_name]
            key_nodes = anim_node.children_of("AnimationKey")

            _tracks_by_type = {}
            for _kn in key_nodes:
                _kn_nums = _kn.nums()
                if len(_kn_nums) >= 2:
                    _tracks_by_type[int(_kn_nums[0])] = _kn
            if (0 in _tracks_by_type and 1 in _tracks_by_type
                    and 2 in _tracks_by_type and 4 not in _tracks_by_type):
                _synth = _compose_type4_from_trs(
                    _tracks_by_type[0], _tracks_by_type[1], _tracks_by_type[2])
                if _synth is not None:
                    key_nodes = [_synth]

            for key_node in key_nodes:
                key_nums = key_node.nums()
                if len(key_nums) < 2:
                    continue

                key_type      = int(key_nums[0])
                key_count     = int(key_nums[1])
                expected_vals = _KEY_TYPE_VALUES.get(key_type)
                if expected_vals is None:
                    continue

                all_key_vals = []
                i = 2
                while i < len(key_nums):
                    if i + 1 >= len(key_nums): break
                    frame_tick = key_nums[i]; i += 1
                    i += 1
                    if i + expected_vals > len(key_nums): break
                    all_key_vals.append((frame_tick, key_nums[i:i + expected_vals]))
                    i += expected_vals
                    if len(all_key_vals) >= key_count: break

                local_rest_q = None
                is_skel_root  = (bone_name in self._skel_root_names)

                _conv_q = self._conv_mat.to_3x3().to_quaternion()

                if key_type == 0:
                    pb = pose_bone
                    if pb.parent:

                        local_rest_bl = (pb.parent.bone.matrix_local.inverted()
                                         @ pb.bone.matrix_local)
                        local_rest_q  = local_rest_bl.to_quaternion()
                    elif is_skel_root:

                        # The skeleton root has no Blender parent, but its
                        # bind pose may have a non-identity rotation (e.g.
                        # Celery's ROOT is tilted ~2.2°, SweetFry's ~3°).
                        # For those, we need to subtract the bind rotation
                        # from abs_q so that pose_q represents the animated
                        # rotation relative to the bind — the same thing
                        # the non-root branch does implicitly.
                        #
                        # bone.matrix_local = conv @ src_bind_col, so the
                        # un-conv'd bind rotation is conv.inv @ matrix_local.
                        # For characters with identity ROOT bind (Apple,
                        # ChiDog, Lizbert, etc.) this is identity and
                        # local_rest_q is identity — matching the previous
                        # behaviour exactly.
                        _src_bind_bl = (self._conv_mat.inverted()
                                        @ pb.bone.matrix_local)
                        local_rest_q = _src_bind_bl.to_3x3().to_quaternion()
                    else:

                        local_rest_q = pb.bone.matrix_local.to_quaternion()

                if key_type == 2:
                    pb = pose_bone
                    if pb.parent:
                        _t2_local_rest = (pb.parent.bone.matrix_local.inverted()
                                          @ pb.bone.matrix_local)
                        _t2_lock = self.lock_leaf_translation and not pb.children
                    else:
                        _t2_local_rest = pb.bone.matrix_local
                        _t2_lock = self.lock_root_translation
                    _t2_rest_head    = _t2_local_rest.to_translation()
                    _t2_rest_rot_inv = _t2_local_rest.to_3x3().inverted()
                    _t2_conv3        = self._conv_mat.to_3x3()

                chan_data: dict = {}

                prev_pose_q = None
                _dbg_done   = False

                for frame_tick, vals in all_key_vals:
                    # Scale file ticks to Blender frame numbers. tick_scale
                    # is normally 1.0; for files with very high tick rates
                    # (e.g. PZ's 4800/sec) it normalizes the timeline to a
                    # sane FPS so the animation isn't spread across
                    # thousands of frames.
                    frame = float(frame_tick) * self.tick_scale
                    try:
                        if key_type == 0:
                            abs_q  = Quaternion((vals[0], -vals[1], -vals[2], -vals[3]))
                            pose_q = local_rest_q.inverted() @ abs_q
                            if prev_pose_q is not None:
                                pose_q.make_compatible(prev_pose_q)
                            prev_pose_q = Quaternion(pose_q)

                            if is_skel_root and not _dbg_done:
                                _dbg_done = True
                                _w3 = (pose_bone.bone.matrix_local @ pose_q.to_matrix().to_4x4()).to_3x3()

                            for ci, v in enumerate(pose_q):
                                chan_data.setdefault(ci, []).append((frame, v))

                        elif key_type == 1:
                            for ci, v in enumerate(vals[:3]):
                                chan_data.setdefault(ci, []).append((frame, v))

                        elif key_type == 2:
                            if _t2_lock:
                                loc = Vector((0.0, 0.0, 0.0))
                            elif pose_bone.parent:
                                anim_t = Vector((vals[0], vals[1], vals[2]))
                                loc = _t2_rest_rot_inv @ (anim_t - _t2_rest_head)
                            else:
                                anim_t = _t2_conv3 @ (
                                    Vector((vals[0], vals[1], vals[2])) * self.global_scale)
                                loc = _t2_rest_rot_inv @ (anim_t - _t2_rest_head)

                            for ci, v in enumerate(loc):
                                chan_data.setdefault(ci, []).append((frame, v))

                        elif key_type == 4:

                            dx_local = _mat4_from_list(vals)
                            if pose_bone.parent:
                                matrix_basis = (pose_bone.bone.matrix_local.inverted()
                                                @ pose_bone.parent.bone.matrix_local
                                                @ dx_local)
                            else:
                                matrix_basis = (pose_bone.bone.matrix_local.inverted()
                                                @ self._conv_mat
                                                @ dx_local)
                            mb_loc = matrix_basis.to_translation()
                            mb_rot = matrix_basis.to_quaternion()
                            mb_sca = matrix_basis.to_scale()
                            if prev_pose_q is not None:
                                mb_rot.make_compatible(prev_pose_q)
                            prev_pose_q = Quaternion(mb_rot)
                            for ci, v in enumerate(mb_loc):
                                chan_data.setdefault(10 + ci, []).append((frame, v))
                            for ci, v in enumerate(mb_rot):
                                chan_data.setdefault(20 + ci, []).append((frame, v))
                            for ci, v in enumerate(mb_sca):
                                chan_data.setdefault(30 + ci, []).append((frame, v))

                    except Exception as exc:
                        keyframe_errors += 1

                _ROT_PATH   = "pose.bones[\"%s\"].rotation_quaternion"
                _SCALE_PATH = "pose.bones[\"%s\"].scale"
                _LOC_PATH   = "pose.bones[\"%s\"].location"
                _data_path_map = {
                    0: (_ROT_PATH,   range(4)),
                    1: (_SCALE_PATH, range(3)),
                    2: (_LOC_PATH,   range(3)),
                }
                if key_type in (0, 1, 2):
                    path_tmpl, indices = _data_path_map[key_type]
                    path = path_tmpl % bone_name
                    pose_bone.rotation_mode = "QUATERNION"
                    for ci in indices:
                        pts = chan_data.get(ci)
                        if not pts:
                            continue
                        fc = _get_or_create_fcurve(action, anim_data, path, ci, bone_name)
                        fc.keyframe_points.add(len(pts))
                        for kp, (fr, v) in zip(fc.keyframe_points[-len(pts):], pts):
                            kp.co = (fr, v)
                            kp.interpolation = "LINEAR"
                        fc.update()
                elif key_type == 4:
                    pose_bone.rotation_mode = "QUATERNION"
                    for offset, path_tmpl, indices in [
                        (10, _LOC_PATH,   range(3)),
                        (20, _ROT_PATH,   range(4)),
                        (30, _SCALE_PATH, range(3)),
                    ]:
                        path = path_tmpl % bone_name
                        for ci in indices:
                            pts = chan_data.get(offset + ci)
                            if not pts:
                                continue
                            fc = _get_or_create_fcurve(action, anim_data, path, ci, bone_name)
                            fc.keyframe_points.add(len(pts))
                            for kp, (fr, v) in zip(fc.keyframe_points[-len(pts):], pts):
                                kp.co = (fr, v)
                                kp.interpolation = "LINEAR"
                            fc.update()

        bpy.ops.object.mode_set(mode="OBJECT")"""
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

        seen_offsets.add(geo_off)

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
    """Recover each bone's parent by geometric inference from its FTM.

    The .xcache `parent_idx` field uses an opaque encoding we don't fully
    reverse-engineer.  Instead we exploit the invariant that for any bone i,
    its world-space `bind_pose` equals `ftm @ parent_bind_pose` (in DX
    row-major × row-vector convention).  We search prior bones for the one
    whose bind_pose makes that equation hold, with tolerance 1e-3.

    Falls back to ROOT (index 0) if no candidate matches — covers ROOT
    itself, which has parent = -1.
    """
    parents = [-1] * len(bones)
    if not bones:
        return parents

    for i in range(1, len(bones)):
        ftm  = bones[i]['ftm']
        bind = bones[i]['bind_pose']
        best, best_err = 0, float('inf')
        for j in range(i):
            par_bind = bones[j]['bind_pose']
            comp = _mat_mul(ftm, par_bind)
            err = max(abs(comp[k] - bind[k]) for k in range(16))
            if err < best_err:
                best_err = err
                best = j
        parents[i] = best if best_err < 1e-3 else 0

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

    is_skel = [not _is_mesh_bone_name(b['name']) for b in bones]
    parents = _resolve_parent_indices(bones)

    # Use the FTM stored in the file directly — it's already parent-local.
    # The parent inference in _resolve_parent_indices guarantees that
    # ftm @ parent_bind == bind, so this matches the geometric truth.
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
