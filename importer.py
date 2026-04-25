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
    """Minimal object mimicking an XNode AnimationKey for synthetic tracks.

    Only .nums() is used by the AnimationKey consumer code, so we don't need
    to implement the full XNode interface.
    """
    __slots__ = ("_nums",)

    def __init__(self, nums_list):
        self._nums = nums_list

    def nums(self):
        return self._nums

def _compose_type4_from_trs(rot_node, scale_node, trans_node):
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

    n = min(len(rot_frames), len(scale_frames), len(trans_frames))
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
             rest_pose_source='BIND',
             infer_sharps=True,
             sharp_angle_deg=75.0,
             lock_root_translation=False,
             lock_leaf_translation=False,
             **_):

    root     = parse_x_file(filepath)
    base_dir = os.path.dirname(filepath)

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
    )

    for node in root.children:
        if node.kind == "AnimTicksPerSecond":
            nums = node.nums()
            if nums:
                state.ticks_per_second = nums[0]
    if anim_fps and anim_fps > 0:
        state.ticks_per_second = anim_fps

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

class _ImportState:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.materials:          dict  = {}
        self.armature_obj:       object = None
        self.ticks_per_second:   float  = 30.0
        self.created_objects:    list   = []
        self._conv_mat           = Matrix.Identity(4)
        self._always_hidden_bones: set = set()
        self._skel_root_names:   set    = set()
        self._bone_rebind:       dict   = {}

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
            roughness = max(0.0, min(1.0, 1.0 - math.log(max(shininess, 1e-6)) / math.log(128.0)))
            bsdf.inputs["Roughness"].default_value = roughness
            mat["_x_power"] = shininess
        if len(nums) >= 8:
            sr, sg, sb = nums[5], nums[6], nums[7]
            spec_val   = (sr + sg + sb) / 3.0
            spec_input = (bsdf.inputs.get("Specular IOR Level")
                          or bsdf.inputs.get("Specular"))
            if spec_input:
                spec_input.default_value = spec_val
            else:
                pass
            mat["_x_specular"] = (sr, sg, sb)
        if len(nums) >= 11:
            mat["_x_emissive"] = (nums[8], nums[9], nums[10])

        if self.import_textures:
            tex_node = node.child("TextureFileName") or node.child("TextureFilename")
            if tex_node:
                strs = tex_node.strings()
                if strs:

                    original_tex_name = strs[0].replace('\\\\', '\\')
                    mat["_x_texture_filename"] = original_tex_name
                    tex_path  = original_tex_name.replace("\\", os.sep).replace("/", os.sep)
                    full_path = os.path.join(base_dir, tex_path)
                    stem      = os.path.splitext(full_path)[0]
                    loaded    = False
                    for candidate in [full_path,
                                      stem + ".png", stem + ".jpg", stem + ".tga",
                                      stem + ".dds"]:
                        if os.path.exists(candidate):
                            img          = bpy.data.images.load(candidate, check_existing=True)
                            tex_img_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
                            tex_img_node.image = img
                            mat.node_tree.links.new(
                                tex_img_node.outputs["Color"],
                                bsdf.inputs["Base Color"],
                            )
                            loaded = True
                            break
                    if not loaded:
                        pass

        self.materials[name] = mat
        return mat

    def build_armature(self, frame_nodes, context, bind_poses, ftm_globals, conv_mat):
        self._conv_mat = conv_mat

        use_ftm_rest = (self.rest_pose_source == 'FRAME_TRANSFORM')
        self._bone_rebind = {}

        skel_roots = [f for f in frame_nodes
                      if any(c.kind == "Frame" for c in f.children)]

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

            for poly in obj.data.polygons:
                poly.use_smooth = True
            obj.data.update()
            if _pending_loop_normals is not None and self.infer_sharps:
                self._infer_sharps_from_normal_list(obj.data, _pending_loop_normals)

        else:

            obj.matrix_world = Matrix.Identity(4)

            for poly in obj.data.polygons:
                poly.use_smooth = True
            obj.data.update()
            if _pending_loop_normals is not None and self.infer_sharps:
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

        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True

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

                        local_rest_q = Quaternion()
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
                    frame = float(frame_tick)
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

        bpy.ops.object.mode_set(mode="OBJECT")