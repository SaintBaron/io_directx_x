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
import re
import bpy
from mathutils import Matrix, Vector, Quaternion

from .parser import parse_x_file


_KEY_TYPE_VALUES = {
    # Per Microsoft Learn AnimationKey spec, keyType values are:
    #   0 = rotation (4-float quaternion)
    #   1 = scale    (3-float vector)
    #   2 = position (3-float vector)
    #   3 = matrix   (16-float 4x4)
    # some files files also use keyType=4 for matrix tracks
    # (an engine-specific extension/deviation). Both 3 and 4 are
    # treated as 16-float matrix keys downstream.
    0: 4,
    1: 3,
    2: 3,
    3: 16,
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


def _is_near_identity(M, tol=1e-4):
    """True if 4x4 matrix M is within tol of the identity in every
    element. Used to skip a bind->FTM rebind that is effectively a no-op
    (the file's frame hierarchy already equals its skin bind, as in
    some exporters-native .x exports) so those import bit-for-bit unchanged.
    A genuine rebind differs from identity by whole units and is kept."""
    ident = Matrix.Identity(4)
    return all(abs(M[r][c] - ident[r][c]) < tol
               for r in range(4) for c in range(4))

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
    tracks share identical tick sequences (e.g. some exports).
    For files where the tracks have different key densities — e.g. Project
    some games, where scale is often just 2 keys while rotation has 21 — the
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


def _compose_type4_from_trs_resampled(rot_node, scale_node, trans_node):
    """Like _compose_type4_from_trs, but tolerant of tracks with DIFFERENT
    tick sequences. Each track (rotation quaternion, scale, translation) is
    linearly resampled at the union of all three tracks' ticks (quaternions
    via nlerp), then composed into a per-frame type-4 matrix key sequence.

    This is used only for bones inside a scaled subtree, where the per-track
    F-curve path cannot apply the scale-inheritance correction. Composing to
    a matrix lets those bones flow through the verified _scaled_subtree_basis
    path. For unscaled bones the plain per-track path is kept (well-tested
    and avoids resampling), so this only ever affects the scaled-bone case.
    Returns a _SyntheticKeyNode, or None if any track is empty/unparseable.
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
            out.append((float(tick), nums[i:i + expected]))
            i += expected
        return out

    rot_frames   = _parse(rot_node,   4)
    scale_frames = _parse(scale_node, 3)
    trans_frames = _parse(trans_node, 3)
    if not rot_frames or not scale_frames or not trans_frames:
        return None

    def _sample_vec(frames, t, n):
        # Linear interpolation of an n-float track at tick t, clamped at ends.
        if t <= frames[0][0]:
            return list(frames[0][1][:n])
        if t >= frames[-1][0]:
            return list(frames[-1][1][:n])
        for j in range(len(frames) - 1):
            t0, v0 = frames[j]; t1, v1 = frames[j + 1]
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return [v0[k] + a * (v1[k] - v0[k]) for k in range(n)]
        return list(frames[-1][1][:n])

    def _sample_quat(frames, t):
        # nlerp between the two bracketing quaternion keys (good enough for
        # the dense facial tracks; matches Blender's own per-frame sampling
        # closely and avoids importing slerp here).
        if t <= frames[0][0]:
            q = list(frames[0][1][:4])
        elif t >= frames[-1][0]:
            q = list(frames[-1][1][:4])
        else:
            q = list(frames[-1][1][:4])
            for j in range(len(frames) - 1):
                t0, q0 = frames[j]; t1, q1 = frames[j + 1]
                if t0 <= t <= t1:
                    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                    q0 = q0[:4]; q1 = q1[:4]
                    # flip for shortest path
                    dot = sum(q0[k] * q1[k] for k in range(4))
                    if dot < 0.0:
                        q1 = [-c for c in q1]
                    q = [q0[k] + a * (q1[k] - q0[k]) for k in range(4)]
                    break
        n = math.sqrt(sum(c * c for c in q)) or 1.0
        return [c / n for c in q]

    ticks = sorted({t for t, _ in rot_frames}
                   | {t for t, _ in scale_frames}
                   | {t for t, _ in trans_frames})
    if not ticks:
        return None

    out_nums = [4.0, float(len(ticks))]
    for t in ticks:
        w, x, y, z = _sample_quat(rot_frames, t)
        sx, sy, sz = _sample_vec(scale_frames, t, 3)
        tx, ty, tz = _sample_vec(trans_frames, t, 3)

        r00 = 1.0 - 2.0*(y*y + z*z);  r01 =       2.0*(x*y - z*w);  r02 =       2.0*(x*z + y*w)
        r10 =       2.0*(x*y + z*w);  r11 = 1.0 - 2.0*(x*x + z*z);  r12 =       2.0*(y*z - x*w)
        r20 =       2.0*(x*z - y*w);  r21 =       2.0*(y*z + x*w);  r22 = 1.0 - 2.0*(x*x + y*y)

        r00 *= sx; r01 *= sx; r02 *= sx
        r10 *= sy; r11 *= sy; r12 *= sy
        r20 *= sz; r21 *= sz; r22 *= sz

        out_nums.extend([
            t, 16.0,
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
                # matrixOffset comes after n vis + n weights.
                # For empty SW (n=0), it's at index 1.
                mo_start = 1 + n + n
                if len(nums) < mo_start + 16:
                    for child in node.children:
                        walk(child)
                    return
                offset_vals = nums[mo_start : mo_start + 16]
                # Validate: skip identity matrices (likely uninit) and
                # singular matrices. A non-trivial bind matrix has a
                # finite determinant != 1.0 in the rotation part OR
                # a non-zero translation. Identity matrices belong
                # to bones that never moved from world origin OR
                # placeholders that the file never filled in.
                offset_mat = _mat4_from_list(offset_vals)
                # Identity check: any non-trivial bind matrix has elements
                # that differ from identity by more than rounding noise.
                # Identity means "no bind data" — skip and let FTM fallback
                # handle this bone.
                ident = Matrix.Identity(4)
                diff = offset_mat - ident
                is_identity = max(abs(diff[r][c]) for r in range(4)
                                                 for c in range(4)) < 1e-5
                if is_identity:
                    for child in node.children:
                        walk(child)
                    return
                try:
                    bind_poses[bone_name] = offset_mat.inverted()
                except ValueError:
                    pass
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


def _collect_ftm_globals_by_node(frame_node, parent_mat=None, out=None):
    """Like _collect_ftm_globals, but keyed by the frame XNode object
    rather than its name string.

    The name-keyed map collapses frames that share a name into a single
    entry (DirectX .x permits non-unique frame names, and exporters lean
    on this: a chain of unnamed helper frames all key to ""). When that
    happens, every colliding frame gets whichever global transform was
    written last, so most of them are positioned wrong. Keying by node
    identity gives each frame its own correct global FTM regardless of
    naming, which is what the bone rest positions must be built from.
    The name-keyed map is still returned by _collect_ftm_globals for
    SkinWeights resolution, which is inherently name-based.
    """
    if out is None:
        out = {}
    if parent_mat is None:
        parent_mat = Matrix.Identity(4)
    ftm       = frame_node.child("FrameTransformMatrix")
    local_mat = _mat4_from_list(ftm.nums()) if ftm else Matrix.Identity(4)
    global_mat = parent_mat @ local_mat
    out[id(frame_node)] = global_mat
    for child in frame_node.children:
        if child.kind == "Frame":
            _collect_ftm_globals_by_node(child, global_mat, out)
    return out

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

def _build_cross_content_search_paths(basename: str, anchor_dir: str) -> list:
    """Return candidate absolute paths for a texture that may live in a
    different subfolder than the file being imported.

    some games textures are stored at full engine paths such as:
        Content/Models/Bugs/Banana/Banana_D.dds

    When the xcache is in e.g. Content/Characters/Banana/ the existing
    four search paths don't reach the Models/ subtree.  This function
    adds three more tiers:

      1. Derive a likely subdirectory name from the texture stem by
         stripping common material suffixes (_D, _N, _S, _AO, …) and
         probe that named sibling folder at every ancestor level up to
         5 hops.  This is the highest-priority hit for the standard
         some games layout: Banana_D.dds → probe …/Banana/Banana_D.dds.

      2. Scan all sibling directories at the same level as anchor_dir.

      3. Walk up to find a "Content" ancestor and do a depth-2 scan of
         every folder inside it (e.g. Content/Models/Bugs/*/Banana_D.dds).
    """
    candidates = []
    stem_full = os.path.splitext(basename)[0]   # "Banana_D"

    # Strip common material-map suffixes to get the likely folder name.
    dir_stem = re.sub(
        r'_(?:D|N|S|R|E|M|AO|ARM|Mask|Roughness|Metalness|Normal|Diffuse|Specular)$',
        '', stem_full, flags=re.IGNORECASE
    ) or stem_full   # "Banana"

    # 1. Stem-named sibling at each ancestor level (highest priority)
    ancestor = anchor_dir
    for _ in range(5):
        parent = os.path.dirname(ancestor)
        if not parent or parent == ancestor:
            break
        candidates.append(os.path.join(parent, dir_stem, basename))
        ancestor = parent

    # 2. All peer sibling directories at anchor_dir's level
    parent = os.path.dirname(anchor_dir)
    if parent and parent != anchor_dir:
        try:
            for sibling in os.listdir(parent):
                sib_path = os.path.join(parent, sibling)
                if os.path.isdir(sib_path):
                    candidates.append(os.path.join(sib_path, basename))
        except OSError:
            pass

    # 3. Walk up to find a "Content" ancestor, depth-2 scan inside it
    content_root = None
    check = anchor_dir
    for _ in range(6):
        if os.path.basename(check).lower() == "content":
            content_root = check
            break
        up = os.path.dirname(check)
        if not up or up == check:
            break
        check = up

    if content_root:
        try:
            for top in os.listdir(content_root):
                top_path = os.path.join(content_root, top)
                if not os.path.isdir(top_path):
                    continue
                # Direct stem-named subdir: e.g. Content/Models/Bugs/Banana/
                candidates.append(os.path.join(top_path, dir_stem, basename))
                try:
                    for sub in os.listdir(top_path):
                        sub_path = os.path.join(top_path, sub)
                        if os.path.isdir(sub_path):
                            candidates.append(os.path.join(sub_path, basename))
                except OSError:
                    continue
        except OSError:
            pass

    return candidates


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
             infer_sharps=True,
             sharp_angle_deg=75.0,
             lock_root_translation=False,
             lock_leaf_translation=False,
             smooth_shade_from_faces=False,
             weld_duplicate_verts=True,
             import_type="standard",
             use_diffuse_alpha=True,
             split_submeshes=True,
             triangulate_quads=True,
             **_):

    # Rest pose is the FrameTransformMatrix hierarchy (the reference exporter's viewport
    # convention). The mesh is rebound from its authored bind pose into that
    # rest via a per-mesh bake — a no-op when a file's FTM already equals its
    # bind (some exporters-native exports) and the correct fix when they differ
    # (various game tools characters authored in a separately-scaled
    # bind pose).

    root = parse_x_file(filepath,
                        split_submeshes=split_submeshes,
                        triangulate_quads=triangulate_quads)

    # Import path is chosen explicitly by the user (import_type), not guessed.
    #   standard : general .x (various exporters and game tools, etc.)
    #   ms_dx8   : some editors DirectX 8.0 — single skinned mesh w/ matrixOffset
    # The some editors DX8 path is independent of the standard path; standard
    # behaviour is exactly as it was before it was added.
    is_ms_dx8 = (import_type == "ms_dx8")

    def _count(node, kind):
        n = 0
        if node.kind == kind:
            n += 1
        for c in node.children:
            n += _count(c, kind)
        return n
    n_frames = _count(root, 'Frame')
    n_meshes = _count(root, 'Mesh')
    n_sw = _count(root, 'SkinWeights')
    n_anim = _count(root, 'Animation')

    base_dir = os.path.dirname(filepath)

    # ----- Passthrough text capture -----
    # The exporter doesn't natively re-emit template declarations or
    # "auxiliary" frames like high-precision's Translation_Data (which exists at the
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
            for _m in re.finditer(r'\btemplate\s+\w+\s*\{', _full_text):
                _start = _m.start()
                _open_brace = _full_text.index('{', _start)
                _end = _extract_braced_block(_full_text, _open_brace)
                passthrough_templates.append(_full_text[_start:_end].rstrip())

            # Any top-level Frame in the source file
            top_frame_names = {c.name for c in root.children if c.kind == "Frame"}
            for _m in re.finditer(r'(?m)^\s*Frame\s+(\w+)\s*\{', _full_text):
                _name = _m.group(1)
                if _name not in top_frame_names:
                    continue
                _start = _full_text.find('Frame', _m.start())
                _open_brace = _full_text.index('{', _start)
                _end = _extract_braced_block(_full_text, _open_brace)
                passthrough_frames[_name] = _full_text[_start:_end].rstrip()

            # Per-mesh DeclData and XSkinMeshHeader
            for _m in re.finditer(r'(?m)^\s*Mesh\s+(\w+)\s*\{', _full_text):
                _mesh_name = _m.group(1)
                _mesh_start = _full_text.find('Mesh', _m.start())
                _mesh_open  = _full_text.index('{', _mesh_start)
                _mesh_end   = _extract_braced_block(_full_text, _mesh_open)
                _mesh_text  = _full_text[_mesh_start:_mesh_end]
                for _kind, _stash in [
                    ('DeclData', passthrough_decldata),
                    ('XSkinMeshHeader', passthrough_xskinheader),
                ]:
                    _km = re.search(r'\b' + _kind + r'\s*\{', _mesh_text)
                    if _km is None:
                        continue
                    _ks = _km.start()
                    _ko = _mesh_text.index('{', _ks)
                    _ke = _extract_braced_block(_mesh_text, _ko)
                    _stash[_mesh_name] = _mesh_text[_ks:_ke].rstrip()

            # Per-target Animation blocks (so non-bone frames like
            # Translation_Data still get a track on round-trip)
            for _m in re.finditer(r'AnimationSet\s+\w+\s*\{', _full_text):
                _line_start = _full_text.rfind('\n', 0, _m.start()) + 1
                if 'template' in _full_text[_line_start:_m.start()]:
                    continue
                _as_start = _m.start()
                _as_open  = _full_text.index('{', _as_start)
                _as_end   = _extract_braced_block(_full_text, _as_open)
                _as_text  = _full_text[_as_start:_as_end]
                for _am in re.finditer(r'Animation\s*\{', _as_text):
                    _aline_start = _as_text.rfind('\n', 0, _am.start()) + 1
                    if 'template' in _as_text[_aline_start:_am.start()]:
                        continue
                    _a_start = _am.start()
                    _a_open  = _as_text.index('{', _a_start)
                    _a_end   = _extract_braced_block(_as_text, _a_open)
                    _a_text  = _as_text[_a_start:_a_end]
                    _ref = re.search(r'\{\s*(\w+)\s*\}', _a_text)
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
        infer_sharps=infer_sharps,
        sharp_angle_deg=sharp_angle_deg,
        lock_root_translation=lock_root_translation,
        lock_leaf_translation=lock_leaf_translation,
        smooth_shade_from_faces=smooth_shade_from_faces,
        weld_duplicate_verts=weld_duplicate_verts,
        use_diffuse_alpha=use_diffuse_alpha,
        split_submeshes=split_submeshes,
    )
    state._passthrough_decldata = passthrough_decldata
    state._passthrough_xskinheader = passthrough_xskinheader
    # some editors DX8 meshes are rigidly bound (every vertex weighted to exactly
    # one bone at weight 1.0), so there are no multi-bone seams that welding
    # would protect — and welding actively harms them: it fuses coincident
    # verts that belong to different bones (the eyelid pairs, which then can't
    # blink) and collapses same-bone hard-edge seam splits (the foot soles,
    # which then pinch). Welding is therefore suppressed on the DX8 path
    # regardless of the weld checkbox.
    state._is_ms_dx8 = is_ms_dx8

    # Create a collection named after the file (without extension) and link it
    # into the active scene so all imported objects land there together rather
    # than loose in the top-level scene collection.
    import_col_name = os.path.splitext(os.path.basename(filepath))[0]
    import_col = bpy.data.collections.new(import_col_name)
    context.scene.collection.children.link(import_col)
    state._import_collection = import_col

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
        # High-precision tick rate (e.g. some games uses 4800) —
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

    # Axis conversion (DirectX left-handed Y-up → Blender right-handed
    # Z-up) applies to ALL geometry, with or without an armature.
    # Previously this matrix was computed and stored on state._conv_mat
    # only inside build_armature, so frameless / skeleton-less files
    # (static props exported as a bare top-level Mesh, e.g. 7b.x) kept
    # the identity matrix and imported unrotated (lying on their side).
    # Compute it once here so every mesh-build path picks it up.
    conv_mat = _axis_matrix(axis_forward, axis_up)
    conv_mat = Matrix.Rotation(math.pi, 4, 'Z') @ conv_mat

    state._conv_mat = conv_mat

    if import_armature and frame_nodes:
        bind_poses = _collect_offset_matrices(root)
        ftm_globals = {}
        ftm_by_node = {}
        for fn in frame_nodes:
            ftm_globals.update(_collect_ftm_globals(fn))
            _collect_ftm_globals_by_node(fn, out=ftm_by_node)

        # SkinWeights matrixOffset is the inverse bind pose (the mesh
        # was authored against this), and FTM is the bone position at
        # the first animation keyframe. For some files and some .x
        # files these disagree; matrixOffset is the ground truth for
        # binding, FTM drives rest-pose orientation. Bones without
        # SkinWeights data fall back to FTM.
        n_with_bind = len(bind_poses)
        n_without_bind = len(ftm_globals) - n_with_bind

        state.build_armature(frame_nodes, context, bind_poses, ftm_globals,
                             conv_mat, ftm_by_node=ftm_by_node)

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

    for node in root.children:
        if node.kind == "Frame":
            try:
                state.import_frame_meshes(node, context)
            except Exception:
                # Per-top-level-frame failures should not abort the import
                # — other frames may still produce useful geometry.
                pass
        elif node.kind == "Mesh":
            # A Mesh can also sit directly at the top level with no
            # enclosing Frame. some exporters-exported static props do this:
            # the file is just the template block followed by a single
            # bare `Mesh { ... }` (e.g. 7b.x, 8.x). The loop used to
            # descend only into Frames, so these imported as an empty
            # collection. Build the mesh directly at identity world
            # transform; _build_mesh still applies state._conv_mat for
            # the axis conversion and global_scale.
            try:
                state._build_mesh(node, context, Matrix.Identity(4),
                                  import_col_name)
            except Exception:
                pass

    if import_animation:
        anim_sets = [n for n in root.children if n.kind == "AnimationSet"]
        for node in anim_sets:
            # Per-AnimationSet failures should not abort the import —
            # the skeleton and meshes are already in scene by this point
            # and a partial set of animation channels is more useful than
            # losing the whole import.
            try:
                state.import_animation_set(node, context)
            except Exception:
                pass

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
            # Hard backstop: Blender's scene.frame_start/end are signed 32-bit
            # and raise if exceeded. Even after tick scaling, clamp to a sane
            # range so a malformed or unrecognised tick scheme can never abort
            # the whole import (the animation data is already keyed by now).
            _FRAME_CAP = 1_048_576   # 2^20 frames (~9.7 hours at 30fps)
            fstart = max(-_FRAME_CAP, min(_FRAME_CAP, int(fstart)))
            fend   = max(fstart,      min(_FRAME_CAP, int(fend)))
            context.scene.frame_start = fstart
            context.scene.frame_end   = fend
            context.scene.frame_set(fstart)

    return {"FINISHED"}

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
    layout. high-precision biped exports use this instead of separate
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
    # Length check is mandatory: passing a wrongly-sized list to
    # normals_split_custom_set causes Blender to read past the
    # mesh's loop buffer in C, producing undefined memory in the
    # custom-normal cache. The crash doesn't fire here — it fires
    # later, the next time the viewport or edit-mode toggle reads
    # those normals — and shows up as a segfault deep inside
    # bm_mesh_loops_calc_normals. Skip the call entirely on
    # mismatch; geometry-derived normals are correct enough.
    n_loops = len(me.loops)
    if len(loop_normals) != n_loops:
        # Make sure auto-smooth is off and any stale custom normals
        # are dropped, so geometry-derived shading takes over cleanly.
        _clear_custom_split_normals(me)
        return

    # Validate mesh structure before the C call. Queen.xcache has 3
    # degenerate source faces ([0,0,0], [2,0,0], [6434,6432,6432])
    # which we filter at parse time, but bmesh.ops.weld_verts can
    # also produce post-weld degeneracies (zero-area tris when two
    # verts of a triangle weld together). The C-side
    # normals_split_custom_set walks the loop topology and segfaults
    # when it hits a polygon with collapsed corners. me.validate()
    # detects and repairs these in pure-Python land before the call.
    try:
        if hasattr(me, "validate"):
            me.validate(verbose=False, clean_customdata=False)
    except Exception:
        pass
    # validate() can re-arrange or remove loops, invalidating our
    # loop_normals list. Re-check length.
    n_loops_post = len(me.loops)
    if n_loops_post != n_loops:
        _clear_custom_split_normals(me)
        return

    # Sanitise every normal: must be finite and unit-length. Bad
    # values (NaN, inf, near-zero, denormalised) survive the size
    # check but get baked into Blender's loop-normal cache as
    # garbage, then crash the C solver later when edit-mode toggles
    # or the viewport rebuilds the mesh. Replace anything suspect
    # with (0, 0, 1) — geometry will still shade reasonably.
    safe_normals = []
    n_bad = 0
    for n in loop_normals:
        if n is None or len(n) != 3:
            safe_normals.append((0.0, 0.0, 1.0))
            n_bad += 1
            continue
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)):
            safe_normals.append((0.0, 0.0, 1.0))
            n_bad += 1
            continue
        mag = (nx*nx + ny*ny + nz*nz) ** 0.5
        if mag < 1e-6:
            safe_normals.append((0.0, 0.0, 1.0))
            n_bad += 1
            continue
        # Always renormalise — even mostly-unit normals can drift to
        # 0.9999 or 1.0001, and the C solver is strict.
        inv = 1.0 / mag
        safe_normals.append((nx * inv, ny * inv, nz * inv))

    # Clear any stale custom split normals BEFORE setting new ones.
    # If the mesh already has loop normals from a previous import
    # or modifier evaluation, they linger in a C-side cache that
    # `normals_split_custom_set` doesn't always overwrite cleanly.
    try:
        if hasattr(me, "free_normals_split"):
            me.free_normals_split()
    except Exception:
        pass

    if hasattr(me, "normals_split_custom_set"):
        # use_auto_smooth must be ON before normals_split_custom_set
        # is called — otherwise the data is stored but ignored by the
        # solver, and edit-mode toggle re-evaluates with stale data.
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True
        try:
            me.normals_split_custom_set(safe_normals)
        except Exception:
            # If even the sanitised set fails, drop custom normals
            # entirely. Smooth-shading from face normals is the
            # safe fallback.
            _clear_custom_split_normals(me)
            return
    else:

        attr = me.attributes.get("custom_normal")
        if attr is None:
            attr = me.attributes.new("custom_normal", "FLOAT_VECTOR", "CORNER")
        for i, n in enumerate(safe_normals):
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
        self._skel_root_nodes:   set    = set()
        self._bone_name_for_node: dict  = {}
        self._real_bone_for_skinname: dict = {}
        self._ftm_by_node:       dict   = {}
        self._bone_rebind:       dict   = {}
        self._ftm_globals:       dict   = {}
        self._refbake_done:  bool   = False
        self._global_parent_map: dict   = {}
        # Data for repairing matrix-key animation on bones inside a scaled
        # subtree (e.g. Colonel-X_L's facial bones under Head_bone's 0.107
        # scale, which Blender's scale-free edit-bones cannot represent).
        # Keyed by bone name -> {"W": scaled rest world, "par": parent name,
        # "par_W": parent's scaled rest world}. Empty for unscaled files.
        self._scaled_subtree_info: dict = {}
        self._has_scaled_subtree:  bool = False
        self._scaled_dx_by_frame:  dict = {}
        # Pre→post weld vertex map, set per mesh by the weld path so the
        # FRAME_TRANSFORM rebind can remap influence indices. None = identity.
        self._last_pre_to_post = None
        # Per-mesh source-text passthroughs (filled in by import_x);
        # default to empty so non-import code paths don't crash.
        self._passthrough_decldata: dict = {}
        self._passthrough_xskinheader: dict = {}
        # Blender collection that receives all objects from this import.
        # Set by import_x immediately after state construction.
        self._import_collection = None

    def parse_material(self, node, base_dir):
        name = node.name or "Material"
        # Per-import cache: if THIS import session already created a
        # material with this name (e.g. via a REF resolution earlier
        # in the same file), reuse it. We deliberately DON'T check
        # bpy.data.materials.get(name) here — that would reuse a
        # material from a PREVIOUS bulk-import iteration. common tools-exported
        # .x files all tend to name their materials "blinn1", "lambert1"
        # etc., so cross-file name collisions are the norm rather than
        # the exception. Reusing them would mean every file after the
        # first inherits the first file's textures.
        existing = self.materials.get(name)
        if existing:
            return existing
        # If a material with this name exists globally (e.g. from a
        # previous import in the same Blender session), let Blender
        # auto-suffix the new one to "blinn1.001". The shader graph
        # then gets THIS file's texture data, not the previous file's.
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

        # Fallback for xcache files that embed no texture paths (e.g. Queen.xcache):
        # derive candidate names from the material / asset name and let the directory
        # search below find a match.  We do NOT write these to _x_texture_filename
        # because they are guesses, not authoritative paths from the file.
        convention_tex_candidates = []
        if not original_tex_name and self.import_textures:
            mat_name = node.name or ""
            # e.g. "QueenMaterial1" -> "Queen", "QueenMaterial" -> "Queen"
            base = re.sub(r'Material\d*$', '', mat_name) or mat_name
            # Try common some games/a game engine texture naming patterns:
            #   Queen.dds, Queen_D.dds, QueenMaterial1.dds, Queen_1_D.dds, …
            convention_tex_candidates = [
                f"{base}.dds",
                f"{base}_D.dds",
                f"{mat_name}.dds",
                f"{mat_name}_D.dds",
            ]
            # Extract material index suffix if present (e.g. "1" from "QueenMaterial1")
            idx_match = re.search(r'(\d+)$', mat_name)
            if idx_match:
                idx = idx_match.group(1)
                convention_tex_candidates += [
                    f"{base}_{idx}.dds",
                    f"{base}_{idx}_D.dds",
                ]

        if self.import_textures and (original_tex_name or convention_tex_candidates):
            # Build the primary search list from the explicit path (if any).
            search_paths = []
            full_path = None  # canonical "intended" path for placeholder
            if original_tex_name:
                tex_path  = original_tex_name.replace("\\", os.sep).replace("/", os.sep)
                full_path = os.path.join(base_dir, tex_path)
                tex_basename = os.path.basename(tex_path)
                tex_subparts = tex_path.split(os.sep)

                # 1. Exact path from the file, anchored at base_dir
                search_paths.append(full_path)

                # 2. Progressively shorter tails of the engine path under
                #    base_dir (e.g. "Apple/Apple_D.dds", "Apple_D.dds")
                for cut in range(1, len(tex_subparts)):
                    search_paths.append(os.path.join(base_dir, *tex_subparts[cut:]))

                # 3. Common texture subfolders next to the xcache
                for sub in ("Textures", "textures", "Tex", "tex"):
                    search_paths.append(os.path.join(base_dir, sub, tex_basename))

                # 4. Walk up to 3 directory levels above base_dir
                ancestor = base_dir
                for _ in range(3):
                    parent = os.path.dirname(ancestor)
                    if not parent or parent == ancestor:
                        break
                    ancestor = parent
                    search_paths.append(os.path.join(ancestor, tex_path))
                    search_paths.append(os.path.join(ancestor, tex_basename))

                # 5. Cross-content search: sibling character/model folders and
                #    a depth-2 scan under any "Content" ancestor.  Handles the
                #    some games layout where textures live at
                #    Content/Models/Bugs/<CharName>/<CharName>_D.dds while the
                #    xcache is in Content/Characters/<CharName>/.
                search_paths.extend(
                    _build_cross_content_search_paths(tex_basename, base_dir)
                )

            # For each candidate location, also try alternate extensions.
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
                            # Record the successfully resolved path so the
                            # exporter can round-trip it.
                            if not original_tex_name:
                                mat["_x_texture_filename"] = os.path.basename(candidate)
                            break
                if img is not None:
                    break

            # Always create a TEX_IMAGE node and connect it to the BSDF so
            # the user can re-link manually if auto-search failed.
            if img is None and original_tex_name:
                # Only create a placeholder for explicitly-named textures —
                # convention guesses that don't exist shouldn't pollute the
                # image datablocks with junk 1×1 placeholders.
                if full_path is None:
                    full_path = os.path.join(base_dir, original_tex_name)
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

                # Connect texture alpha → material Alpha when the
                # "Use Diffuse Alpha" option is on. Many some games
                # textures encode transparency in the diffuse alpha
                # channel (leaf cutouts, eye highlights, etc.).
                # Connecting this is safe even for fully-opaque
                # textures — the alpha channel just reads as 1.0
                # everywhere and the material renders as if alpha
                # had never been hooked up. Also flip the material
                # blend mode to alpha-blend so Blender's viewport
                # actually respects the alpha; without this the
                # connection is silent and the artist might think
                # the option didn't work.
                if getattr(self, 'use_diffuse_alpha', False):
                    try:
                        mat.node_tree.links.new(
                            tex_img_node.outputs["Alpha"],
                            bsdf.inputs["Alpha"],
                        )
                        # Use 'HASHED' (dithered transparency) by
                        # default — it works well for both cutouts
                        # (leaves, fences) and soft-edged transparency
                        # (eye highlights) without needing the user
                        # to pick between CLIP and BLEND.
                        mat.blend_method = 'HASHED'
                    except Exception:
                        pass

        self.materials[name] = mat
        return mat

    def build_armature(self, frame_nodes, context, bind_poses, ftm_globals,
                       conv_mat, ftm_by_node=None):
        self._conv_mat = conv_mat
        self._bind_poses = bind_poses   # stashed for _build_mesh LBS pre-transform
        self._ftm_globals = ftm_globals  # name-keyed frame-hierarchy world matrices
        self._ftm_by_node = ftm_by_node or {}

        # Maps populated as bones are created so later stages can resolve
        # a frame node (or a SkinWeights bone name) to the ACTUAL Blender
        # bone name. Blender uniquifies colliding edit-bone names
        # ("Bone", "Bone.001", ...), so the name we asked for is not
        # always the name we got; vertex groups must use the real name
        # or the mesh part silently fails to follow its bone.
        self._bone_name_for_node = {}     # id(frame_node) -> real bone name
        self._real_bone_for_skinname = {} # SkinWeights name -> real bone name

        # Rest pose is ALWAYS the FrameTransformMatrix hierarchy, which
        # is what some exporters uses for its viewport (confirmed against its
        # Rest pose is always the FrameTransformMatrix hierarchy. The mesh is
        # rebound from its authored bind pose into that rest; the per-mesh bake
        # (_bake_ref_rest) is the primary path, and this name->matrix
        # rebind is retained as the fallback used only if a bake fails.
        self._bone_rebind = {}

        # Identify which top-level frames root the skeleton.
        #
        # A mesh-free top frame is always a skeleton root (the common
        # common tools/some game tools case: a bare "Bip01" hierarchy alongside
        # separate "MESH_*" wrapper frames).
        #
        # A mesh-CARRYING top frame is a skeleton root only if it also
        # has Frame children — i.e. it is a real bone that happens to
        # also hold geometry, and a bone hierarchy hangs off it. This is
        # the some editors "mesh-per-bone" convention (e.g. burger.x, whose
        # single top frame Burger_ROOTSHJnt holds the body mesh AND roots
        # all 68 leg/eye/tusk bones). Without the "has frame children"
        # clause that root was dropped and the entire armature came out
        # empty.
        #
        # A mesh-carrying top frame with NO frame children is left out:
        # that is a pure static-mesh wrapper (common tools "WatermelonGeoShape"
        # containers, some game tools "MESH_body"), not a bone.
        skel_roots = [
            f for f in frame_nodes
            if (not any(c.kind == "Mesh" for c in f.children))
            or any(c.kind == "Frame" for c in f.children)
        ]

        # Skeleton-root membership is tested by node identity, not name.
        # The old name-set collapsed every unnamed root to a single ""
        # entry, so files whose helper frames are unnamed (or share a
        # name) lost the ability to distinguish roots. Keep a name set
        # too for backward-compatible callers, but the identity set is
        # authoritative.
        self._skel_root_nodes = {id(f) for f in skel_roots}
        self._skel_root_names = {f.name for f in skel_roots}

        # Set of bone names actually referenced by SkinWeights anywhere
        # in the file. Used to decide whether an unnamed helper frame is
        # inert (safe to skip) or a real deforming bone.
        skin_ref_names = set()
        def _collect_skinrefs(n):
            if n.kind == "SkinWeights":
                bn = next((v for t, v in n.values if t == "STR"), None)
                if bn is None:
                    bn = next((v for t, v in n.values if t == "WORD"), None)
                if bn:
                    skin_ref_names.add(bn)
            for c in n.children:
                _collect_skinrefs(c)
        # frame_nodes are top-level frames; walk the whole tree from each.
        for _fr in frame_nodes:
            _collect_skinrefs(_fr)

        skip_inert = getattr(self, "skip_inert_helper_bones", True)

        def _is_inert_helper(fn):
            """True for frames that contribute nothing to the rig and only
            clutter it as 'Bone'/'Bone.NNN' entries: an UNNAMED frame that
            is a leaf (no Frame children), carries no Mesh, and is not
            referenced by any SkinWeights. These are biped nub/twist
            locators in files like aiko_L / SS-officer_L. Named frames are
            always kept (a name implies the author meant it as a joint),
            as are any frames with children, meshes, or skin references."""
            if not skip_inert:
                return False
            if fn.name:                      # named → keep
                return False
            if any(c.kind == "Frame" for c in fn.children):   # structural → keep
                return False
            if any(c.kind == "Mesh" for c in fn.children):    # holds mesh → keep
                return False
            if fn.name in skin_ref_names:    # skinned (empty-name match) → keep
                return False
            return True

        arm_data = bpy.data.armatures.new("Armature")
        arm_data.display_type = "STICK"
        arm_obj  = bpy.data.objects.new("Armature", arm_data)
        self._import_collection.objects.link(arm_obj)
        self.armature_obj = arm_obj
        self.created_objects.append(arm_obj)
        arm_obj.matrix_world = Matrix.Identity(4)

        context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")

        bone_count     = [0]
        fallback_count = [0]

        def add_bone(frame_node, parent_edit_bone):
            # Skip inert unnamed helper/locator frames so they don't
            # show up as spurious 'Bone.NNN' clutter. They have no
            # children, mesh, or skin influence, so dropping them
            # changes nothing about deformation. (Any frame children
            # they might have had are still visited with the current
            # parent, preserving hierarchy in the general case.)
            if _is_inert_helper(frame_node):
                for child in frame_node.children:
                    if child.kind == "Frame":
                        add_bone(child, parent_edit_bone)
                return
            name = frame_node.name or "Bone"
            parent_name = parent_edit_bone.name if parent_edit_bone else None
            # Record the full bone parent map (every bone, not just scaled
            # ones) so the scaled-subtree animation pre-pass can compose the
            # DX world up the chain past the scaled boundary (e.g. through
            # Bip01_Head and the spine above Head_bone).
            if not hasattr(self, "_global_parent_map"):
                self._global_parent_map = {}
            self._global_parent_map[name] = parent_name

            # Per-node FTM global (collision-free). Falls back to the
            # name-keyed map only if a node somehow isn't in the by-node
            # map (shouldn't happen, but keeps behaviour safe).
            ftm_glob_node = self._ftm_by_node.get(id(frame_node))
            if ftm_glob_node is None:
                ftm_glob_node = ftm_globals.get(name)

            # Rest = frame-hierarchy world (FTM). Fall back to the
            # inverse-bind only for a skinned bone with no FTM of its own.
            # Rest = frame-hierarchy world (FTM). Blender edit-bones cannot
            # store scale, so a scaled subtree (Colonel-X_L's face under
            # Head_bone's 0.107) lands at its SCALED position with scale
            # dropped — jaw at its true ~3u-from-head pivot. That near-the-mesh
            # pivot is what keeps the facial deform cohesive under LBS; the
            # inherited scale is reintroduced per-frame by the scaled-subtree
            # target-pose path in import_animation_set. Fall back to the
            # inverse-bind only for a skinned bone with no FTM of its own.
            if ftm_glob_node is not None:
                rest_mat = conv_mat @ ftm_glob_node
            elif name in bind_poses:
                rest_mat = conv_mat @ bind_poses[name]
                fallback_count[0] += 1
            else:
                rest_mat = Matrix.Identity(4)

            # Capture data for matrix-key animation repair on scaled subtrees.
            # A bone qualifies if its own rest world carries a non-unit scale
            # OR its (named) parent already qualified — scale propagates down.
            # Blender's edit-bones drop this scale, so without repair the lost
            # scale corrupts the pose chain — shrinking rigidly-skinned meshes
            # (the lower teeth, bound to Jaw_Bone) and fanning the face out.
            #
            # CRITICAL: DirectX .x permits many frames to share a name, and
            # these models contain ~31 differently-scaled UNNAMED ("") bind-
            # pose helper frames scattered across the whole skeleton (arms,
            # legs, spine). Because _scaled_subtree_info is name-keyed, an
            # unnamed frame would collapse all of them into one bogus entry
            # AND mark every real bone parented under one as "scaled",
            # corrupting the entire upper body during animation. Those helper
            # frames are never animation REF targets nor skin targets, so we
            # exclude unnamed frames entirely and only let scale propagate
            # through genuinely-named deform bones.
            _named = bool(name) and name.strip() != ""
            try:
                _wscale = rest_mat.to_scale()
                _is_scaled = _named and any(abs(c - 1.0) > 1e-3 for c in _wscale)
            except Exception:
                _is_scaled = False
            _par_scaled = (_named
                           and parent_name is not None
                           and parent_name in self._scaled_subtree_info)
            if _named and (_is_scaled or _par_scaled):
                # Parent's TRUE (scaled) rest world, needed at the subtree
                # boundary (e.g. Head_bone's parent Bip01_Head, outside the
                # scaled set). Pulled from FTM globals.
                _par_W = None
                if parent_name is not None:
                    _pg = ftm_globals.get(parent_name)
                    if _pg is not None:
                        _par_W = (conv_mat @ _pg)
                self._scaled_subtree_info[name] = {
                    "W":     rest_mat.copy(),
                    "par":   parent_name,
                    "par_W": _par_W,
                }
                self._has_scaled_subtree = True

            # Record the bind->FTM rebind as a fallback (used by _build_mesh
            # only when the per-mesh bake fails). Within epsilon of identity
            # for some exporters-native files, so they are untouched.
            if (ftm_glob_node is not None
                    and name in bind_poses):
                try:
                    R = (conv_mat @ ftm_glob_node) @ (conv_mat @ bind_poses[name]).inverted()
                    if not _is_near_identity(R):
                        self._bone_rebind[name] = R
                except ValueError:
                    pass

            eb = arm_data.edit_bones.new(name)
            # Blender may have uniquified the name (e.g. a second frame
            # also called "Bone" becomes "Bone.001"). Record the REAL
            # stored name keyed by node identity, and map the requested
            # SkinWeights name to the first real bone that claimed it, so
            # vertex groups bind to a bone that actually exists.
            real_name = eb.name
            self._bone_name_for_node[id(frame_node)] = real_name
            if name not in self._real_bone_for_skinname:
                self._real_bone_for_skinname[name] = real_name
            if self.global_scale != 1.0:
                scaled = rest_mat.copy()
                scaled.translation *= self.global_scale
                rest_mat = scaled

            eb.matrix = rest_mat
            if (eb.tail - eb.head).length < 1e-4:
                eb.tail = eb.head + rest_mat.to_3x3() @ Vector((0, 0.1, 0))

            # Workaround for Blender's mat3_to_vec_roll singular branch.
            #
            # When the bone Y axis (head→tail) is very close to world ±Y,
            # Blender's vec_roll_to_mat3_normalized branches on
            # theta_alt = nor.x² + nor.z² > (2.5e-4)² = 6.25e-8. The
            # standard and singular branches produce matrices differing by
            # ~90° around bone-local Y. In symmetric models (Shishkabob)
            # the bones in a symmetric set can sit on opposite sides of the
            # threshold due to floating-point noise — one bone gets rotated
            # 90° relative to its siblings even though we passed in
            # numerically equivalent rest matrices.
            #
            # Workaround: read back eb.matrix; if Blender's stored X axis
            # differs from the target, search roll offsets of ±90/180/270°
            # for the one that reproduces the target X. Uses Blender's own
            # reconstruction in the search loop, so it's robust to whichever
            # branch fires.
            try:
                target_x = rest_mat.to_3x3().col[0].normalized()
                actual_x = eb.matrix.to_3x3().col[0].normalized()
                if actual_x.dot(target_x) < 0.99:
                    base_roll = eb.roll
                    best_diff = float('inf')
                    best_roll = base_roll
                    # Candidates are k * 90° for k in {-1, 0, 1, 2}; this
                    # covers all four distinct roll offsets modulo 2π.
                    for delta in (0.0, math.pi/2, math.pi, -math.pi/2):
                        eb.roll = base_roll + delta
                        diff = (eb.matrix.to_3x3().col[0].normalized()
                                - target_x).length
                        if diff < best_diff:
                            best_diff = diff
                            best_roll = base_roll + delta
                    eb.roll = best_roll
            except Exception:
                # Leave the bone as Blender stored it on failure.
                pass

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
                # Per-mesh failures should not abort the entire import.
                # A skeleton-only result (or a partial set of meshes) is
                # more useful than nothing.
                try:
                    self._build_mesh(child, context, world_mat,
                                     frame_node.name, parent_frame=frame_node)
                except Exception:
                    pass
            elif child.kind == "Frame":
                self.import_frame_meshes(child, context, world_mat)

    def _bake_ref_rest(self, mesh_node, verts):
        """some exporters bake-and-reset: return verts moved to their true world
        rest position.

        Implements the verified recipe:
            v_baked = sum_i w_i * (W[b_i] @ O_source[b_i]) @ v
        with W[b] the frame-hierarchy world (Blender space, incl. global_scale)
        and O_source[b] the file's authored matrixOffset. Unweighted (or
        zero-total) verts are returned unchanged.

        `verts` is the already conv/scale-converted vertex list. We must build
        W[b] and O_source[b] in the SAME space, so both go through conv_mat and
        the global_scale translation factor used for `verts`.
        """
        sw_nodes = mesh_node.children_of("SkinWeights")
        if not sw_nodes:
            return verts

        conv = self._conv_mat
        gscale = float(getattr(self, "global_scale", 1.0))
        bind_poses = getattr(self, "_bind_poses", {}) or {}
        ftm_globals = getattr(self, "_ftm_globals", {}) or {}

        def to_blender(mat_dx):
            """conv @ M @ conv^-1, then apply global_scale to translation,
            matching how `verts` were transformed (point: (conv@v)*scale)."""
            try:
                m = conv @ mat_dx @ conv.inverted()
            except ValueError:
                return None
            if gscale != 1.0:
                m = m.copy()
                m.translation = m.translation * gscale
            return m

        # Build per-bone net bind transform  P[b] = W_blender[b] @ O_source_blender[b]
        # where O_source = (bind_poses[b])^-1 (bind_poses already stores
        # offset^-1 = the inverse-bind). W_blender = conv @ ftm_globals[b].
        #
        # NOTE on spaces: `verts` = (conv @ v_dx) * gscale. We need P to act on
        # these converted verts and yield converted world rest. Since both W and
        # O are conjugated by conv and share the gscale convention, P built from
        # to_blender(W_dx) @ to_blender(O_dx) operates correctly on the
        # converted verts. (Verified: identity bind -> P == I -> verts
        # unchanged.)
        net = {}
        for sw in sw_nodes:
            bn = next((v for t, v in sw.values if t == "STR"), None)
            if bn is None:
                bn = next((v for t, v in sw.values if t == "WORD"), None)
            if not bn:
                continue
            W_dx = ftm_globals.get(bn)
            if W_dx is None:
                continue
            # Use THIS mesh's OWN matrixOffset (the trailing 16 floats of its
            # SkinWeights), not the name-keyed global bind_poses. In DirectX a
            # bone's matrixOffset is authored per mesh — it maps each mesh's own
            # local space into the bone — so several meshes binding the same
            # bone legitimately carry DIFFERENT offsets. The global map keeps
            # only the last writer, so a mesh whose offset differed got baked
            # with another mesh's offset and was flung off (severe on
            # SS-officer's teeth_upper, ~8u; subtle on every model's head,
            # ~1-2u — the "head forward"). Fall back to the global inverse-bind
            # only if a mesh somehow lacks an inline offset.
            O_dx = None
            _swn = sw.nums()
            if _swn:
                _n = int(_swn[0])
                _mo = 1 + 2 * _n
                if len(_swn) >= _mo + 16:
                    O_dx = _mat4_from_list(_swn[_mo:_mo + 16])
            if O_dx is None:
                inv_bind = bind_poses.get(bn)   # = O_source^-1 (4x4 Matrix)
                if inv_bind is None:
                    continue
                try:
                    O_dx = inv_bind.inverted()  # = O_source (the authored offset)
                except ValueError:
                    continue
            Wb = to_blender(W_dx)
            Ob = to_blender(O_dx)
            if Wb is None or Ob is None:
                continue
            net[bn] = Wb @ Ob

        if not net:
            return verts

        # Per-vertex weighted accumulation of P[b] @ v.
        # Parse weights once: vert_index -> list[(bone, weight)]
        nv = len(verts)
        vw = [None] * nv
        for sw in sw_nodes:
            bn = next((v for t, v in sw.values if t == "STR"), None)
            if bn is None:
                bn = next((v for t, v in sw.values if t == "WORD"), None)
            if not bn or bn not in net:
                continue
            nums = sw.nums()
            if not nums:
                continue
            n = int(nums[0])
            if n <= 0 or len(nums) < 1 + 2 * n:
                continue
            for k in range(n):
                vi = int(nums[1 + k])
                w = float(nums[1 + n + k])
                if 0 <= vi < nv:
                    lst = vw[vi]
                    if lst is None:
                        lst = []
                        vw[vi] = lst
                    lst.append((bn, w))

        out = list(verts)
        for vi in range(nv):
            infl = vw[vi]
            if not infl:
                continue
            v = Vector((verts[vi][0], verts[vi][1], verts[vi][2]))
            acc = Vector((0.0, 0.0, 0.0))
            tot = 0.0
            for bn, w in infl:
                P = net.get(bn)
                if P is None:
                    continue
                acc = acc + (P @ v) * w
                tot += w
            if tot > 1e-9:
                acc = acc / tot
                out[vi] = (acc.x, acc.y, acc.z)

        # Mark that this mesh was some exporters-baked, so the downstream
        # _bone_rebind pass is skipped for it (the bake already placed verts).
        self._refbake_done = True
        return out

    def _build_mesh(self, mesh_node, context, world_mat, frame_name,
                    parent_frame=None):
        name_hint = mesh_node.name or frame_name or "Mesh"
        # Reset per-mesh reconstruction map (set only if this mesh references
        # vertices it never defined; consulted by the UV back-fill below).
        self._recon_src_vert = {}

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

        # Filter out degenerate faces: faces with fewer than 3 unique
        # vertex indices, or OOB indices. Queen.xcache has 3 such
        # faces (e.g. [0,0,0], [2,0,0]) that Blender's from_pydata
        # would silently keep as zero-area polygons. Later when
        # normals_split_custom_set is called the C-side reads through
        # those broken loops and segfaults. Tracking which source
        # face index each kept face came from lets us continue to
        # apply face-indexed metadata (MeshNormals per-face indices,
        # material per-face indices, face_to_submesh) without
        # off-by-one errors.
        # Some buggy exporters (e.g. a non-JT "DirectX Exporter") write a
        # vertex count that's SHORT of what the faces actually reference:
        # the faces cite indices [n_verts .. max_idx] for vertices that were
        # never emitted. The old behaviour dropped those faces, leaving
        # visible holes in the mesh (the burger's 45 trailing faces / 25
        # missing verts). Instead, reconstruct each missing vertex at the
        # centroid of the real vertices it shares a face with, so every face
        # survives and the hole is filled with geometry that sits in the
        # right local neighbourhood (verified to land inside the source
        # bounding box, no spikes to the origin). We only synthesize indices
        # in the contiguous run just past the end of the vertex array;
        # negative or absurd indices are still treated as corruption and
        # their faces dropped below. UVs/normals for the new verts are
        # back-filled from a co-face vertex so per-vertex data stays aligned.
        n_verts_total = len(verts)
        _max_face_idx = -1
        for face in faces:
            for vi in face:
                if vi > _max_face_idx:
                    _max_face_idx = vi
        _recon_start = n_verts_total
        _n_reconstructed = 0
        # Map each missing index -> list of valid co-face vertex indices.
        self._recon_src_vert = {}   # new vi -> a valid source vi (for UV/normal copy)
        if _max_face_idx >= n_verts_total:
            from collections import defaultdict as _dd
            _neighbors = _dd(list)
            for face in faces:
                missing = [vi for vi in face
                           if n_verts_total <= vi <= _max_face_idx]
                if not missing:
                    continue
                valid = [vi for vi in face if 0 <= vi < n_verts_total]
                for mvi in missing:
                    _neighbors[mvi].extend(valid)
            # Append reconstructed positions for the contiguous missing run.
            # Iterate in index order so verts.append() lands at the right id.
            for new_vi in range(n_verts_total, _max_face_idx + 1):
                src = _neighbors.get(new_vi)
                if src:
                    cx = sum(verts[s][0] for s in src) / len(src)
                    cy = sum(verts[s][1] for s in src) / len(src)
                    cz = sum(verts[s][2] for s in src) / len(src)
                    self._recon_src_vert[new_vi] = src[0]
                else:
                    # No valid co-face vertex to anchor to: fall back to the
                    # mesh centroid so the vert is at least inside the model.
                    if verts:
                        cx = sum(v[0] for v in verts) / len(verts)
                        cy = sum(v[1] for v in verts) / len(verts)
                        cz = sum(v[2] for v in verts) / len(verts)
                    else:
                        cx = cy = cz = 0.0
                    self._recon_src_vert[new_vi] = 0 if verts else None
                verts.append((cx, cy, cz))
                _n_reconstructed += 1
            n_verts_total = len(verts)
            if _n_reconstructed:
                import warnings as _w
                _w.warn(
                    f"{name_hint}: the source .x file references "
                    f"{_n_reconstructed} vertex(es) it never defined "
                    f"(declared {_recon_start}, faces use up to "
                    f"{_max_face_idx}). Reconstructed them from neighbouring "
                    f"geometry so all faces import without holes."
                )

        _kept_face_src_idx = []      # post-filter idx -> source face idx
        _filtered_faces = []
        n_degen_dropped = 0
        n_oob_dropped = 0
        for fsi, face in enumerate(faces):
            if len(face) < 3:
                n_degen_dropped += 1
                continue
            if len(set(face)) < 3:
                # Duplicate index within face — e.g. [0,0,0], [2,0,0].
                # Not a valid polygon.
                n_degen_dropped += 1
                continue
            # Out-of-range vertex index. Some real exports contain these
            # (e.g. a source mesh with 19 faces referencing verts
            # past the end of the vertex array) — passing them to
            # from_pydata corrupts the mesh or errors out, and the later
            # custom-normals C call can segfault on the broken loops. Drop
            # the offending faces and warn rather than failing the import.
            if any(vi < 0 or vi >= n_verts_total for vi in face):
                n_oob_dropped += 1
                continue
            _filtered_faces.append(face)
            _kept_face_src_idx.append(fsi)
        faces = _filtered_faces

        if n_oob_dropped:
            import warnings as _w
            _w.warn(
                f"{name_hint}: dropped {n_oob_dropped} face(s) with "
                f"out-of-range vertex indices (mesh has {n_verts_total} "
                f"vertices). The source .x file is malformed; the rest of "
                f"the mesh was imported normally."
            )

        me  = bpy.data.meshes.new(name_hint)
        obj = bpy.data.objects.new(name_hint, me)
        self._import_collection.objects.link(obj)
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
            # Stash the original mesh name so the exporter can look it up.
            obj["_x_mesh_name"] = _src_mesh_name
        # Stash the wrapping Frame's name too. In .x text format the
        # common tools export convention is to nest the Mesh inside a Frame
        # where the Frame is named e.g. "WatermelonGeo3" and the
        # Mesh inside is named "WatermelonGeo3Shape". On round-trip
        # the exporter needs both names: Frame name for the outer
        # wrapper, Mesh name for the inner block. Without this stash
        # the exporter falls back to the Blender object name (which
        # the importer set to the Mesh name) and the round-trip
        # produces a Frame named "WatermelonGeo3Shape" instead of
        # the original "WatermelonGeo3".
        if frame_name and frame_name != _src_mesh_name:
            obj["_x_frame_name"] = frame_name

        # If this mesh came from a multi-material .x split (parser-level
        # _split_x_mesh_by_material post-process), stash the split-group
        # provenance. The .x exporter uses these to re-merge the split
        # objects back into a single Mesh node with a multi-material
        # MeshMaterialList — matching the original .x file structure.
        # `split_source_mesh` is the name of the ORIGINAL pre-split
        # mesh (e.g. 'BoatGeoLowShape'). `split_group_idx` is this
        # object's position in the split (0 = first material's faces).
        node_meta = getattr(mesh_node, 'meta', None)
        if node_meta and 'split_source_mesh' in node_meta:
            obj["_x_split_source_mesh"] = node_meta['split_source_mesh']
            obj["_x_split_group_idx"]   = node_meta['split_group_idx']
            obj["_x_split_group_total"] = node_meta['split_group_total']

        # Reset the per-mesh some exporters-bake flag so it never leaks from a
        # previously-baked mesh to a later one that doesn't bake.
        self._refbake_done = False

        M = self._conv_mat
        s = self.global_scale
        if M != Matrix.Identity(4) or s != 1.0:
            verts = [((M @ Vector(v)) * s).to_tuple() for v in verts]

        # some exporters bake-and-reset bind (the only bind path). For a skinned
        # mesh, bake each vertex to its true world rest:
        #     v_baked = sum_i w_i * (W[b_i] @ O_source[b_i]) @ v
        # with W[b] the frame-hierarchy world (Blender space) and O_source[b]
        # the mesh's OWN authored matrixOffset. This resolves the
        # bind-vs-hierarchy mismatch once in vertex space, before the armature
        # binds anything; the skin offsets are then effectively W[b]^-1, so
        # standard skinning reproduces the baked rest and animates correctly.
        if (self.import_weights and self.armature_obj
                and mesh_node.children_of("SkinWeights")):
            self._refbake_done = False
            try:
                verts = self._bake_ref_rest(mesh_node, verts)
            except Exception:
                # On failure, fall back to authored verts (rebind path applies).
                self._refbake_done = False

        me.from_pydata(verts, [], faces)
        me.update()
        # Snapshot of source vert positions — used later as the weld
        # key, BEFORE any rebinding micro-displaces them. Without this
        # snapshot, bone-rebind moves verts at the same source XYZ
        # apart by ~1e-6, and the rounded-position weld key no longer
        # groups them.
        _source_positions = list(verts)

        _pending_loop_normals = None
        _pending_per_vi_normal = None
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

                # Build a per-(pre-vert, per-corner) normal table that
                # survives the weld. After weld we re-walk me.polygons
                # and look up the normal by the surviving vert's index.
                # NOTE: the file's face-normal-index block is sized to
                # the SOURCE face count (incl. degenerates). We pre-
                # parse all of it into a source-indexed array, then
                # extract entries for the post-filter face order via
                # _kept_face_src_idx so face_norm_indices stays
                # aligned with `faces`.
                _source_face_norm_indices = []
                _src_ni = ni
                for _fsi in range(fcount):
                    if _src_ni >= len(norms):
                        _source_face_norm_indices.append([])
                        continue
                    _src_cc = int(norms[_src_ni]); _src_ni += 1
                    _src_idxs = [int(norms[_src_ni + k]) if _src_ni + k < len(norms) else 0
                                 for k in range(_src_cc)]
                    _src_ni += _src_cc
                    _source_face_norm_indices.append(_src_idxs)
                ni = _src_ni
                face_norm_indices = [_source_face_norm_indices[fsi]
                                     for fsi in _kept_face_src_idx]
                # Pad missing entries with [0]*cc so downstream loops
                # don't crash on short lists.
                for k, face in enumerate(faces):
                    if len(face_norm_indices[k]) < len(face):
                        face_norm_indices[k] = (
                            list(face_norm_indices[k])
                            + [0] * (len(face) - len(face_norm_indices[k])))

                # Per-pre-vi normal lookup: average all the corner
                # normals that referenced this pre-vi. (For most verts
                # the corner normals are identical anyway; this just
                # picks something reasonable when they differ.)
                pre_vi_normal: dict = {}
                for face_idx, face in enumerate(faces):
                    for corner_idx, pre_vi in enumerate(face):
                        if corner_idx >= len(face_norm_indices[face_idx]):
                            continue
                        ni_val = face_norm_indices[face_idx][corner_idx]
                        if ni_val < len(normals_list):
                            pre_vi_normal.setdefault(pre_vi, normals_list[ni_val])
                _pending_per_vi_normal = pre_vi_normal
                # Keep the pre-weld loop_normals build for the
                # no-weld path (no armature). The post-weld path
                # rebuilds from _pending_per_vi_normal instead.
                loop_normals = []
                for poly, norm_idxs in zip(me.polygons, face_norm_indices):
                    for corner, _ in enumerate(poly.loop_indices):
                        ni_val = norm_idxs[corner] if corner < len(norm_idxs) else 0
                        loop_normals.append(
                            normals_list[ni_val] if ni_val < len(normals_list) else (0.0, 0.0, 1.0)
                        )
                _pending_loop_normals = loop_normals

        uv_node = mesh_node.child("MeshTextureCoords")
        if self.import_uvs and uv_node:
            uvnums  = uv_node.nums()
            uvcount = int(uvnums[0]) if uvnums else 0
            # Per Microsoft Learn MeshTextureCoords spec, nTextureCoords
            # MUST equal nVertices. Log a warning if they disagree,
            # then proceed with the smaller of the two to avoid
            # out-of-range UVs.
            uvs     = []
            ui      = 1
            for _ in range(uvcount):
                if ui + 2 > len(uvnums):
                    break
                uvs.append((uvnums[ui], 1.0 - uvnums[ui + 1]))
                ui += 2
            # If vertices were reconstructed (the source file referenced
            # verts it never defined), the UV array is short by exactly
            # those verts. Back-fill each reconstructed vert's UV from the
            # co-face source vertex we anchored its position to, so the
            # per-loop lookup below maps it correctly instead of skipping it.
            _recon_src = getattr(self, "_recon_src_vert", None)
            if _recon_src and uvs:
                _need = max(_recon_src) + 1
                while len(uvs) < _need:
                    new_vi = len(uvs)
                    src = _recon_src.get(new_vi)
                    if src is not None and src < len(uvs):
                        uvs.append(uvs[src])
                    else:
                        uvs.append((0.0, 0.0))
            uv_layer = me.uv_layers.new(name="UVMap")
            for poly in me.polygons:
                for loop_idx in poly.loop_indices:
                    vi = me.loops[loop_idx].vertex_index
                    if vi < len(uvs):
                        uv_layer.data[loop_idx].uv = uvs[vi]

        # high-precision biped exports often skip MeshNormals and
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
                # Per Microsoft Learn MeshMaterialList spec, nFaceIndexes
                # MUST equal the parent mesh's nFaces. Log a warning if
                # they disagree, then clamp to avoid mis-assigning
                # materials to non-existent faces.
                face_mat_indices = [int(mat_nums[2 + i])
                                    for i in range(min(face_mat_count, fcount))
                                    if 2 + i < len(mat_nums)]

                # If degenerate faces were filtered before from_pydata,
                # the surviving mesh polygons no longer match the
                # source face order 1:1. Re-extract the per-face
                # material indices in the surviving-face order.
                if len(_kept_face_src_idx) != fcount:
                    face_mat_indices = [
                        face_mat_indices[fsi]
                        if fsi < len(face_mat_indices) else 0
                        for fsi in _kept_face_src_idx
                    ]

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

        # Only take the skinned path when the mesh actually carries
        # SkinWeights. A mesh with none — e.g. the rigid mesh-per-bone
        # pieces of a some editors/JT export like burger.x — must fall to
        # the rigid-placement branch below, which positions it at its
        # frame's world transform and bone-parents it. Without the
        # SkinWeights check, building an armature (which now happens for
        # burger.x) diverted these meshes here, left them at the origin
        # with fallback weights, and bypassed the placement entirely.
        if self.import_weights and self.armature_obj and mesh_node.children_of("SkinWeights"):
            sw_nodes = mesh_node.children_of("SkinWeights")

            pre_weld_skin = []
            placeholder_bones = []  # Bones with SkinWeights but n=0
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
                # Empty SkinWeights placeholders: bone is declared as
                # part of the mesh's skin set but has zero painted
                # weights. These exist in some files (e.g. Watermelon's
                # BigFrontLeg / BigBackLeg Hip bones) to keep the
                # XSkinMeshHeader bone count consistent with the
                # engine's expectations. Track them so we can create
                # an empty vertex group later, which preserves the
                # placeholder for round-trip export.
                if influence_count == 0:
                    placeholder_bones.append(bone_name)
                    continue
                # Guard against truncated data: require enough values
                # for indices + weights + (offset matrix). Without this,
                # files with a corrupted influence_count would IndexError
                # and abort the whole import.
                needed = 1 + influence_count * 2
                if len(sw_nums) < needed:
                    influence_count = max(0, (len(sw_nums) - 1) // 2)
                    if influence_count == 0:
                        placeholder_bones.append(bone_name)
                        continue
                indices = [int(sw_nums[1 + i])             for i in range(influence_count)]
                weights = [sw_nums[1 + influence_count + i] for i in range(influence_count)]
                pre_weld_skin.append((bone_name, list(zip(indices, weights))))

            obj.matrix_world  = Matrix.Identity(4)
            obj.parent        = self.armature_obj
            arm_mod           = obj.modifiers.new("Armature", "ARMATURE")
            arm_mod.object    = self.armature_obj
            arm_mod.use_vertex_groups = True

            do_weld = getattr(self, "weld_duplicate_verts", False)

            # DX8 meshes are rigidly bound; welding only fuses eyelids and
            # pinches feet, never helps. Force it off for that path.
            if getattr(self, "_is_ms_dx8", False):
                do_weld = False

            if do_weld:
                # Optional: weld duplicate-position verts so bone-deformation
                # boundaries don't visibly separate when animated (the xcache
                # stores UV/normal seams + skin-set boundaries as separate
                # verts at the same XYZ). OFF by default — only matters when
                # animating, and weld blends weights that approximate skin
                # boundaries rather than preserving the file's exact authoring.
                self._weld_and_assign_skin(
                    obj, me, pre_weld_skin, _source_positions,
                    _pending_per_vi_normal, _pending_loop_normals)
            else:
                # Default: no weld. Assign skin weights per-vert directly,
                # exactly as authored. Per-loop normals from MeshNormals
                # get applied later by the unwelded-shading path below.
                self._assign_skin_no_weld(obj, pre_weld_skin)

            # Create empty vertex groups for placeholder bones. These
            # are bones declared as part of the skin set in the source
            # file but with zero painted weights (e.g. Watermelon's
            # BigFrontLeg/BigBackLeg Hip bones). Creating an empty VG
            # preserves the placeholder so a subsequent export emits
            # the same empty SkinWeights block, keeping the XSkinMesh
            # Header bone count consistent on round-trip.
            for bone_name in placeholder_bones:
                if bone_name not in obj.vertex_groups:
                    obj.vertex_groups.new(name=bone_name)


            # Apply vertex-position rebind AFTER weight assignment
            # (FRAME_TRANSFORM rest mode only). This moves the verts from
            # their authored bind-pose space into the FTM rest pose, the
            # transform some exporters applies on import — verified to
            # reproduce the reference exporter's SMD export to <0.001 units.
            #
            # CRITICAL: `pre_weld_skin` is indexed in PRE-weld vertex
            # space, but when welding is on `me.vertices` has the smaller
            # POST-weld count and a different index order. We therefore
            # remap each pre-weld influence index to its surviving
            # post-weld vertex via the map the welder built
            # (`self._last_pre_to_post`). Without this remap the rebind
            # was scattered onto the wrong verts, tearing the mesh — the
            # reason FRAME_TRANSFORM appeared broken with the default
            # weld-on setting. When welding is off the map is the
            # identity (None), so pre==post.
            # Skip the fallback per-bone rebind when the bake already placed
            # the verts at their true world rest (running both would double-
            # move the mesh). The rebind only applies if a bake failed.
            _baked = getattr(self, "_refbake_done", False)
            if self._bone_rebind and not _baked:
                pre_to_post     = getattr(self, "_last_pre_to_post", None)
                n_post          = len(me.vertices)
                vert_accum      = [None] * n_post
                vert_weight_sum = [0.0]  * n_post
                for bone_name, influences in pre_weld_skin:
                    rebind = self._bone_rebind.get(bone_name)
                    if rebind is None:
                        continue
                    for vi, w in influences:
                        post_vi = pre_to_post.get(vi) if pre_to_post is not None else vi
                        if post_vi is None or post_vi >= n_post:
                            continue
                        contrib = rebind * w
                        if vert_accum[post_vi] is None:
                            vert_accum[post_vi] = contrib
                        else:
                            vert_accum[post_vi] += contrib
                        vert_weight_sum[post_vi] += w

                # Verts with NO skinning influence (vert_weight_sum == 0)
                # would otherwise be skipped below and left stranded at
                # their authored bind-pose position while every weighted
                # vert around them is rebound into FTM rest space. On a
                # mesh where weighted and unweighted verts share faces
                # (e.g. the some editors burger's 268 unweighted verts) that
                # tears the surface into long spikes. some exporters avoids
                # this by binding stray verts to a fallback bone. We do the
                # equivalent: propagate a rebind outward from weighted
                # geometry by repeated neighbour-averaging, so each stray
                # vert ends up with the average rebind of its nearest
                # rebound neighbours and travels with its local surface.
                # Truly isolated unweighted islands (no path to any
                # weighted vert) fall back to the mesh's dominant rebind.
                _unweighted = [vi for vi in range(n_post)
                               if vert_weight_sum[vi] <= 0.0]
                if _unweighted and any(a is not None for a in vert_accum):
                    # Normalised rebind per resolved vert (weighted verts
                    # first; unweighted verts get filled in as we flood).
                    resolved_rebind = [None] * n_post
                    for vi in range(n_post):
                        if vert_weight_sum[vi] > 0.0 and vert_accum[vi] is not None:
                            resolved_rebind[vi] = (vert_accum[vi]
                                                   * (1.0 / vert_weight_sum[vi]))
                    # Adjacency only for the unweighted verts we must fill.
                    unresolved = set(_unweighted)
                    neigh = {vi: set() for vi in unresolved}
                    for poly in me.polygons:
                        vlist = poly.vertices
                        for a in vlist:
                            if a in neigh:
                                neigh[a].update(vlist)
                    # Dominant fallback: rebind of the highest-weight vert.
                    _best_vi = max(
                        (vi for vi in range(n_post) if vert_weight_sum[vi] > 0.0),
                        key=lambda vi: vert_weight_sum[vi], default=None)
                    _fallback = (resolved_rebind[_best_vi]
                                 if _best_vi is not None else None)
                    # Flood outward: each pass resolves verts adjacent to an
                    # already-resolved vert, averaging their rebinds. Repeat
                    # until no progress (bounded, so it always terminates).
                    while unresolved:
                        progressed = False
                        for vi in list(unresolved):
                            acc = None
                            cnt = 0
                            for nb in neigh[vi]:
                                if nb == vi:
                                    continue
                                rb = resolved_rebind[nb]
                                if rb is None:
                                    continue
                                if acc is None:
                                    acc = rb.copy()
                                else:
                                    acc += rb
                                cnt += 1
                            if acc is not None and cnt > 0:
                                resolved_rebind[vi] = acc * (1.0 / cnt)
                                vert_accum[vi] = resolved_rebind[vi]
                                vert_weight_sum[vi] = 1.0
                                unresolved.discard(vi)
                                progressed = True
                        if not progressed:
                            break
                    # Anything still unresolved is a fully-isolated island.
                    if unresolved and _fallback is not None:
                        for vi in unresolved:
                            vert_accum[vi] = _fallback.copy()
                            vert_weight_sum[vi] = 1.0

                for vi, accum in enumerate(vert_accum):
                    if accum is None or vert_weight_sum[vi] <= 0.0:
                        continue
                    v_old = me.vertices[vi].co
                    v_h   = Vector((v_old.x, v_old.y, v_old.z, 1.0))
                    v_new = accum @ v_h
                    ws    = vert_weight_sum[vi]
                    me.vertices[vi].co = Vector((v_new.x / ws,
                                                 v_new.y / ws,
                                                 v_new.z / ws))
                me.update()

            for poly in obj.data.polygons:
                poly.use_smooth = True
            obj.data.update()
            if getattr(self, "smooth_shade_from_faces", False):
                _clear_custom_split_normals(obj.data)
            elif _pending_loop_normals is not None and not do_weld:
                # Unwelded path: pre-built _pending_loop_normals is
                # already sized to match the mesh's loop count (no
                # weld changed the geometry).
                _apply_custom_normals(obj.data, _pending_loop_normals)
                if self.infer_sharps:
                    self._infer_sharps_from_normal_list(obj.data, _pending_loop_normals)
            elif (do_weld and _pending_per_vi_normal is not None
                    and self.import_normals):
                # WELDED path: the weld changed the loop topology, so the
                # pre-weld loop_normals no longer line up. Rebuild per-loop
                # normals from the per-PRE-vertex normal table, mapping each
                # post-weld loop's vertex back to a pre-weld vertex via the
                # weld map. Without this, the welded skinned mesh (the body)
                # gets Blender's auto-computed normals, which point INWARD on
                # the concave facial region (eye sockets, mouth) — the bright
                # "inverted normals" band seen in the face-orientation overlay.
                # The bake places verts correctly but never restored normals;
                # this closes that gap. (Rigid sub-meshes already restore
                # normals in their own branch.)
                try:
                    me2 = obj.data
                    pre_to_post = getattr(self, "_last_pre_to_post", None)
                    # Invert pre_to_post: post_vi -> a representative pre_vi.
                    post_to_pre = {}
                    if pre_to_post:
                        for pre_vi, post_vi in pre_to_post.items():
                            post_to_pre.setdefault(post_vi, pre_vi)
                    loop_normals2 = []
                    default_n = (0.0, 0.0, 1.0)
                    for poly in me2.polygons:
                        for li in poly.loop_indices:
                            vi_post = me2.loops[li].vertex_index
                            pre_vi = (post_to_pre.get(vi_post, vi_post)
                                      if post_to_pre else vi_post)
                            n = _pending_per_vi_normal.get(pre_vi)
                            loop_normals2.append(n if n is not None else default_n)
                    if loop_normals2:
                        _apply_custom_normals(me2, loop_normals2)
                        if self.infer_sharps:
                            self._infer_sharps_from_normal_list(me2, loop_normals2)
                except Exception:
                    # Leave Blender's computed normals if anything goes wrong;
                    # better a shading quirk than a failed import.
                    pass

        else:

            # No skin weights on this mesh. Two cases:
            # No skin weights on this mesh.
            #
            # IMPORTANT regression note: historically this branch set
            # matrix_world = Identity, and that is CORRECT for the many
            # files whose unskinned mesh verts are authored directly in
            # final world space (static props like ROOM.X's 130 tiles,
            # bulb.x, Bones.X, model.X). Those must keep Identity rest
            # placement or they double-transform and scatter. So we do
            # NOT change the resting vertex placement here.
            #
            # What we DO add: if there is an armature and this mesh sits
            # under a Frame that became a bone (e.g. AIKO's 7674-vert
            # body under MESH_body01, which has an FTM but no
            # SkinWeights), rigidly parent the object to that bone so it
            # FOLLOWS ANIMATION as a solid piece, using
            # matrix_parent_inverse to pin it at its current (Identity)
            # world position. This leaves the rest pose byte-identical
            # to the old behaviour (the mesh doesn't move at bind time)
            # but makes it track its bone when animated, instead of
            # being left behind while skinned parts move — which is the
            # "parts moving separately" symptom. For files with no
            # armature, behaviour is exactly as before: Identity.
            # Decide resting placement. Two authoring conventions exist
            # for unskinned meshes nested under a Frame, and they need
            # OPPOSITE handling:
            #
            #   (a) WORLD-space authored (ROOM.X tiles, bulb.x, AIKO's
            #       body): verts are already at their final world
            #       position. Must stay at Identity or they double-
            #       transform and scatter.
            #   (b) BONE-LOCAL authored (some editors "mesh-per-bone" rigid
            #       rigs like burger.x, exported via the JT DirectX
            #       importer): each mesh is centred on its bone's local
            #       origin and MUST be moved out to the frame's world
            #       transform, exactly as the DirectX spec and some exporters
            #       do. Without this every piece collapses onto the origin
            #       in a heap.
            #
            # We can't tell these apart from the block structure alone, so
            # we ask a geometric question that cleanly separates them:
            # does applying the frame's world transform move the mesh
            # centroid CLOSER to its bone than leaving it at Identity?
            #   * local-space mesh: centroid ~origin, far from the bone ->
            #     applying the transform snaps it onto the bone (closer).
            #   * world-space mesh: centroid already at the bone ->
            #     applying the transform would shove it away (farther).
            # For world-space files the test fails and we keep Identity,
            # so their behaviour is unchanged.
            placed_world = Matrix.Identity(4)
            ftm_node_glob = None
            if parent_frame is not None:
                ftm_node_glob = self._ftm_by_node.get(id(parent_frame))

            if ftm_node_glob is not None and len(obj.data.vertices):
                conv = self._conv_mat
                try:
                    frame_world = conv @ ftm_node_glob @ conv.inverted()
                except ValueError:
                    frame_world = None
                if frame_world is not None:
                    if self.global_scale != 1.0:
                        frame_world = frame_world.copy()
                        frame_world.translation *= self.global_scale
                    bone_t = frame_world.translation
                    n_v    = len(obj.data.vertices)
                    c = Vector((0.0, 0.0, 0.0))
                    for v in obj.data.vertices:
                        c += v.co
                    c /= n_v
                    c_applied = frame_world @ c
                    d_identity = (c - bone_t).length
                    d_applied  = (c_applied - bone_t).length
                    # Tiny epsilon so a mesh genuinely centred on its bone
                    # (d_identity ~ d_applied ~ 0, e.g. a world-space mesh
                    # whose bone happens to sit at the origin) is left at
                    # Identity rather than nudged.
                    if d_applied < d_identity - 1e-6:
                        placed_world = frame_world

            obj.matrix_world = placed_world

            # Stash the rest world the mesh-per-bone join will bake the piece
            # through. Default to the computed frame world; if the piece is
            # bone-parented we OVERRIDE this below with the bone's ACTUAL rest
            # matrix, so the baked geometry and the deforming bone use the same
            # matrix and cannot diverge (see the rigid_bone block).
            obj["_x_rest_world"] = [list(row) for row in placed_world]

            rigid_bone = None
            if (self.armature_obj is not None
                    and parent_frame is not None
                    and getattr(self, "rigid_parent_unskinned", True)):
                rigid_bone = self._bone_name_for_node.get(id(parent_frame))

            if rigid_bone is not None:
                bone = self.armature_obj.pose.bones.get(rigid_bone)
                if bone is not None:
                    obj.parent = self.armature_obj
                    obj.parent_type = "BONE"
                    obj.parent_bone = rigid_bone

                    # Pin the object at its resting world position so
                    # bone-parenting doesn't shift it at rest; the bone
                    # then drives it during animation.
                    try:
                        obj.matrix_parent_inverse = (
                            self.armature_obj.matrix_world @ bone.matrix
                        ).inverted()
                    except ValueError:
                        pass
                    obj.matrix_world = placed_world

            for poly in obj.data.polygons:
                poly.use_smooth = True
            obj.data.update()
            if getattr(self, "smooth_shade_from_faces", False):
                _clear_custom_split_normals(obj.data)
            elif _pending_loop_normals is not None:
                _apply_custom_normals(obj.data, _pending_loop_normals)
                if self.infer_sharps:
                    self._infer_sharps_from_normal_list(obj.data, _pending_loop_normals)

    def _assign_skin_no_weld(self, obj, pre_weld_skin):
        """Default path: assign skin weights per-vert, no welding.

        Each pre-vert keeps its exact file-authored bone weighting.
        Verts at the same XYZ position with different skin sets stay
        separate; the file's per-loop normals make this invisible at
        rest pose. The trade-off is that animation may show micro-
        gaps at skin-set boundaries, since duplicate-position verts
        weighted to different bones move with their respective bones.
        Acceptable for almost all viewing/editing tasks.
        """
        n_actual = len(obj.data.vertices)
        # No weld: pre-weld and post-weld vertex indices coincide, so the
        # FRAME_TRANSFORM rebind needs no remap (identity).
        self._last_pre_to_post = None
        # Batch per-bone: build {bone -> {vi -> weight}} then write
        # each VG exactly once.
        accum: dict = {}
        for bone_name, influences in pre_weld_skin:
            bone_map = accum.setdefault(bone_name, {})
            for vi, w in influences:
                if vi >= n_actual:
                    continue
                if not (0.0 <= w <= 100.0):
                    continue
                # Duplicate vi within one bone (Beffica's
                # LowerJaw_Aux has 45 of these): keep the LAST
                # weight, matching the engine's "last writer wins"
                # semantics that we already verified against the
                # dev .x.
                bone_map[vi] = w
        for bone_name, bone_map in accum.items():
            if not bone_map:
                continue
            vg = (obj.vertex_groups.get(bone_name)
                  or obj.vertex_groups.new(name=bone_name))
            for vi, w in bone_map.items():
                vg.add([vi], w, "REPLACE")

        # Per-vert normalization (some xcache files store
        # unnormalized weights — Honey's root + wing bones sum to 2.0
        # at body verts).
        rewrites = []
        for v in obj.data.vertices:
            if not v.groups:
                continue
            total = sum(g.weight for g in v.groups)
            if total <= 0.001:
                continue
            if 0.99 <= total <= 1.01:
                continue
            inv = 1.0 / total
            for g in v.groups:
                rewrites.append((v.index, g.group, g.weight * inv))
        for vi, group_idx, new_w in rewrites:
            obj.vertex_groups[group_idx].add([vi], new_w, "REPLACE")

        # Fallback for unweighted verts: assign them to the same
        # bone as their geometrically-nearest weighted vert. This
        # prevents verts with no skin data from staying glued to
        # world origin when the armature animates. The xcache
        # encoding sometimes drops weights for verts that appear
        # at UV/normal seam duplicates of weighted verts; without
        # this fallback those duplicates would tear away from the
        # mesh during posing.
        self._assign_fallback_weights(obj)


    def _assign_fallback_weights(self, obj):
        """For each vertex with no vertex-group weights, copy weights
        from the geometrically-nearest weighted vertex.

        Mirrors what Blender's "Skinning > Quick Fill" would do, but
        targeted: only fills empties, never overwrites authored data.
        Preserves local rigidity — unrigged geometry inside a sleeve
        moves with the sleeve, not with the armature root.

        Falls back to the first armature root bone (if any) for verts
        with no nearby weighted neighbor (isolated decorative geometry,
        free-floating verts).
        """
        n_verts = len(obj.data.vertices)
        if n_verts == 0:
            return

        # Find unweighted and weighted vert indices
        weighted = []
        unweighted = []
        for v in obj.data.vertices:
            if v.groups and any(g.weight > 0 for g in v.groups):
                weighted.append(v.index)
            else:
                unweighted.append(v.index)
        if not unweighted:
            return
        if not weighted:
            # No weighted verts at all — assign everything to skel
            # root if we have one.
            fallback_bone = self._pick_fallback_root_bone(obj)
            if fallback_bone is None:
                return
            vg = (obj.vertex_groups.get(fallback_bone)
                  or obj.vertex_groups.new(name=fallback_bone))
            for vi in unweighted:
                vg.add([vi], 1.0, "REPLACE")
            return

        # Build per-vert position arrays
        positions = [tuple(v.co) for v in obj.data.vertices]

        # For each unweighted vert, find nearest weighted vert.
        # Bucket by rounded XYZ for O(1) lookup on exact-position
        # duplicates (UV/normal seam duplicates — common case, often
        # 100% on clothing patches).
        pos_buckets: dict = {}
        for vi in weighted:
            p = positions[vi]
            k = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
            pos_buckets.setdefault(k, []).append(vi)

        # For truly-isolated unweighted verts, use a KD-tree
        # (mathutils.kdtree, ships with Blender) for log-time
        # nearest-neighbor instead of an O(n²) Python brute-force
        # scan. Queen.xcache has ~2000 truly-isolated verts; brute
        # force is ~12M Python iterations = many seconds.
        try:
            from mathutils import kdtree
            _kd = kdtree.KDTree(len(weighted))
            for local_idx, src_vi in enumerate(weighted):
                p = positions[src_vi]
                _kd.insert((p[0], p[1], p[2]), local_idx)
            _kd.balance()
        except Exception:
            _kd = None

        # Collect per-bone fallback assignments.
        #
        # Note: an earlier version of this function had a side-aware
        # filter that refused to assign left-side bones to right-side
        # verts. That was a band-aid for the real bug, which was in
        # the parser: it was reading "chunk-membership" entries as
        # blend weights, inflating per-vert weight sums to 2.0+ and
        # causing leg-influences-body deformation. With the parser
        # now correctly filtering on trailer == pad, the fallback
        # can safely copy weights from the geometrically-nearest
        # weighted vertex without side restrictions.
        per_bone_fallback: dict = {}
        n_exact = 0
        n_nearest = 0

        for vi in unweighted:
            p = positions[vi]
            k = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
            cands = pos_buckets.get(k)
            if cands:
                src_vi = cands[0]
                n_exact += 1
            elif _kd is not None:
                # Nearest weighted vert via KD-tree
                _co, _local_idx, _dist = _kd.find((p[0], p[1], p[2]))
                src_vi = weighted[_local_idx]
                n_nearest += 1
            else:
                # KD-tree unavailable; brute-force fallback
                best_vi = weighted[0]
                best_d2 = float('inf')
                px, py, pz = p[0], p[1], p[2]
                for wvi in weighted:
                    wx, wy, wz = positions[wvi]
                    dx = wx - px; dy = wy - py; dz = wz - pz
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 < best_d2:
                        best_d2 = d2
                        best_vi = wvi
                src_vi = best_vi
                n_nearest += 1
            # Copy weights from src_vi
            src_vert = obj.data.vertices[src_vi]
            for g in src_vert.groups:
                if g.weight <= 0:
                    continue
                group_idx = g.group
                # Resolve group name from index
                bone_name = obj.vertex_groups[group_idx].name
                per_bone_fallback.setdefault(bone_name, []).append((vi, g.weight))

        # Apply
        for bone_name, entries in per_bone_fallback.items():
            vg = (obj.vertex_groups.get(bone_name)
                  or obj.vertex_groups.new(name=bone_name))
            for vi, w in entries:
                vg.add([vi], w, "REPLACE")

        # Fallback-weighted verts need the same rebind applied to them
        # as the explicitly-weighted verts — otherwise the mesh splits
        # between two coordinate spaces (post-rebind FTM-rest vs
        # pre-rebind mesh source). Same accumulation logic as the
        # main rebind block in _build_mesh.
        rebind = getattr(self, '_bone_rebind', None) or {}
        if rebind:
            vi_to_accum = {}      # vi -> 4x4 accumulator
            vi_to_wsum  = {}      # vi -> total weight applied
            for bone_name, entries in per_bone_fallback.items():
                bone_rebind = rebind.get(bone_name)
                if bone_rebind is None:
                    continue
                for vi, w in entries:
                    if vi >= len(obj.data.vertices):
                        continue
                    contrib = bone_rebind * w
                    if vi in vi_to_accum:
                        vi_to_accum[vi] += contrib
                    else:
                        vi_to_accum[vi] = contrib
                    vi_to_wsum[vi] = vi_to_wsum.get(vi, 0.0) + w
            n_rebound = 0
            for vi, accum in vi_to_accum.items():
                ws = vi_to_wsum[vi]
                if ws <= 0:
                    continue
                v_old = obj.data.vertices[vi].co
                v_h = Vector((v_old.x, v_old.y, v_old.z, 1.0))
                v_new = accum @ v_h
                obj.data.vertices[vi].co = Vector((v_new.x / ws,
                                                    v_new.y / ws,
                                                    v_new.z / ws))
                n_rebound += 1

    def _pick_fallback_root_bone(self, obj):
        """Return a bone name to use as the catch-all for unweighted
        verts when no weighted verts exist on the mesh. Prefers a
        skeleton root if one is registered, else returns None.
        """
        roots = getattr(self, '_skel_root_names', None) or set()
        if roots:
            return sorted(roots)[0]
        # Last resort: find any armature bone
        arm = getattr(self, 'armature_obj', None)
        if arm and arm.data and arm.data.bones:
            return arm.data.bones[0].name
        return None

    def _weld_and_assign_skin(self, obj, me, pre_weld_skin,
                              source_positions, pending_per_vi_normal,
                              pending_loop_normals):
        """Optional path: weld duplicate-position verts and merge
        their per-bone weights. Triggered by the
        "Weld Duplicate Vertices" import option. Use when animating
        the mesh, to avoid visible gaps at skin-set boundaries.

        Welds purely by position; collapses 9424 verts → ~2370 for
        Beffica. Skin weights from all collapsed pre-verts are
        summed, then re-normalised per vert.
        """
        import bmesh as _bmesh

        # Source positions snapshot, taken before any bone-rebind
        # micro-displacements, ensures the weld key groups what was
        # ORIGINALLY a single XYZ.
        pre_weld_positions = list(source_positions)
        _pre_count = len(pre_weld_positions)

        # Weld key: position + full weight signature. Two verts merge
        # only if they sit at the same XYZ AND carry identical bone
        # influences. UV/normal seam duplicates have identical weights
        # so they collapse correctly; blendshape alternates (e.g. eyelid
        # verts weighted to UpperLidOpen vs UpperLidClosed at the same
        # XYZ) have different weight signatures and stay separate.
        EPS = 5

        vi_to_weights: dict = {}
        for bone_name, influences in pre_weld_skin:
            for vi, w in influences:
                vi_to_weights.setdefault(vi, []).append((bone_name, round(w, EPS)))
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

        # Build pre_to_post mapping
        keeper_pre_vis = sorted(min(vs) for vs in key_to_vis.values())
        keep_to_post = {keep: i for i, keep in enumerate(keeper_pre_vis)}
        pre_to_keep = {}
        for vs in key_to_vis.values():
            keep = min(vs)
            for vi in vs:
                pre_to_keep[vi] = keep
        pre_to_post = {vi: keep_to_post[pre_to_keep[vi]] for vi in range(_pre_count)}
        _post_count = len(keeper_pre_vis)

        # Apply weld via bmesh
        _bm = _bmesh.new()
        _bm.from_mesh(obj.data)
        _bm.verts.ensure_lookup_table()

        targetmap = {}
        for vs in key_to_vis.values():
            if len(vs) < 2:
                continue
            keep = min(vs)
            if keep >= len(_bm.verts):
                continue
            keep_v = _bm.verts[keep]
            for other in vs:
                if other != keep and other < len(_bm.verts):
                    targetmap[_bm.verts[other]] = keep_v

        if targetmap:
            try:
                _bmesh.ops.weld_verts(_bm, targetmap=targetmap)
            except Exception:
                _bm.free()
                raise
        _bm.verts.ensure_lookup_table()
        try:
            _bm.to_mesh(obj.data)
        except Exception:
            _bm.free()
            raise
        _bm.free()
        obj.data.update()

        # Rebuild pre_to_post from the post-weld mesh: bmesh weld may
        # reorder surviving verts, so we can't use the pre-weld sort
        # order. Match by (position + weight signature) to correctly
        # handle blendshape alternates (two post-verts at the same XYZ
        # but with different bone influences) — a position-only key
        # would clobber one bone's weights with another's.
        pos_to_post_vis: dict = {}   # pos_key -> list of post_vi
        for v in obj.data.vertices:
            pk = (round(v.co.x, EPS), round(v.co.y, EPS), round(v.co.z, EPS))
            pos_to_post_vis.setdefault(pk, []).append(v.index)

        # Build inverse: for each position that has multiple post-verts,
        # we need to know which pre-vert keeper maps to which post-vert.
        # Since the weld preserved the keeper (min pre-vi in each group),
        # and bmesh may have reindexed, we match by finding which
        # post-vi each keeper's position resolves to in order.
        # For positions with only one post-vert this is unambiguous.
        # For positions with multiple post-verts (blendshape alternates),
        # we assign them in keeper order (sorted ascending).
        keeper_for_pk: dict = {}   # pos_key -> [keeper_pre_vi, ...]
        for vis in key_to_vis.values():
            keep = min(vis)
            p = pre_weld_positions[keep]
            pk = (round(p[0], EPS), round(p[1], EPS), round(p[2], EPS))
            keeper_for_pk.setdefault(pk, []).append(keep)
        for pk in keeper_for_pk:
            keeper_for_pk[pk].sort()

        # Map each keeper pre-vi to its post-vi by position-slot index.
        keeper_to_post: dict = {}
        for pk, keepers in keeper_for_pk.items():
            post_vis_at_pk = sorted(pos_to_post_vis.get(pk, []))
            for i, keeper in enumerate(keepers):
                if i < len(post_vis_at_pk):
                    keeper_to_post[keeper] = post_vis_at_pk[i]

        pre_to_post = {}
        for vi in range(_pre_count):
            keeper = pre_to_keep.get(vi)
            if keeper is not None and keeper in keeper_to_post:
                pre_to_post[vi] = keeper_to_post[keeper]

        # Expose the mapping so the FRAME_TRANSFORM rebind (which works in
        # pre-weld index space) can land each influence on its surviving
        # post-weld vertex.
        self._last_pre_to_post = pre_to_post

        # Accumulate skin weights from all collapsed pre-verts.
        # We take the MAX weight per (bone, post_vi) instead of
        # summing — multiple skin layers in Queen.xcache encode the
        # same final weight redundantly, so summing inflates each
        # bone's influence by 5-10x and then per-vert normalization
        # crushes max-weights to ~0.15 across all bones. Taking
        # max preserves the original authored influence.
        # For 2-chunk files like Honey (where mesh1 and mesh2 are
        # disjoint vertex sets that the welder sometimes collapses
        # at the boundary), max is also correct: same-position
        # boundary verts each independently encode their full
        # influence; max picks the strongest of the two layers.
        n_actual = len(obj.data.vertices)
        accum: dict = {}   # bone_name -> {post_vi: max_weight}
        for bone_name, influences in pre_weld_skin:
            bone_map = accum.setdefault(bone_name, {})
            for pre_vi, w in influences:
                post_vi = pre_to_post.get(pre_vi)
                if post_vi is None or post_vi >= n_actual:
                    continue
                if not (w >= 0.0 and w <= 100.0):
                    continue
                if post_vi in bone_map:
                    if w > bone_map[post_vi]:
                        bone_map[post_vi] = w
                else:
                    bone_map[post_vi] = w
        for bone_name, bone_map in accum.items():
            if not bone_map:
                continue
            vg = (obj.vertex_groups.get(bone_name)
                  or obj.vertex_groups.new(name=bone_name))
            for post_vi, w in bone_map.items():
                if w > 100.0:
                    w = 100.0
                vg.add([post_vi], w, "REPLACE")

        # Per-vert normalization
        rewrites = []
        for v in obj.data.vertices:
            if not v.groups:
                continue
            total = sum(g.weight for g in v.groups)
            if total <= 0.001:
                continue
            if 0.99 <= total <= 1.01:
                continue
            inv = 1.0 / total
            for g in v.groups:
                rewrites.append((v.index, g.group, g.weight * inv))
        for vi, group_idx, new_w in rewrites:
            obj.vertex_groups[group_idx].add([vi], new_w, "REPLACE")

        # Fallback for unweighted post-weld verts (mirrors the
        # no-weld path). Verts left empty after dedup get weights
        # from their nearest weighted neighbor.
        self._assign_fallback_weights(obj)

        # Rebuild post-weld custom normals by averaging across
        # collapsed pre-verts. Uses the corrected pre_to_post built
        # from actual post-weld mesh positions (above).
        if pending_per_vi_normal is not None:
            post_vi_normals: dict = {}
            for pre_vi, post_vi in pre_to_post.items():
                n = pending_per_vi_normal.get(pre_vi)
                if n is None:
                    continue
                post_vi_normals.setdefault(post_vi, []).append(n)
            post_vi_to_normal: dict = {}
            for post_vi, ns in post_vi_normals.items():
                if len(ns) == 1:
                    post_vi_to_normal[post_vi] = ns[0]
                    continue
                sx = sum(n[0] for n in ns)
                sy = sum(n[1] for n in ns)
                sz = sum(n[2] for n in ns)
                mag = (sx*sx + sy*sy + sz*sz) ** 0.5
                if mag > 1e-9:
                    post_vi_to_normal[post_vi] = (sx/mag, sy/mag, sz/mag)
                else:
                    post_vi_to_normal[post_vi] = ns[0]
            post_loop_normals = []
            for poly in obj.data.polygons:
                for loop_idx in poly.loop_indices:
                    post_vi = obj.data.loops[loop_idx].vertex_index
                    n = post_vi_to_normal.get(post_vi)
                    post_loop_normals.append(n if n is not None else (0.0, 0.0, 1.0))
            _apply_custom_normals(obj.data, post_loop_normals)
            if self.infer_sharps:
                self._infer_sharps_from_normal_list(obj.data, post_loop_normals)
        elif pending_loop_normals is not None:
            # Fallback: file had no MeshNormals, only DeclData
            _apply_custom_normals(obj.data, pending_loop_normals)
            if self.infer_sharps:
                self._infer_sharps_from_normal_list(obj.data, pending_loop_normals)

    def _infer_sharps_from_normal_list(self, me, loop_normals):
        threshold_cos = math.cos(math.radians(self.sharp_angle_deg))

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

        # Drop file-authored loop normals so Blender smooth-shades across
        # faces, breaking only at the sharp edges we just marked.
        _clear_custom_split_normals(me)
        for poly in me.polygons:
            poly.use_smooth = True
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True
        me.update()

    def _build_scaled_target_poses(self, anim_set_node):
        """Populate self._scaled_dx_by_frame[frame_key][bone] = target_pose
        (a Blender-space 4x4 Matrix) for every scaled-subtree bone.

        target_pose = (W_anim @ W_rest^-1) @ strip_scale(W_rest)
        where W_anim is the bone's DX animated world (chain of animated locals,
        WITH scale) and W_rest is its DX rest world. Computed in DX space, then
        converted to Blender space by conv_mat. frame_key is round(frame_tick).
        """
        import mathutils
        conv = self._conv_mat

        # Need, per bone: its parent name, DX rest local, and DX animated locals
        # keyed by frame. Build the rest-local + parent maps from the frame
        # hierarchy (already walked once; rebuild cheaply here from ftm data is
        # not enough since we need LOCALS, so walk the anim set for keys and use
        # stored rest worlds to recover locals where a bone is unanimated).
        scaled = getattr(self, "_scaled_subtree_info", {})
        if not scaled:
            return

        # Collect raw animated local matrices per bone from the AnimationSet.
        # Only matrix keys (type 3/4) carry a full local transform; TRS-animated
        # bones are uncommon in this chain, and if present we fall back to their
        # static rest local (captured below).
        anim_local = {}   # bone -> {frame_key: Matrix (DX local)}
        for anim_node in anim_set_node.children_of("Animation"):
            ref = anim_node.child("REF")
            if not ref:
                continue
            bn = next((v for t, v in ref.values if t in ("WORD", "STR")), None)
            if not bn:
                continue
            for kn in anim_node.children_of("AnimationKey"):
                nums = kn.nums()
                if len(nums) < 2:
                    continue
                kt = int(nums[0])
                if kt not in (3, 4):
                    continue
                nk = int(nums[1])
                i = 2
                fk_map = anim_local.setdefault(bn, {})
                cnt = 0
                while i + 1 < len(nums) and cnt < nk:
                    ftick = nums[i]; i += 1
                    i += 1  # skip nvals
                    if i + 16 > len(nums):
                        break
                    M = _mat4_from_list(nums[i:i + 16])
                    i += 16
                    fk_map[round(float(ftick) * self.tick_scale)] = M
                    cnt += 1

        # DX rest locals & parent map from the frame hierarchy. We reconstruct
        # them from the stored Blender-space rest worlds W (= conv @ W_dx):
        #   W_dx = conv^-1 @ W ;  local_dx = parent_W_dx^-1 @ W_dx
        # par_W (parent's Blender rest world) is stored for boundary bones; for
        # internal scaled bones the parent is also scaled and present in `scaled`.
        def W_dx_of(bone):
            info = scaled.get(bone)
            if info and info.get("W") is not None:
                return conv.inverted() @ info["W"]
            return None

        def parentname(bone):
            info = scaled.get(bone)
            return info.get("par") if info else None

        def parent_W_dx(bone):
            info = scaled.get(bone)
            if not info:
                return None
            pn = info.get("par")
            if pn in scaled and scaled[pn].get("W") is not None:
                return conv.inverted() @ scaled[pn]["W"]
            if info.get("par_W") is not None:
                return conv.inverted() @ info["par_W"]
            return None

        # rest DX local per scaled bone
        rest_local = {}
        for bn in scaled:
            Wb = W_dx_of(bn)
            Wp = parent_W_dx(bn)
            if Wb is None:
                continue
            rest_local[bn] = (Wp.inverted() @ Wb) if Wp is not None else Wb

        # Gather all frame keys that appear in any scaled-chain animation, plus
        # frame 0. Bones up the chain ABOVE the scaled set (e.g. Bip01_Head and
        # spine) also animate; we need their animated locals too. Collect those
        # from the same anim_local map (any bone, not just scaled).
        frame_keys = set([0])
        for bn, fk in anim_local.items():
            frame_keys.update(fk.keys())

        # Chain to root for a bone, using the FULL frame hierarchy parent map.
        # scaled bones know their parent; above the scaled set we rely on the
        # ftm parent recorded during build. Rebuild a global parent map:
        gparent = getattr(self, "_global_parent_map", None)
        if gparent is None:
            gparent = {}
        # rest world (DX) for ANY bone, for ancestors above the scaled set:
        # use ftm_globals (name-keyed DX worlds).
        ftm_globals = getattr(self, "_ftm_globals", {}) or {}

        def dx_world_anim(bone, fk):
            """DX animated world by composing animated locals up the chain.
            Uses anim_local where present, else the bone's static rest local."""
            # Build chain root->bone via parent map (scaled first, then global).
            chain = []
            seen = set()
            b = bone
            while b is not None and b not in seen and len(chain) < 200:
                chain.append(b); seen.add(b)
                b = parentname(b) if b in scaled else gparent.get(b)
            chain.reverse()
            W = mathutils.Matrix.Identity(4)
            for b in chain:
                al = anim_local.get(b, {})
                if fk in al:
                    L = al[fk]
                elif b in rest_local:
                    L = rest_local[b]
                elif b in ftm_globals:
                    # recover local from world/parent-world
                    pb = parentname(b) if b in scaled else gparent.get(b)
                    Wb = ftm_globals[b]
                    Wp = ftm_globals.get(pb) if pb else None
                    L = (Wp.inverted() @ Wb) if Wp is not None else Wb
                else:
                    L = mathutils.Matrix.Identity(4)
                W = W @ L
            return W

        def strip_scale(M):
            t = M.to_translation()
            q = M.to_quaternion()
            R = q.to_matrix().to_4x4()
            R.translation = t
            return R

        # Actual Blender rest per scaled bone, read back from the armature.
        # This is what matters, not the rest we *asked* for: Blender's roll
        # handling can store a different roll for one bone of a symmetric pair
        # (near-zero-length facial bones whose Y axis sits near world ±Y), and
        # the basis below uses this stored matrix_local for `rel`. Building the
        # target from the SAME stored matrix_local makes the basis collapse to
        # identity at the rest frame (so the mesh is undeformed at rest), and
        # because the deform reduces to (delta @ v) the stored roll cancels and
        # the result stays exact. Using the *intended* rest here instead was
        # what dented L_brow_inner / L_cheek / L_brow_outer (and the nubs).
        arm = getattr(self, "armature_obj", None)
        actual_rest = {}
        if arm is not None:
            for bn in scaled:
                pbn = arm.pose.bones.get(bn)
                if pbn is not None:
                    actual_rest[bn] = pbn.bone.matrix_local.copy()

        for fk in frame_keys:
            frame_entry = self._scaled_dx_by_frame.setdefault(fk, {})
            for bn in scaled:
                Wb_rest = W_dx_of(bn)
                if Wb_rest is None:
                    continue
                W_anim = dx_world_anim(bn, fk)
                try:
                    delta = W_anim @ Wb_rest.inverted()   # scale cancels
                    # Applied on the bone's ACTUAL Blender rest (post roll /
                    # zero-length processing), not the intended one.
                    ml = actual_rest.get(bn)
                    if ml is None:
                        ml = strip_scale(scaled[bn]["W"])
                    target = (conv @ delta @ conv.inverted()) @ ml
                    frame_entry[bn] = target
                except (ValueError, ZeroDivisionError):
                    continue

        # Boundary parents: an unscaled bone that is the parent of a scaled
        # bone (e.g. Bip01_Head above the face). It animates by the standard
        # path, so its pose at frame fk is conv @ (its animated DX world).
        # Cache that here, in the SAME space as the scaled targets, so
        # _scaled_subtree_basis always resolves parent_target from the cache.
        # Previously the boundary fell through to reading the live pose bone,
        # which is NOT set to frame fk while F-curves are being built — that
        # stale read was the source of the frame 1-15 facial blow-up.
        boundary_parents = set()
        for bn, info in scaled.items():
            pn = info.get("par")
            if pn and pn not in scaled:
                boundary_parents.add(pn)
        for fk in frame_keys:
            frame_entry = self._scaled_dx_by_frame.setdefault(fk, {})
            for bp in boundary_parents:
                if bp in frame_entry:
                    continue
                try:
                    frame_entry[bp] = conv @ dx_world_anim(bp, fk)
                except (ValueError, ZeroDivisionError):
                    continue

    def _scaled_subtree_basis(self, bone_name, dx_local, frame_tick):
        """Corrected, frame-constant matrix_basis for a bone whose rest world
        carries an inherited non-unit scale (Colonel-X_L's facial rig hangs
        off Head_bone, whose matrixOffset encodes a 0.107 uniform scale).

        Diagnostic dumps from live Blender proved two things that simplify
        this enormously:

          1. These facial bones do NOT animate independently — every one has
             a constant local transform across all frames. They move only by
             rigidly riding the animated head/neck/spine chain above them.
             So there is no per-frame facial animation to reconstruct; the
             correct behaviour is a single constant basis.

          2. Blender drops the inherited 0.107 scale from the scale-free
             edit-bone rests, but the pose chain still applies it once at the
             Head_bone -> facial boundary, shrinking the facial offsets to
             ~1/9th and fanning the mesh out. The fix is to cancel that one
             inherited scale so each facial bone rides its parent with its
             true (scale-free) rest offset.

        The verified basis (matches live-Blender ground truth to 1e-4 at every
        sampled frame) is:

            local = rest[parent]^-1 @ rest[bone]            (scale-free)
            basis = local^-1 @ Sinv @ local

        where Sinv cancels the parent's *pose* scale. That scale is the
        inherited 0.107 only for the direct children of the scale-carrying
        bone (their parent is the boundary bone); once a bone is corrected it
        rides scale-free, so its own children see a unit-scale parent and need
        Sinv = identity. Returns None on any problem so the caller falls back
        to the standard per-track path.

        (dx_local and frame_tick are accepted for call-site compatibility but
        unused: the correction is frame-independent because the bones are
        static.)
        """
        info = self._scaled_subtree_info.get(bone_name)
        if info is None:
            return None
        arm = self.armature_obj
        if arm is None:
            return None
        # Use the per-frame target pose computed by _build_scaled_target_poses.
        # target_pose[b] = (W_anim_dx[b] @ W_rest_dx[b]^-1) @ strip(W_rest_bl[b])
        # is the world pose this bone must have so that the baked (scale-free)
        # mesh deforms exactly as DirectX intends — with the 0.107 parent scale
        # correctly applied to the local offsets (fixes the flung jaw).
        #
        # Blender builds pose_world = parent_pose @ rel @ basis, where
        #   rel = strip(W_parent_bl)^-1 @ strip(W_bone_bl).
        # So the local basis we must emit is:
        #   basis = rel^-1 @ parent_target^-1 @ target
        # which telescopes correctly because the parent's pose IS parent_target
        # (every scaled bone uses this scheme; the boundary parent above the
        # scaled set lands at strip(its DX world), which equals parent_target
        # for an unscaled bone).
        fk = round(float(frame_tick) * self.tick_scale) if frame_tick is not None else 0
        cache = getattr(self, "_scaled_dx_by_frame", None)
        if not cache:
            return None
        frame_entry = cache.get(fk)
        if frame_entry is None:
            # nearest available frame key (animation sampled at sparse ticks)
            if cache:
                fk = min(cache.keys(), key=lambda k: abs(k - fk))
                frame_entry = cache.get(fk)
        if not frame_entry:
            return None
        target = frame_entry.get(bone_name)
        if target is None:
            return None
        try:
            pb = arm.pose.bones.get(bone_name)
            if pb is None or pb.parent is None:
                return None
            par_name = info.get("par")
            # Parent's target pose: another scaled bone's cached target, or the
            # boundary parent's scale-free rest world (strip of its Blender W).
            parent_target = frame_entry.get(par_name)
            if parent_target is None:
                # boundary: parent is above the scaled set. Its pose at this
                # frame is its own animated world (scale-free). Reconstruct from
                # the live pose bone, which the standard path has already set.
                ppb = pb.parent
                parent_target = ppb.matrix.copy()
            rest_b   = pb.bone.matrix_local
            rest_par = pb.parent.bone.matrix_local
            rel      = rest_par.inverted() @ rest_b
            basis    = rel.inverted() @ parent_target.inverted() @ target
            return basis
        except (ValueError, ZeroDivisionError):
            return None
        except Exception:
            return None

    def import_animation_set(self, anim_set_node, context):
        if not self.armature_obj:
            return

        arm_obj = self.armature_obj
        scene   = context.scene

        target_fps = int(round(self.ticks_per_second))
        if scene.render.fps != target_fps:
            scene.render.fps = target_fps

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

        # Pre-pass: build per-frame TARGET POSE for every scaled-subtree bone.
        # The facial bones (jaw/mouth/eyes) hang off Head_bone, which carries a
        # 0.107 world scale. Their animation keys are LOCAL and authored to be
        # multiplied by that parent scale. Blender strips scale from rest bones,
        # so the naive basis (rest^-1 @ parent_rest @ dx_local) applies the
        # local offset UNSCALED -> the jaw flies ~9.3x too far when animated.
        #
        # Fix (verified to 0.00000 against the source DX chain + some exporters SMD):
        # compose each scaled bone's DX animated WORLD down the chain WITH scale
        # present, form delta = W_anim @ W_rest^-1 (scale cancels, det +1), and
        # target_pose = delta @ strip(W_rest). _scaled_subtree_basis then turns
        # this into the local matrix_basis Blender needs. Caching it here keeps
        # the work order-independent (the per-bone anim loop may not be in
        # hierarchy order).
        self._scaled_dx_by_frame = {}
        try:
            self._build_scaled_target_poses(anim_set_node)
        except Exception:
            # On any failure, leave the cache empty; _scaled_subtree_basis then
            # returns None and the standard path runs (no worse than before).
            self._scaled_dx_by_frame = {}

        for _anim_idx, anim_node in enumerate(anim_nodes):
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
                    and 2 in _tracks_by_type
                    and 3 not in _tracks_by_type
                    and 4 not in _tracks_by_type):
                _synth = _compose_type4_from_trs(
                    _tracks_by_type[0], _tracks_by_type[1], _tracks_by_type[2])
                # For a bone inside a scaled subtree, the per-track F-curve
                # path cannot apply the scale-inheritance correction, so it
                # would mis-deform exactly like the matrix path did before the
                # fix. Force a composed matrix track (resampling if the TRS
                # tracks have different densities) so the bone flows through
                # the corrected _scaled_subtree_basis. Unscaled bones keep the
                # plain per-track path untouched.
                if (_synth is None
                        and self._has_scaled_subtree
                        and bone_name in self._scaled_subtree_info):
                    _synth = _compose_type4_from_trs_resampled(
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

                # Compute the bone's local rest rotation/translation. These
                # are wrapped in try/except because a degenerate rest pose
                # (singular matrix on an imported armature) would otherwise
                # raise and abort the entire animation channel for this
                # bone. Falling back to identity gives no per-bone animation
                # but keeps the rest of the import working.
                if key_type == 0:
                    pb = pose_bone
                    try:
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
                    except (ValueError, ZeroDivisionError):
                        local_rest_q = Quaternion((1.0, 0.0, 0.0, 0.0))

                if key_type == 2:
                    pb = pose_bone
                    try:
                        if pb.parent:
                            _t2_local_rest = (pb.parent.bone.matrix_local.inverted()
                                              @ pb.bone.matrix_local)
                            _t2_lock = self.lock_leaf_translation and not pb.children
                        else:
                            _t2_local_rest = pb.bone.matrix_local
                            _t2_lock = self.lock_root_translation
                        _t2_rest_head    = _t2_local_rest.to_translation()
                        _t2_rest_rot_inv = _t2_local_rest.to_3x3().inverted()
                    except (ValueError, ZeroDivisionError):
                        _t2_local_rest   = Matrix.Identity(4)
                        _t2_lock         = False
                        _t2_rest_head    = Vector((0.0, 0.0, 0.0))
                        _t2_rest_rot_inv = Matrix.Identity(3)
                    _t2_conv3        = self._conv_mat.to_3x3()

                chan_data: dict = {}

                prev_pose_q = None

                for frame_tick, vals in all_key_vals:
                    # Scale file ticks to Blender frame numbers. tick_scale
                    # is normally 1.0; for files with very high tick rates
                    # (e.g. high-precision's 4800/sec) it normalizes the timeline to a
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

                            for ci, v in enumerate(pose_q):
                                chan_data.setdefault(ci, []).append((frame, v))

                        elif key_type == 1:
                            for ci, v in enumerate(vals[:3]):
                                chan_data.setdefault(ci, []).append((frame, v))

                        elif key_type == 2:
                            if _t2_lock:
                                loc = Vector((0.0, 0.0, 0.0))
                            elif pose_bone.parent:
                                anim_t = (Vector((vals[0], vals[1], vals[2]))
                                          * self.global_scale)
                                loc = _t2_rest_rot_inv @ (anim_t - _t2_rest_head)
                            else:
                                anim_t = _t2_conv3 @ (
                                    Vector((vals[0], vals[1], vals[2])) * self.global_scale)
                                loc = _t2_rest_rot_inv @ (anim_t - _t2_rest_head)

                            for ci, v in enumerate(loc):
                                chan_data.setdefault(ci, []).append((frame, v))

                        elif key_type in (3, 4):
                            # Matrix key per spec (3) or engine
                            # extension (4). Either way the value is
                            # a flat 16-float 4x4 transform.

                            # Matrix key per spec (3) or engine extension (4):
                            # a flat 16-float 4x4 local transform.
                            dx_local = _mat4_from_list(vals)
                            # The rest bones (and mesh) may have been scaled by
                            # global_scale. The animation key's translation is
                            # in the file's native (unscaled) units, so scale it
                            # to match — otherwise the animated bone spacing is
                            # 1/scale of the rest skeleton and the armature
                            # collapses small during playback while the rest
                            # pose looks right.
                            if self.global_scale != 1.0:
                                dx_local = dx_local.copy()
                                _t = dx_local.translation * self.global_scale
                                dx_local.translation = _t
                            # Scaled subtree (e.g. the face under Head_bone's
                            # 0.107 scale): the rest bone is scale-free at its
                            # SCALED position, so the animated local must be
                            # turned into a per-frame target pose that recreates
                            # the scaled delta about that near pivot. Verified
                            # to keep the facial mesh cohesive under LBS (no
                            # spike) across frames 1-15. Unscaled bones take the
                            # standard basis below unchanged.
                            _corr = None
                            if (self._has_scaled_subtree
                                    and bone_name in self._scaled_subtree_info):
                                _corr = self._scaled_subtree_basis(
                                    bone_name, dx_local, frame_tick)
                            if _corr is not None:
                                matrix_basis = _corr
                            elif pose_bone.parent:
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

                    except Exception:
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
                elif key_type in (3, 4):
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