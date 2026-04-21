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
import bpy
import bmesh
from mathutils import Matrix, Vector, Quaternion

from .xlog import XLog

log = XLog("exporter")


# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────
def _mat4_to_dx(mat):
    """Format a 4x4 Blender Matrix as a comma-separated DX matrix string.

    DX stores row-major but written column-by-column when transposed, i.e.
    the flat list is m00,m10,m20,m30, m01,m11,m21,m31, ... matching how
    the importer parses the 16-value list.
    """
    t = mat.transposed()
    return ",".join(f"{t[r][c]:.6f}" for r in range(4) for c in range(4))


def _axis_matrix(axis_forward, axis_up):
    """Build the Blender-to-DX axis conversion matrix.

    The importer uses _axis_matrix('-Z', 'Y') = [[1,0,0],[0,0,-1],[0,1,0]] as
    DX-to-Blender. The exporter needs the inverse: Blender-to-DX.
    """
    import numpy as np
    _AXES = {'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0),'Z':(0,0,1),'-Z':(0,0,-1)}
    fwd = np.array(_AXES[axis_forward], float)
    upv = np.array(_AXES[axis_up],      float)
    rgt = np.cross(fwd, upv)
    # Importer builds:  M3 = B @ inv(F)  where B=[X,Z,Y], F=[right,up,forward]
    # So export (inverse) is: inv(M3) = F @ inv(B)
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
    """Return a list of loop normals, compatible with Blender <=4.0 and 4.1+.

    Blender 4.1 removed calc_normals_split() and added me.corner_normals.
    """
    if hasattr(me, "corner_normals") and len(me.corner_normals) > 0:
        # Blender 4.1+: the corner_normals attribute is auto-computed.
        return [tuple(cn.vector) for cn in me.corner_normals]
    if hasattr(me, "calc_normals_split"):
        # Blender <=4.0
        me.calc_normals_split()
        return [tuple(l.normal) for l in me.loops]
    # Fallback: use polygon normals (flat shading)
    return [tuple(me.polygons[me.loops[li].polygon_index].normal) if hasattr(me.loops[li], 'polygon_index')
            else (0.0, 0.0, 1.0) for li in range(len(me.loops))]


# ─────────────────────────────────────────────────────────────
#  Main export function
# ─────────────────────────────────────────────────────────────
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
             **_):

    log.info("filepath: %s", filepath)
    log.info("options: scale=%.3f forward=%s up=%s normals=%s uvs=%s mats=%s tex=%s arm=%s weights=%s anim=%s fps=%.1f frames=%d-%d tri=%s",
             global_scale, axis_forward, axis_up,
             export_normals, export_uvs, export_materials, export_textures,
             export_armature, export_weights, export_animation,
             anim_fps, anim_frame_start, anim_frame_end, triangulate)

    scene     = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    objects   = context.selected_objects if use_selection else list(context.scene.objects)

    # Axis conversion Blender→DX, with scale embedded.
    # Apply inverse of what the importer applied when reading.
    bl_to_dx_3 = _axis_matrix(axis_forward, axis_up).to_3x3()
    inv_scale  = 1.0 / global_scale if global_scale != 0.0 else 1.0

    out = []
    w   = out.append

    # ─── header ────────────────────────────────────────────
    w("xof 0303txt 0032\n\n")
    w(f"AnimTicksPerSecond {{\n\t{int(anim_fps)};\n}}\n")

    mesh_objs     = [o for o in objects if o.type == "MESH"]
    armature_objs = [o for o in objects if o.type == "ARMATURE"]
    arm_obj       = armature_objs[0] if armature_objs else None
    log.info("exporting: %d meshes, %d armatures", len(mesh_objs), len(armature_objs))

    # ─── materials (top-level) ─────────────────────────────
    written_mats = {}  # name -> original Material object (for face material_index lookup)
    if export_materials:
        for obj in mesh_objs:
            for slot in obj.material_slots:
                mat = slot.material
                if not mat or mat.name in written_mats:
                    continue
                written_mats[mat.name] = mat
                log.debug("material '%s'", mat.name)
                bsdf = _get_principled(mat)
                # Prefer values the importer preserved from the original .x file
                # (stored as custom properties on the material). Fall back to
                # deriving them from the Principled BSDF for materials that
                # were created in Blender.
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
                        # Round-trip the original string verbatim (DOS path and all),
                        # escaping backslashes for the .x string literal form.
                        esc = x_tex_name.replace("\\", "\\\\")
                        w(f'\tTextureFileName {{"{esc}";}}\n')
                        log.debug("  texture (from _x_texture_filename): %s", x_tex_name)
                    else:
                        tp = _tex_path(mat)
                        if tp:
                            try:
                                rel = os.path.relpath(tp, os.path.dirname(filepath))
                            except ValueError:
                                rel = os.path.basename(tp)
                            rel = rel.replace("\\", "\\\\")
                            w(f'\tTextureFileName {{"{rel}";}}\n')
                            log.debug("  texture: %s", rel)
                w("}\n")
        log.info("materials written: %d", len(written_mats))

    # ─── armature (skeleton roots as top-level Frames) ─────
    if export_armature and arm_obj:
        arm_data = arm_obj.data
        log.info("armature '%s': %d bones", arm_obj.name, len(arm_data.bones))

        def write_bone(bone, indent):
            ind = "\t" * indent
            # Bone frame transform matrix = parent-local rest matrix in DX space.
            # Blender's bone.matrix_local is in ARMATURE (object) space with the
            # importer's conv_mat applied. Un-apply conv for the ROOT level only,
            # then express every level in parent-local.
            # Prefer the original FrameTransformMatrix if the importer stashed it.
            # The original .x file's FTM values often differ from the bind-pose
            # derived matrix (they're a separate piece of data — Burger.x uses
            # simplified head-translation-only FTMs). Round-tripping those exact
            # 16 floats is the only way to make the output byte-identical.
            stashed = arm_data.get(f"_x_ftm:{bone.name}")
            if stashed is not None and len(stashed) >= 16:
                ftm_string = ",".join(f"{float(v):.6f}" for v in stashed[:16])
            else:
                if bone.parent:
                    local_mat = bone.parent.matrix_local.inverted() @ bone.matrix_local
                    # Child-bone local matrix: conv_mat cancels between parent and child,
                    # so the matrix is already numerically the DX parent-local.
                else:
                    # Root bone: un-apply conv_mat and armature world transform.
                    # bone.matrix_local == conv_mat @ bind_pose_dx (as per importer).
                    # We also need to remove global_scale from translation only.
                    local_mat = _bl_bone_to_dx_world(bone.matrix_local, bl_to_dx_3, inv_scale)
                ftm_string = _mat4_to_dx(local_mat)

            w(f"{ind}Frame {bone.name} {{\n")
            w(f"{ind}\tFrameTransformMatrix {{\n")
            w(f"{ind}\t\t{ftm_string};;\n")
            w(f"{ind}\t}}\n")
            log.debug("  bone '%s' depth=%d", bone.name, indent)
            for child in bone.children:
                write_bone(child, indent + 1)
            w(f"{ind}}}\n")

        for rb in (b for b in arm_data.bones if not b.parent):
            write_bone(rb, 0)

    # ─── meshes (as sibling Frames, NOT children of bones) ─
    # All meshes live at top level in the importer's expected format. If a
    # mesh is parented to the armature it stays in armature world space; we
    # still emit it at top level.
    armature_set = set(armature_objs)
    for obj in mesh_objs:
        _write_mesh_frame(obj, out, 0,
                          depsgraph, use_mesh_modifiers,
                          export_normals, export_uvs,
                          export_materials, export_weights,
                          arm_obj, bl_to_dx_3, inv_scale,
                          triangulate, written_mats)

    # ─── animations ────────────────────────────────────────
    if export_animation and arm_obj:
        orig_frame = scene.frame_current
        frame_count = anim_frame_end - anim_frame_start + 1
        log.info("baking animation: %d bones x %d frames",
                 len(arm_obj.pose.bones), frame_count)

        # Pre-compute bone ancestry in Blender armature space. We need to convert
        # Blender pose delta → absolute DX parent-local quaternion for each frame.
        # Strategy: at each frame, read pose_bone.matrix (armature-space), build
        # parent-local matrix, un-apply conv_mat at root level, then decompose.

        # Cache rest matrices (armature-space Blender) per bone for derivation
        rest_arm_bl = {b.name: b.matrix_local.copy() for b in arm_obj.data.bones}

        # Bake per-frame data first so we have all keyframes before writing.
        # Map: bone_name -> {"rot": {frame: (w,x,y,z)}, "scale": {...}, "pos": {...}}
        baked = {b.name: {"rot": {}, "scale": {}, "pos": {}} for b in arm_obj.pose.bones}

        conv_3 = Matrix.Identity(3)  # identity placeholder — actual conv below
        # conv_mat equivalent that the IMPORTER used to go DX→BL. Exporter needs BL→DX = conv.T.
        # In the importer: conv_mat = _axis_matrix('-Z', 'Y') = [[1,0,0],[0,0,-1],[0,1,0]]
        # So conv_inv (BL→DX) = conv.T = [[1,0,0],[0,0,1],[0,-1,0]]
        # But the user-selected axis_forward/axis_up may change this. The exporter's
        # bl_to_dx_3 already is BL→DX, so conv_inv_3 = bl_to_dx_3, and conv_3 = inv.
        conv_inv_3 = bl_to_dx_3
        conv_3 = conv_inv_3.transposed()  # orthogonal → inverse == transpose

        for fr in range(anim_frame_start, anim_frame_end + 1):
            scene.frame_set(fr)

            for pb in arm_obj.pose.bones:
                name = pb.name

                # Armature-space world matrix of this bone at this frame.
                world_bl = pb.matrix.copy()

                # Express in PARENT-LOCAL space (Blender armature space).
                if pb.parent:
                    parent_world_bl = pb.parent.matrix.copy()
                    local_bl = parent_world_bl.inverted() @ world_bl
                    # conv_mat cancels between parent and child since both
                    # got the same conv applied. So local_bl is numerically
                    # the same as DX parent-local.
                    dx_local = local_bl
                else:
                    # Root: un-apply conv_mat to get true DX world matrix.
                    dx_local = _bl_bone_to_dx_world(world_bl, bl_to_dx_3, inv_scale)

                dx_rot = dx_local.to_3x3()
                dx_t   = dx_local.to_translation()
                dx_s   = dx_local.to_scale()
                q      = dx_rot.to_quaternion()

                # DX uses LH convention; the importer conjugates xyz on read.
                # So on write we conjugate xyz to cancel (w, -x, -y, -z) output.
                qw, qx, qy, qz = q.w, -q.x, -q.y, -q.z
                # The importer specifically negates w for root type-0 via the
                # (-1,0,0,0) identity pattern seen in the file. We match by
                # NOT re-negating w; the standard quat sign is kept and the
                # reader's logic flips xyz.

                baked[name]["rot"]  [fr] = (qw, qx, qy, qz)
                baked[name]["scale"][fr] = (dx_s.x, dx_s.y, dx_s.z)
                baked[name]["pos"]  [fr] = (dx_t.x, dx_t.y, dx_t.z)

        scene.frame_set(orig_frame)
        log.debug("baked %d frames for %d bones", frame_count, len(baked))

        w("AnimationSet anim {\n")
        for pb in arm_obj.pose.bones:
            name = pb.name
            rot_keys   = baked[name]["rot"]
            scale_keys = baked[name]["scale"]
            pos_keys   = baked[name]["pos"]

            w(f"\tAnimation {{\n\t\t{{ {name} }}\n")

            # type-0: rotation (quaternion)
            w(f"\t\tAnimationKey {{\n\t\t\t0;\n\t\t\t{len(rot_keys)};\n")
            entries = [f"\t\t\t{fr};4;{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f};;"
                       for fr, (qw, qx, qy, qz) in sorted(rot_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            # type-1: scale (3 values, but nvals field still written as 4 to
            # match the file format convention produced by Maya's exporter).
            w(f"\t\tAnimationKey {{\n\t\t\t1;\n\t\t\t{len(scale_keys)};\n")
            entries = [f"\t\t\t{fr};4;{sx:.6f},{sy:.6f},{sz:.6f};;"
                       for fr, (sx, sy, sz) in sorted(scale_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            # type-2: position
            w(f"\t\tAnimationKey {{\n\t\t\t2;\n\t\t\t{len(pos_keys)};\n")
            entries = [f"\t\t\t{fr};4;{px:.6f},{py:.6f},{pz:.6f};;"
                       for fr, (px, py, pz) in sorted(pos_keys.items())]
            w(",\n".join(entries) + ";\n\t\t}\n")

            w("\t}\n")

        w("}\n")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("".join(out))

    log.info("export done: %s", filepath)
    return {"FINISHED"}


# ─────────────────────────────────────────────────────────────
#  Convert a Blender-armature-space matrix to DX world space
# ─────────────────────────────────────────────────────────────
def _bl_bone_to_dx_world(bl_mat, bl_to_dx_3, inv_scale):
    """Invert the importer's rest-pose transformation.

    Importer did: rest_bl = conv_mat @ bind_pose_dx, then scaled translation.
    bl_to_dx_3 is the inverse of conv_mat's 3x3 (the importer's conv_mat is
    orthogonal so inverse == transpose, but we use the precomputed BL→DX matrix).

    To reverse: bind_pose_dx = bl_to_dx @ rest_bl, translation un-scaled.

    However there's a subtlety with the full 4x4: applying a 3x3 rotation R to
    a 4x4 matrix M to re-express it in a different coordinate frame is
    M' = R @ M @ R.T (conjugation), not just R @ M. But since rest_bl =
    conv_mat @ bind_pose_dx is LEFT multiplication only (not a conjugation),
    the inverse is simply bl_to_dx @ rest_bl with no right-mul.

    Note: the importer's conv_mat left-multiplies the DX bind pose, which is
    unusual but matches what the importer code does at line ~444:
        rest_mat = conv_mat @ bind_poses[name]
    So the inverse really is just left-mul by bl_to_dx.
    """
    # Apply BL→DX to the entire 4x4 by treating it as a rotation matrix
    # promoted to 4x4 with identity translation row.
    conv_4 = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            conv_4[r][c] = bl_to_dx_3[r][c]
    dx_mat = conv_4 @ bl_mat
    # Un-apply the global_scale that the importer applied to bone translations.
    dx_mat[0][3] *= inv_scale
    dx_mat[1][3] *= inv_scale
    dx_mat[2][3] *= inv_scale
    return dx_mat


# ─────────────────────────────────────────────────────────────
#  Per-mesh writer
# ─────────────────────────────────────────────────────────────
def _write_mesh_frame(obj, out, indent,
                      depsgraph, use_mesh_modifiers,
                      export_normals, export_uvs,
                      export_materials, export_weights,
                      arm_obj, bl_to_dx_3, inv_scale,
                      triangulate, written_mats):
    w   = out.append
    ind = "\t" * indent
    log.info("  mesh '%s'", obj.name)

    # Vertex positions must come from the bind pose, not the current animated
    # frame.  obj.data always holds the raw undeformed vertex positions
    # regardless of the current frame or depsgraph state.
    me_src = obj.data

    # Work on a bmesh copy so we can optionally triangulate without touching
    # the original data. If triangulate=False we preserve n-gons so the
    # output matches the original file's quad structure.
    bm = bmesh.new()
    bm.from_mesh(me_src)
    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces)
    me_work = bpy.data.meshes.new("_x_export_tmp")
    bm.to_mesh(me_work)
    bm.free()

    me_work.update()

    conv_inv_3 = bl_to_dx_3   # BL→DX rotation

    # ── UN-WELD: emit one vertex per face corner ─────────────
    # DX .x doesn't store "welds" — the file just has vertices and face indices.
    # Per-corner attributes (split normals, UV seams) are encoded by duplicating
    # a vertex at each corner that needs different data. Maya's exporter produces
    # ~4x more vertices than the welded mesh for this reason (Burger.x has 10880
    # verts for 2800 quads = 11200 corners, with some shared where attributes match).
    #
    # For clean round-trip we un-weld on export: emit one vertex per face corner.
    # Skin weights get re-indexed from (source Blender vertex) to (new corner-vert
    # indices that came from that Blender vertex).
    # The importer will re-weld on read, producing the same welded topology.

    # Build un-welded vertex list + face index list
    bl_verts = me_work.vertices
    new_to_src = []                  # new_vi → source Blender vertex index
    faces = []                       # new vertex indices per face
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
    # Emit vertex positions (one per un-welded corner, in DX space)
    verts_dx = [(conv_inv_3 @ Vector(bl_verts[src_vi].co)) * inv_scale
                for src_vi in new_to_src]
    log.info("    %d verts (unwelded from %d), %d faces",
             n_verts, len(bl_verts), n_faces)

    # Loop normals: gather once up front; may be rotated into DX space below.
    loop_normals = _get_corner_normals(me_work) if export_normals else []

    # ─── Frame + FrameTransformMatrix ─────────────────────
    # Prefer original names / FTM from the import, when stashed. This makes
    # the output match the source file even when the Blender-derived values
    # would differ (e.g. Burger.x uses a simplified identity-ish FTM that
    # isn't recoverable from the mesh data alone).
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

    # ─── Mesh block ───────────────────────────────────────
    w(f"{ind}\tMesh {mesh_name} {{\n")

    # vertices — DX format: "x.xxxxxx; y.yyyyyy; z.zzzzzz;,"
    w(f"{ind}\t\t{n_verts};\n")
    vert_lines = [f"{ind}\t\t {v.x:.6f}; {v.y:.6f}; {v.z:.6f};" for v in verts_dx]
    w(",\n".join(vert_lines) + ";\n")

    # faces — DX format: "N;i0,i1,...,iN-1;"
    w(f"\n{ind}\t\t{n_faces};\n")
    face_lines = [f"{ind}\t\t{len(f)};{','.join(str(i) for i in f)};" for f in faces]
    w(",\n".join(face_lines) + ";\n")

    # ─── MeshNormals ───────────────────────────────────────
    if export_normals and loop_normals:
        # Rotate loop normals from Blender space back to DX space.
        dx_loop_normals = [conv_inv_3 @ Vector(n) for n in loop_normals]

        # Deduplicate to match original file format (unique normal pool + per-corner index).
        unique_norms   = []
        norm_key_to_idx = {}
        # Per-face per-corner normal index
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

        log.debug("    normals: %d unique", len(unique_norms))
        w(f"\n{ind}\t\tMeshNormals {{\n{ind}\t\t\t{len(unique_norms)};\n")
        w(",\n".join(f"{ind}\t\t\t {nx:.6f},{ny:.6f},{nz:.6f};"
                     for nx, ny, nz in unique_norms) + ";\n")
        w(f"\n{ind}\t\t\t{n_faces};\n")
        face_norm_lines = [f"{ind}\t\t\t{len(fni)};{','.join(str(i) for i in fni)};"
                           for fni in face_norm_idx]
        w(",\n".join(face_norm_lines) + ";\n")
        w(f"{ind}\t\t}}\n")

    # ─── MeshTextureCoords ─────────────────────────────────
    if export_uvs and me_work.uv_layers:
        uv_layer = me_work.uv_layers.active
        # The importer does (u, 1.0 - v) on read. To round-trip, we write
        # (u, 1.0 - v_blender) which produces the V-flipped values like the
        # original file (which has negative V coordinates).
        #
        # DX stores ONE UV per vertex. With un-welding, each face corner has
        # its own vertex so we can preserve per-loop UVs directly: write one
        # UV per un-welded vertex using that corner's UV from the source loop.
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

    # ─── MeshMaterialList ──────────────────────────────────
    if export_materials and obj.material_slots:
        mats = [s.material for s in obj.material_slots if s.material]
        if mats:
            log.debug("    MaterialList: %d materials, %d faces", len(mats), n_faces)
            w(f"\n{ind}\t\tMeshMaterialList {{\n{ind}\t\t\t{len(mats)};\n{ind}\t\t\t{n_faces};\n")
            w(",\n".join(f"{ind}\t\t\t{p.material_index}" for p in me_work.polygons) + ";\n")
            for mat in mats:
                # Original file uses {blinn1} — tight braces. Preserve that.
                w(f"{ind}\t\t\t{{{mat.name}}}\n")
            w(f"{ind}\t\t}}\n")

    # ─── XSkinMeshHeader + SkinWeights ─────────────────────
    if export_weights and arm_obj and obj.vertex_groups:
        bone_names = {b.name for b in arm_obj.data.bones}
        # Use ORIGINAL object's vertex groups (evaluated mesh's are the same).
        # Work from the original object's mesh data since that carries the
        # skin-weight assignments.
        orig_me = obj.data
        vgroups = [vg for vg in obj.vertex_groups if vg.name in bone_names]
        log.info("    SkinWeights: %d bone groups", len(vgroups))

        if vgroups:
            # Collect influences per group. We use the WORKING mesh's vertex
            # positions (not the original pre-modifier) — for modifier-applied
            # exports this is correct because vertex indices match.
            # If modifiers changed vertex count, fall back gracefully.
            ref_me = orig_me if not use_mesh_modifiers or len(orig_me.vertices) == n_verts else me_work

            # Build reverse map: source BL vertex → list of new un-welded indices
            src_to_new = {}
            for new_vi, src_vi in enumerate(new_to_src):
                src_to_new.setdefault(src_vi, []).append(new_vi)

            group_influences = []  # list of (vgroup, [(new_vi, weight), ...])
            for vg in vgroups:
                influences = []
                gi = vg.index
                for v in ref_me.vertices:
                    for g in v.groups:
                        if g.group == gi and g.weight > 0.0:
                            # Emit one influence entry per un-welded copy of this source vertex.
                            for nv in src_to_new.get(v.index, []):
                                influences.append((nv, g.weight))
                            break
                if influences:
                    group_influences.append((vg, influences))

            if group_influences:
                n_groups = len(group_influences)
                # XSkinMeshHeader fields: max per-vertex weights, max per-face weights, bone count
                # We use bone count for all three — matches the original file's pattern.
                w(f"\n{ind}\t\tXSkinMeshHeader {{\n")
                w(f"{ind}\t\t\t{n_groups};\n{ind}\t\t\t{n_groups};\n{ind}\t\t\t{n_groups};\n")
                w(f"{ind}\t\t}}\n")

                for vg, influences in group_influences:
                    bone = arm_obj.data.bones.get(vg.name)
                    log.debug("      '%s': %d influences", vg.name, len(influences))

                    # Offset matrix = inv(bind pose in DX world space).
                    # bone.matrix_local is in Blender armature space. We need to
                    # un-apply conv_mat to get back to DX world space, then invert.
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

    # Close Mesh and Frame
    w(f"{ind}\t}}\n")
    w(f"{ind}}}\n")

    bpy.data.meshes.remove(me_work)
