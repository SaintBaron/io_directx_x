"""
DirectX .x Format Importer/Exporter for Blender
================================================
Supports: geometry, normals, UVs, materials, textures,
          armatures (Frame hierarchy), skin weights,
          and keyframe animations (rotation/scale/position).
"""

bl_info = {
    "name": "DirectX X Format (.x)",
    "author": "",
    "version": (1, 3, 0),
    "blender": (3, 0, 0),
    "location": "File > Import-Export",
    "description": "Import/Export DirectX .x files — full armature, skin, animation, material and texture support",
    "category": "Import-Export",
}

import bpy
import os
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty,
    EnumProperty, IntProperty,
)
from bpy_extras.io_utils import ImportHelper, ExportHelper

from .importer import import_x
from .exporter import export_x

class ImportDirectX(bpy.types.Operator, ImportHelper):
    """Import a DirectX .x file"""
    bl_idname  = "import_scene.directx_x"
    bl_label   = "Import DirectX X (.x)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".x"
    filter_glob: StringProperty(
        # Include uppercase variants because some games (and some
        # legacy DirectX SDK exports) ship files as `.X` rather than `.x`,
        # and fnmatch is case-sensitive on Linux/macOS.
        default="*.x;*.X",
        options={"HIDDEN"},
    )

    # Multi-file selection: Blender populates `files` automatically when
    # the user selects multiple files in the file browser.  `directory`
    # holds the folder they were chosen from.
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    directory: StringProperty(
        subtype="DIR_PATH",
        options={"HIDDEN", "SKIP_SAVE"},
    )

    use_apply_transform: BoolProperty(
        name="Apply Transform",
        description="Apply the root Frame transform matrix to mesh data",
        default=True,
    )
    global_scale: FloatProperty(
        name="Scale", default=1.0, min=0.001, max=1000.0,
    )
    axis_forward: EnumProperty(
        name="Forward Axis",
        items=[("-Z","−Z",""),("Z","Z",""),("Y","Y",""),
               ("-Y","−Y",""),("X","X",""),("-X","−X","")],
        default="Z",
    )
    axis_up: EnumProperty(
        name="Up Axis",
        items=[("Y","Y",""),("-Y","−Y",""),("Z","Z",""),
               ("-Z","−Z",""),("X","X",""),("-X","−X","")],
        default="Y",
    )

    import_normals:   BoolProperty(name="Import Normals",   default=True)
    import_uvs:       BoolProperty(name="Import UVs",       default=True)
    import_materials: BoolProperty(name="Import Materials", default=True)
    import_textures:  BoolProperty(name="Import Textures",  default=True)
    use_diffuse_alpha: BoolProperty(
        name="Use Diffuse Alpha",
        description=(
            "Connect the alpha channel of the diffuse (_D) texture to "
            "the material's Alpha input, and set the blend mode to "
            "alpha clip/blend. some games textures sometimes encode "
            "transparency in the diffuse alpha channel (e.g. leaf "
            "cutouts on plants, eye highlights). Without this, the "
            "textures look opaque in Blender's viewport even when "
            "they were authored with alpha.\n\n"
            "If a texture has no alpha channel (most opaque materials), "
            "connecting alpha is a no-op so this is safe to leave on."
        ),
        default=True,
    )
    smooth_shade_from_faces: BoolProperty(
        name="Smooth Shade from Faces",
        description=(
            "Apply Blender's shade-smooth pass to imported meshes so face "
            "normals are averaged at shared vertices.  Useful for files "
            "where the file-stored per-loop normals don't render cleanly "
            "(e.g. faceting that the artist didn't intend).  Untick to "
            "keep the file's authored normals exactly as-is"
        ),
        default=False,
    )

    split_submeshes: BoolProperty(
        name="Split Sub-Meshes",
        description=(
            "Import multi-part models as separate Blender objects "
            "sharing one armature, rather than merged into one mesh.\n\n"
            "Applies to .x files whose Mesh has MULTIPLE materials "
            "assigned to different faces.\n\n"
            "ON (default): each material group becomes its own object, "
            "named after the source mesh. Each has its own material, "
            "texture set, and vertex groups for the bones that weight "
            "it.\n\n"
            "OFF: all materials are kept in one Blender object with "
            "per-face material assignments and merged skin weights. "
            "Single-material .x files are never split regardless of "
            "this setting"
        ),
        default=True,
    )

    triangulate_quads: BoolProperty(
        name="Triangulate Quads",
        description=(
            "Convert quad and n-gon faces into pairs of triangles on "
            "import.\n\n"
            "ON (default): all imported meshes have only triangle "
            "faces, matching what most engines use internally. Many .x "
            "files are exported with the original quad topology; without "
            "this option, those quads come through into Blender as-is.\n\n"
            "OFF: preserve the original quad / n-gon topology from .x "
            "files. Better for editing, subdivision, and animation "
            "workflows that benefit from quad mesh structure. Round-trip "
            "to .x preserves the quads"
        ),
        default=True,
    )

    import_type: EnumProperty(
        name="Import Type",
        description=(
            "Which exporter produced this .x file. Each is a separate, "
            "self-contained import path — picking the right one avoids the "
            "guesswork of auto-detection and keeps the conversions independent"
        ),
        items=[
            ("standard", "Standard .x",
             "General DirectX .x files from common exporters. This is the "
             "original, well-tested path"),
            ("ms_dx8", "DirectX 8.0 (rigid skin)",
             "A single skinned mesh with SkinWeights / matrixOffset bind "
             "matrices and rigid (one-bone-per-vertex) binding"),
        ],
        default="standard",
    )

    import_armature:  BoolProperty(name="Import Armature",  default=True)
    import_weights:   BoolProperty(name="Import Weights",   default=True)
    import_animation: BoolProperty(name="Import Animation", default=True)

    weld_duplicate_verts: BoolProperty(
        name="Weld Duplicate Vertices",
        description=(
            "Collapse vertices that sit at the same XYZ position into "
            "one and blend their bone weights.\n\n"
            "ON (default): produces smooth shading across face "
            "boundaries. The .x format stores per-loop normals as "
            "split verts; without welding, adjacent faces don't "
            "share vertices, so smooth-shading has nothing to "
            "average across and the mesh looks blocky.\n\n"
            "OFF: keeps the file's exact vertex authoring. Use when "
            "round-trip preservation matters more than shading "
            "quality, e.g. when bit-comparing exports against a "
            "reference file.\n\n"
            "Ignored for the DirectX 8.0 rigid-skin path, where welding "
            "would fuse coincident verts that belong to different bones "
            "(e.g. eyelid pairs) and collapse hard-edge seam splits"
        ),
        default=True,
    )

    anim_fps: FloatProperty(
        name="Animation FPS",
        description="Override FPS (0 = read from file's AnimTicksPerSecond)",
        default=0.0, min=0.0, max=240.0,
    )
    set_frame_range: BoolProperty(
        name="Set Scene Frame Range",
        description=(
            "Scan the animation data and set scene frame_start / frame_end "
            "to match the authored range.  The file's tick numbers are "
            "preserved verbatim, so an animation authored from tick 0 to "
            "120 will play from frame 0 to 120 regardless of what the "
            "scene range was set to before"
        ),
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        box = layout.box()
        box.label(text="Import Type", icon="FILE_3D")
        box.prop(self, "import_type")

        box = layout.box()
        box.label(text="Transform", icon="ORIENTATION_GLOBAL")
        box.prop(self, "global_scale")
        box.prop(self, "axis_forward")
        box.prop(self, "axis_up")
        box.prop(self, "use_apply_transform")

        box = layout.box()
        box.label(text="Data", icon="MESH_DATA")
        box.prop(self, "import_normals")
        box.prop(self, "smooth_shade_from_faces")
        box.prop(self, "import_uvs")
        box.prop(self, "import_materials")
        box.prop(self, "import_textures")
        box.prop(self, "use_diffuse_alpha")
        box.prop(self, "split_submeshes")
        box.prop(self, "triangulate_quads")

        box = layout.box()
        box.label(text="Armature & Animation", icon="ARMATURE_DATA")
        box.prop(self, "import_armature")
        box.prop(self, "import_weights")
        # The weld toggle is always available. It has no effect on the
        # DirectX 8.0 rigid-skin path (welding is suppressed there), but
        # stays visible so its meaning is discoverable.
        box.prop(self, "weld_duplicate_verts")
        box.prop(self, "import_animation")
        box.prop(self, "anim_fps")
        box.prop(self, "set_frame_range")

    def execute(self, context):
        # Build the list of files to import.  When the user selects multiple
        # files in the browser, `self.files` is populated and `self.directory`
        # holds their common parent.  When only one file is chosen (or the
        # operator is called programmatically), fall back to `self.filepath`.
        if self.files and self.directory:
            filepaths = [
                os.path.join(self.directory, f.name)
                for f in self.files
                if f.name
            ]
        else:
            filepaths = [self.filepath]

        # Remove duplicates while preserving order (can happen when both
        # `files` and `filepath` point at the same item).
        seen = set()
        unique_paths = []
        for p in filepaths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        keywords = self.as_keywords(ignore=("filter_glob", "files", "directory"))
        last_result = {"FINISHED"}
        errors = []
        for fp in unique_paths:
            keywords["filepath"] = fp
            try:
                last_result = import_x(context, **keywords)
            except Exception as e:
                errors.append((fp, e))

        if errors:
            msg = "; ".join(f"{os.path.basename(fp)}: {e}" for fp, e in errors)
            self.report({'ERROR'}, f"Import failed: {msg}")
            return {'CANCELLED'} if len(errors) == len(unique_paths) else {'FINISHED'}

        return last_result

class ExportDirectX(bpy.types.Operator, ExportHelper):
    """Export selected objects as a DirectX .x file"""
    bl_idname  = "export_scene.directx_x"
    bl_label   = "Export DirectX X (.x)"
    bl_options = {"REGISTER"}

    filename_ext = ".x"
    filter_glob: StringProperty(
        default="*.x;*.X",
        options={"HIDDEN"},
    )

    use_selection: BoolProperty(
        name="Selected Only",
        description="Export only selected objects",
        default=False,
    )
    use_mesh_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Apply mesh modifiers before exporting",
        default=True,
    )

    global_scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0)
    axis_forward: EnumProperty(
        name="Forward Axis",
        items=[("-Z","−Z",""),("Z","Z",""),("Y","Y",""),
               ("-Y","−Y",""),("X","X",""),("-X","−X","")],
        # Match the importer's default ("Z") so a re-export of an imported
        default="Z",
    )
    axis_up: EnumProperty(
        name="Up Axis",
        items=[("Y","Y",""),("-Y","−Y",""),("Z","Z",""),
               ("-Z","−Z",""),("X","X",""),("-X","−X","")],
        default="Y",
    )

    export_normals:   BoolProperty(name="Export Normals",   default=True)
    export_uvs:       BoolProperty(name="Export UVs",       default=True)
    export_materials: BoolProperty(name="Export Materials", default=True)
    export_textures:  BoolProperty(name="Export Textures",  default=True)
    export_armature:  BoolProperty(name="Export Armature",  default=True)
    export_weights:   BoolProperty(name="Export Weights",   default=True)
    export_animation: BoolProperty(name="Export Animation", default=True)

    anim_key_format: EnumProperty(
        name="Animation Keys",
        description=(
            "How animation is written to the .x file.\n\n"
            "Quaternion (default): three separate tracks per bone — "
            "rotation as a 4-component quaternion (keyType 0), scale "
            "(keyType 1) and position (keyType 2). This is what most "
            "DirectX tools emit, and what the widest range of importers "
            "accept.\n\n"
            "Matrix: one 4x4 transform per frame per bone (keyType 4). "
            "This gives the most faithful round-trip for files authored "
            "with matrix keys (no decomposition step), and preserves "
            "transforms that a quaternion+scale split cannot represent "
            "(e.g. shear)"
        ),
        items=[
            ("TRS", "Quaternion",
             "Separate rotation (quaternion), scale and position tracks"),
            ("MATRIX", "Matrix",
             "One 4x4 matrix per frame (keyType 4)"),
        ],
        default="TRS",
    )

    unweld_on_export: BoolProperty(
        name="Unweld on Export",
        description=(
            "Split every face-loop into its own output vertex so the "
            "exported mesh has one vertex per loop (no shared verts "
            "between faces).\n\n"
            "ON (default): restores the original .x file's vert count "
            "when the mesh was imported with welding enabled. Many .x "
            "files store per-loop normals/UVs by splitting verts at "
            "UV and smoothing-group seams, so welding-on-import "
            "collapsed those splits — this option reverses that for "
            "round-trip fidelity.\n\n"
            "OFF: writes the mesh with the in-Blender vert count. Only "
            "minimal de-duplication is done (verts shared between faces "
            "that genuinely have identical normals + UVs). Smaller "
            "files, but normals may render differently than the "
            "original .x file"
        ),
        default=True,
    )

    use_original_material_data: BoolProperty(
        name="Use Original Material Data",
        description=(
            "When ticked, exports the material values as originally imported "
            "from the .x file, ignoring any changes made in Blender. "
            "When unticked (default), exports the current Blender material "
            "state so your edits are included in the output"
        ),
        default=False,
    )

    export_format: EnumProperty(
        name="Format",
        description="Output file format",
        items=[
            ("TEXT_X",            "Text .x",              "DirectX text format (human-readable)"),
            ("BINARY_X",          "Binary .x",            "DirectX binary format (smaller, faster to parse)"),
            ("COMPRESSED_TEXT_X", "Compressed Text .x",   "DirectX text format compressed with MSZIP (tzip)"),
            ("COMPRESSED_BIN_X",  "Compressed Binary .x", "DirectX binary format compressed with MSZIP (bzip)"),
        ],
        default="TEXT_X",
    )

    pz_compat: BoolProperty(
        name="High-precision animation ticks",
        description=(
            "Write the animation block with high-precision tick "
            "conventions used by some biped/game exports:\n"
            "  • AnimTicksPerSecond = 4800 (vs the FPS field's value)\n"
            "  • AnimationKey nodes named 'R', 'S', 'T' (rotation/scale/translation)\n"
            "Blender frame numbers are scaled to the 4800-tick rate so the "
            "animation plays at the correct real-time speed."
        ),
        default=False,
    )

    anim_fps: FloatProperty(
        name="FPS",
        description=(
            "Animation tick rate (AnimTicksPerSecond) and target real-time "
            "playback rate. Blender frame numbers are written directly as "
            "tick values. When High-precision animation ticks is on, this "
            "field is ignored and 4800 is used instead."
        ),
        default=30.0, min=1.0, max=10000.0,
    )
    anim_frame_start: IntProperty(name="Frame Start", default=1)
    anim_frame_end:   IntProperty(name="Frame End",   default=250)

    triangulate: BoolProperty(
        name="Triangulate Faces",
        description=(
            "Convert all polygons to triangles on export.\n\n"
            "ON (default): produces triangle-only meshes, which load "
            "reliably in all DirectX viewers.\n\n"
            "OFF: preserves quads and n-gons. Turn this off to keep "
            "quad topology on round-trip"
        ),
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        box = layout.box()
        box.label(text="Include", icon="OBJECT_DATA")
        box.prop(self, "use_selection")
        box.prop(self, "use_mesh_modifiers")

        box = layout.box()
        box.label(text="Transform", icon="ORIENTATION_GLOBAL")
        box.prop(self, "global_scale")
        box.prop(self, "axis_forward")
        box.prop(self, "axis_up")

        box = layout.box()
        box.label(text="Data", icon="MESH_DATA")
        box.prop(self, "export_normals")
        box.prop(self, "export_uvs")
        box.prop(self, "triangulate")
        box.prop(self, "unweld_on_export")
        box.prop(self, "export_materials")
        box.prop(self, "export_textures")
        if self.export_materials:
            box.prop(self, "use_original_material_data")

        if self.export_materials and self.export_textures:
            mats_with_tex = []
            for obj in context.scene.objects:
                if obj.type != "MESH":
                    continue
                for slot in obj.material_slots:
                    mat = slot.material
                    if mat and mat.get("_x_texture_filename") is not None:
                        if mat not in mats_with_tex:
                            mats_with_tex.append(mat)
            if mats_with_tex:
                tbox = layout.box()
                tbox.label(text="Texture Paths", icon="IMAGE_DATA")
                tbox.label(text="Edit before export if needed:", icon="INFO")
                for mat in mats_with_tex:
                    row = tbox.row()
                    row.label(text=mat.name, icon="MATERIAL")
                    row.prop(mat, '["_x_texture_filename"]', text="")

        box = layout.box()
        box.label(text="Format", icon="FILE_BLANK")
        box.prop(self, "export_format")

        box = layout.box()
        box.label(text="Armature & Animation", icon="ARMATURE_DATA")
        box.prop(self, "export_armature")
        box.prop(self, "export_weights")
        box.prop(self, "export_animation")
        box.prop(self, "anim_key_format")
        box.prop(self, "pz_compat")
        row = box.row()
        row.prop(self, "anim_fps")
        row = box.row()
        row.prop(self, "anim_frame_start")
        row.prop(self, "anim_frame_end")

    def invoke(self, context, event):
        # Pre-populate the frame range from the current scene so that after
        sc = context.scene
        self.anim_frame_start = sc.frame_start
        self.anim_frame_end   = sc.frame_end
        return super().invoke(context, event)

    def check(self, context):
        """Keep the filename extension as .x."""
        fp = self.filepath or ""
        base, cur_ext = os.path.splitext(fp)
        wanted_ext = ".x"

        changed = False
        cur_ext_lower = cur_ext.lower()
        if cur_ext_lower in (".x", ".xanim"):
            if cur_ext_lower != wanted_ext:
                self.filepath = base + wanted_ext
                changed = True
            self.filename_ext = wanted_ext
        elif not cur_ext:
            self.filepath = fp + wanted_ext
            self.filename_ext = wanted_ext
            changed = True
        else:
            self.filename_ext = wanted_ext

        super_changed = super().check(context)
        return changed or super_changed

    def execute(self, context):
        keywords = self.as_keywords(ignore=("filter_glob", "check_existing"))

        # Pull format off keywords; default routes to text .x
        fmt = keywords.pop("export_format", "TEXT_X")

        base, cur_ext = os.path.splitext(self.filepath)
        if not cur_ext:
            self.filepath = base + ".x"
            keywords["filepath"] = self.filepath

        # Map format choice to (binary_format, compress) flags
        keywords["binary_format"] = fmt in ("BINARY_X", "COMPRESSED_BIN_X")
        keywords["compress"]      = fmt in ("COMPRESSED_TEXT_X", "COMPRESSED_BIN_X")
        result, warnings = export_x(context, **keywords)

        for msg in warnings:
            self.report({'WARNING'}, msg)
        return result

def menu_import(self, context):
    self.layout.operator(ImportDirectX.bl_idname, text="DirectX X (.x)")

def menu_export(self, context):
    self.layout.operator(ExportDirectX.bl_idname, text="DirectX X (.x)")

classes = (ImportDirectX, ExportDirectX)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
