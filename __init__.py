"""
DirectX .x Format Importer/Exporter for Blender
================================================
Supports: geometry, normals, UVs, materials, textures,
          armatures (Frame hierarchy), skin weights,
          and keyframe animations (rotation/scale/position).
"""

bl_info = {
    "name": "DirectX X Format (.x)",
    "author": "Generated for Burger.x",
    "version": (1, 1, 6),
    "blender": (3, 0, 0),
    "location": "File > Import-Export",
    "description": "Import/Export DirectX .x files — full armature, skin, animation, material and texture support",
    "category": "Import-Export",
}

import bpy
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
        # Include uppercase variants because Project Zomboid (and some
        # legacy DirectX SDK exports) ship files as `.X` rather than `.x`,
        # and fnmatch is case-sensitive on Linux/macOS.
        default="*.x;*.X;*.xcache;*.XCACHE",
        options={"HIDDEN"},
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

    import_armature:  BoolProperty(name="Import Armature",  default=True)
    import_weights:   BoolProperty(name="Import Weights",   default=True)
    import_animation: BoolProperty(name="Import Animation", default=True)

    rest_pose_source: EnumProperty(
        name="Rest Pose Source",
        description=(
            "Where to read bone rest poses from.\n\n"
            "Frame Hierarchy (default): uses the FrameTransformMatrix "
            "chain — the canonical .x convention, matching dev-supplied "
            ".x files.\n\n"
            "Bind Pose: uses the SkinWeights offset matrices. Kept for "
            "backwards compatibility with files whose FTM is ambiguous."
        ),
        items=[
            ('FRAME_TRANSFORM', "Frame Hierarchy", "Use FrameTransformMatrix chain (canonical .x convention)"),
            ('BIND',            "Bind Pose",       "Use SkinWeights offset matrices (legacy)"),
        ],
        default='FRAME_TRANSFORM',
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

        box = layout.box()
        box.label(text="Armature & Animation", icon="ARMATURE_DATA")
        box.prop(self, "import_armature")
        box.prop(self, "import_weights")
        box.prop(self, "import_animation")
        box.prop(self, "rest_pose_source")
        box.prop(self, "anim_fps")
        box.prop(self, "set_frame_range")

    def execute(self, context):
        keywords = self.as_keywords(ignore=("filter_glob",))
        import os, tempfile, traceback
        debug = bool(os.environ.get("SNAKFORGE_DEBUG"))
        log_path = os.path.join(tempfile.gettempdir(), "snakforge_import.log")
        try:
            result = import_x(context, **keywords)
        except Exception as e:
            # When debug logging is on, dump the traceback so it
            # survives even if Blender was launched without a console.
            # When debug is off, skip the file I/O entirely — Blender's
            # own error reporting will still surface the exception.
            if debug:
                try:
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write("\n!!! UNHANDLED EXCEPTION in execute():\n")
                        fh.write(traceback.format_exc())
                        fh.flush()
                except Exception:
                    pass
                self.report({'ERROR'}, f"Import failed: {e}. Log: {log_path}")
            else:
                self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}
        if debug:
            self.report({'INFO'}, f"Import OK. Log: {log_path}")
        return result

class ExportDirectX(bpy.types.Operator, ExportHelper):
    """Export selected objects as a DirectX .x file"""
    bl_idname  = "export_scene.directx_x"
    bl_label   = "Export DirectX X (.x)"
    bl_options = {"REGISTER"}

    filename_ext = ".x"
    filter_glob: StringProperty(
        default="*.x;*.X;*.xcache;*.XCACHE",
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
            ("XCACHE",            "Binary .xcache",      "Horsepower Engine SEMS .xcache (skeleton + animation only — no mesh)"),
        ],
        default="TEXT_X",
    )

    pz_compat: BoolProperty(
        name="High-precision animation ticks",
        description=(
            "Write the animation block with the conventions used by 3DS Max "
            "biped exports (and games like Project Zomboid):\n"
            "  • AnimTicksPerSecond = 4800 (vs the FPS field's value)\n"
            "  • AnimationKey nodes named 'R', 'S', 'T' (rotation/scale/translation)\n"
            "Blender frame numbers are scaled to the 4800-tick rate so the "
            "animation plays at the correct real-time speed. Has no effect "
            "when exporting .xcache."
        ),
        default=False,
    )

    anim_fps: FloatProperty(
        name="FPS",
        description=(
            "Animation tick rate (AnimTicksPerSecond) and target real-time "
            "playback rate. Blender frame numbers are written directly as "
            "tick values. When Project Zomboid Compatibility is on, this "
            "field is ignored and 4800 is used instead."
        ),
        default=30.0, min=1.0, max=10000.0,
    )
    anim_frame_start: IntProperty(name="Frame Start", default=1)
    anim_frame_end:   IntProperty(name="Frame End",   default=250)

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
        # The high-precision-ticks option only affects .x animation
        # output; .xcache has its own engine-specific layout.
        if self.export_format != "XCACHE":
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
        """Update the filename extension when the format dropdown changes."""
        import os as _os
        fp = self.filepath or ""
        base, cur_ext = _os.path.splitext(fp)
        wanted_ext = ".xcache" if self.export_format == "XCACHE" else ".x"

        changed = False
        cur_ext_lower = cur_ext.lower()
        if cur_ext_lower in (".x", ".xcache", ".xanim"):
            # One of our extensions is already there.  If it matches what
            # the dropdown wants, keep it.  Otherwise swap it for the
            # dropdown's choice so the user doesn't have to retype the
            # filename when toggling between .x and .xcache.
            if cur_ext_lower != wanted_ext:
                self.filepath = base + wanted_ext
                changed = True
            self.filename_ext = wanted_ext
        elif not cur_ext:
            self.filepath = fp + wanted_ext
            self.filename_ext = wanted_ext
            changed = True
        else:
            # Some unrelated extension — keep filename_ext aligned with
            # the dropdown.
            self.filename_ext = wanted_ext

        # Defer to ExportHelper for any other normalization it wants to do
        super_changed = super().check(context)
        return changed or super_changed

    def execute(self, context):
        keywords = self.as_keywords(ignore=("filter_glob", "check_existing"))

        # Pull format off keywords; default routes to text .x
        fmt = keywords.pop("export_format", "TEXT_X")

        # If the user explicitly typed a filename ending in .xcache, that
        import os as _os
        base, cur_ext = _os.path.splitext(self.filepath)
        cur_ext_lower = cur_ext.lower()
        if cur_ext_lower == ".xcache":
            fmt = "XCACHE"
        elif cur_ext_lower == ".x":
            # Honour the dropdown's choice between TEXT_X and BINARY_X,
            # but never let it pick XCACHE when the file is .x.
            if fmt == "XCACHE":
                fmt = "TEXT_X"
        elif not cur_ext_lower:
            # No extension at all — fall back to the dropdown and append
            # the matching extension.
            wanted_ext = ".xcache" if fmt == "XCACHE" else ".x"
            self.filepath = base + wanted_ext
            keywords["filepath"] = self.filepath

        if fmt == "XCACHE":
            from .exporter import export_xcache_from_blender
            # The Blender bridge takes only the parameters it understands;
            # filter the keyword dict to drop .x-specific options it ignores.
            xc_keywords = {k: v for k, v in keywords.items() if k in (
                "filepath", "use_selection", "use_mesh_modifiers", "global_scale",
                "axis_forward", "axis_up",
                "export_armature", "export_weights", "export_animation",
                "anim_frame_start", "anim_frame_end",
            )}
            result, warnings = export_xcache_from_blender(context, **xc_keywords)
        else:
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
