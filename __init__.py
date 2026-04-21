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
    "version": (1, 8, 0),
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

from .xlog    import XLog
from .importer import import_x
from .exporter import export_x

_log = XLog("init")


# ─────────────────────────────────────────────────────────────
#  Import operator
# ─────────────────────────────────────────────────────────────
class ImportDirectX(bpy.types.Operator, ImportHelper):
    """Import a DirectX .x file"""
    bl_idname  = "import_scene.directx_x"
    bl_label   = "Import DirectX X (.x)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".x"
    filter_glob: StringProperty(default="*.x", options={"HIDDEN"})

    # Transform
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
        default="-Z",
    )
    axis_up: EnumProperty(
        name="Up Axis",
        items=[("Y","Y",""),("-Y","−Y",""),("Z","Z",""),
               ("-Z","−Z",""),("X","X",""),("-X","−X","")],
        default="Y",
    )

    # Data
    import_normals:   BoolProperty(name="Import Normals",   default=True)
    import_uvs:       BoolProperty(name="Import UVs",       default=True)
    import_materials: BoolProperty(name="Import Materials", default=True)
    import_textures:  BoolProperty(name="Import Textures",  default=True)
    import_armature:  BoolProperty(name="Import Armature",  default=True)
    import_weights:   BoolProperty(name="Import Weights",   default=True)
    import_animation: BoolProperty(name="Import Animation", default=True)

    # Normals
    infer_sharps: BoolProperty(
        name="Infer Sharp Edges",
        description=(
            "Mark edges as sharp where adjacent face normals diverge beyond the "
            "threshold angle. Produces correct hard edges from the source normals "
            "without requiring manual crease marking"
        ),
        default=True,
    )
    sharp_angle_deg: FloatProperty(
        name="Sharp Angle",
        description="Edges with normals diverging beyond this angle are marked sharp",
        default=75.0, min=0.0, max=180.0, subtype="ANGLE",
    )

    # Animation
    anim_fps: FloatProperty(
        name="Animation FPS",
        description="Override FPS (0 = read from file's AnimTicksPerSecond)",
        default=0.0, min=0.0, max=240.0,
    )
    lock_root_translation: BoolProperty(
        name="Lock Root Translation",
        description=(
            "Pin the root bone's world position to its rest pose for every keyframe. "
            "The character animates in place (feet rooted) rather than walking through "
            "the scene. Matches the behaviour of tools like Fragmotion that strip "
            "root-motion translation from the animation. Disable if you need root motion "
            "for a character controller"
        ),
        default=True,
    )

    lock_leaf_translation: BoolProperty(
        name="Lock Leaf Bone Translation",
        description=(
            "Pin the translation of leaf bones (bones with no children, e.g. foot tips) "
            "to their rest pose. The game engine normally overrides these with IK at "
            "runtime, so locking them matches the planted-feet look seen in fragmotion"
        ),
        default=False,
    )

    # Logging
    verbose_log: BoolProperty(
        name="Verbose Logging",
        description=(
            "Print detailed per-bone/per-mesh DEBUG output to the terminal "
            "you launched Blender from.  INFO/WARN/ERROR always appear."
        ),
        default=False,
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
        box.prop(self, "import_uvs")
        box.prop(self, "import_materials")
        box.prop(self, "import_textures")
        box.prop(self, "infer_sharps")
        row = box.row()
        row.prop(self, "sharp_angle_deg")
        row.enabled = self.infer_sharps

        box = layout.box()
        box.label(text="Armature & Animation", icon="ARMATURE_DATA")
        box.prop(self, "import_armature")
        box.prop(self, "import_weights")
        box.prop(self, "import_animation")
        box.prop(self, "anim_fps")
        box.prop(self, "lock_root_translation")
        box.prop(self, "lock_leaf_translation")

        box = layout.box()
        box.label(text="Diagnostics", icon="INFO")
        box.prop(self, "verbose_log")
        if self.verbose_log:
            box.label(text="Output goes to the launch terminal", icon="CONSOLE")

    def execute(self, context):
        XLog.set_verbose(self.verbose_log)
        _log.section("IMPORT")
        keywords = self.as_keywords(ignore=("filter_glob", "verbose_log"))
        result = import_x(context, **keywords)
        return result


# ─────────────────────────────────────────────────────────────
#  Export operator
# ─────────────────────────────────────────────────────────────
class ExportDirectX(bpy.types.Operator, ExportHelper):
    """Export selected objects as a DirectX .x file"""
    bl_idname  = "export_scene.directx_x"
    bl_label   = "Export DirectX X (.x)"
    bl_options = {"REGISTER"}

    filename_ext = ".x"
    filter_glob: StringProperty(default="*.x", options={"HIDDEN"})

    # Selection
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

    # Transform
    global_scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0)
    axis_forward: EnumProperty(
        name="Forward Axis",
        items=[("-Z","−Z",""),("Z","Z",""),("Y","Y",""),
               ("-Y","−Y",""),("X","X",""),("-X","−X","")],
        default="-Z",
    )
    axis_up: EnumProperty(
        name="Up Axis",
        items=[("Y","Y",""),("-Y","−Y",""),("Z","Z",""),
               ("-Z","−Z",""),("X","X",""),("-X","−X","")],
        default="Y",
    )

    # Data
    export_normals:   BoolProperty(name="Export Normals",   default=True)
    export_uvs:       BoolProperty(name="Export UVs",       default=True)
    export_materials: BoolProperty(name="Export Materials", default=True)
    export_textures:  BoolProperty(name="Export Textures",  default=True)
    export_armature:  BoolProperty(name="Export Armature",  default=True)
    export_weights:   BoolProperty(name="Export Weights",   default=True)
    export_animation: BoolProperty(name="Export Animation", default=True)

    # Animation
    anim_fps:         FloatProperty(name="FPS", default=30.0, min=1.0, max=240.0)
    anim_frame_start: IntProperty(name="Frame Start", default=1)
    anim_frame_end:   IntProperty(name="Frame End",   default=250)

    # Logging
    verbose_log: BoolProperty(
        name="Verbose Logging",
        description="Print detailed DEBUG output to the launch terminal",
        default=False,
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
        box.prop(self, "export_materials")
        box.prop(self, "export_textures")

        box = layout.box()
        box.label(text="Armature & Animation", icon="ARMATURE_DATA")
        box.prop(self, "export_armature")
        box.prop(self, "export_weights")
        box.prop(self, "export_animation")
        row = box.row()
        row.prop(self, "anim_fps")
        row = box.row()
        row.prop(self, "anim_frame_start")
        row.prop(self, "anim_frame_end")

        box = layout.box()
        box.label(text="Diagnostics", icon="INFO")
        box.prop(self, "verbose_log")
        if self.verbose_log:
            box.label(text="Output goes to the launch terminal", icon="CONSOLE")

    def invoke(self, context, event):
        scene = context.scene
        self.anim_frame_start = scene.frame_start
        self.anim_frame_end   = scene.frame_end
        self.anim_fps         = scene.render.fps
        return super().invoke(context, event)

    def execute(self, context):
        XLog.set_verbose(self.verbose_log)
        _log.section("EXPORT")
        keywords = self.as_keywords(ignore=("filter_glob", "check_existing", "verbose_log"))
        return export_x(context, **keywords)


# ─────────────────────────────────────────────────────────────
#  Registration
# ─────────────────────────────────────────────────────────────
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
    # Confirm the addon loaded — visible in terminal even before any import
    XLog("init")._write("[DX.x] addon registered (v1.8.0)")

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
