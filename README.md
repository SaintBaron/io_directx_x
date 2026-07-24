# io_directx_x

A Blender addon for the DirectX `.x` file format. Imports and exports
geometry, materials, textures, armatures, skin weights, and keyframe animation
in both directions, across the text, binary, and MSZIP-compressed variants of
the format.

There is a separate import path for DirectX 8.0 rigid-skinned characters.

---

## Format support

| Variant | Import | Export |
|---|---|---|
| Text (`txt `) | Yes | Yes |
| Binary (`bin `) | Yes | Yes |
| MSZIP-compressed text (`tzip`) | Yes | Yes |
| MSZIP-compressed binary (`bzip`) | Yes | Yes |
| 32-bit and 64-bit float variants | Yes | Yes |
| High-precision biped files (`AnimTicksPerSecond = 4800`) | Yes | Yes, preserving DeclData, auxiliary frames and templates |

Requires Blender 4.2 or newer.

---

## Supported games & tools

Formats and conventions from the following have been validated against real
files (regression corpora and/or in-engine load tests):

| Game / tool | Direction | Notes |
|---|---|---|
| FPS Creator / FPSC Classic | import + export | Characters (incl. DirectX 8.0 rigid-skin "_L" biped rigs), entities, HUDs |
| Dark BASIC Professional | import + export | Same loader family as FPSC; text and binary |
| Game Guru (Classic) | import + export | FPSC-lineage assets |
| Bugsnax (Horsepower Engine `.x` interchange) | import + export | Byte-identical round-trips |
| Project Zomboid | import + export | Character/animation `.x`; verified loading in-game |
| Virtual Sailor NG | import + export | Binary `bin 0032` vessels; inline-material and zero-UV strict options |
| MilkShape 3D | import | DirectX 8.0 rigid-skin exports |
| Ultimate Unwrap 3D | import | Standard `.x` output |
| kwXport (3ds Max) | import | Text and binary |
| conv3ds / legacy 3D Studio pipelines | import | Including scaled-frame scene files |

Other exporters producing standard `.x` generally work. The list above is
what has test coverage.

---

## Installation

Requires **Blender 4.2 or newer**.

- **Extension zip** (recommended): drag `io_directx_x-extension.zip` into
  Blender, or **Edit → Preferences → Get Extensions → ⌄ → Install from
  Disk…** and select it.
- **Legacy add-on zip**: **Edit → Preferences → Add-ons → ⌄ → Install from
  Disk…** with `io_directx_x-release.zip`, then enable
  **"DirectX X Format (.x)"**.

The addon then appears at:

- **File → Import → DirectX X (.x)**
- **File → Export → DirectX X (.x)**

---

## Import Options

### Transform

| Option | Default | Description |
|---|---|---|
| Scale | 1.0 | Global scale multiplier applied to all geometry and bone positions |
| Forward Axis | Z | Which Blender axis maps to DX +Z (forward) |
| Up Axis | Y | Which Blender axis maps to DX +Y (up) |
| Apply Transform | On | Bake the root Frame transform matrix into mesh data |

### DirectX Format

The switches that control how the `.x` file's own conventions are interpreted,
grouped in one box:

| Option | Default | Description |
|---|---|---|
| **DirectX 8.0 (rigid skin)** | off | Dedicated import path for DirectX 8.0-style files: a single skinned mesh with `SkinWeights` / `matrixOffset` bind matrices and rigid (one-bone-per-vertex) binding. Leave off for general `.x` files. |
| **Rest Pose from First Frame** | off | Use the first animation frame as the rest pose instead of the bind pose. |
| **Flatten Mirrored Frames** | on | Fold mirrored / scaled frame hierarchies into geometry so Blender bones (which cannot store scale or reflection at rest) reproduce the file's worlds. |
| **Weld Duplicate Vertices** | on | Collapse the format's split vertices back into shared verts for smooth shading (inert on the DirectX 8.0 path, where welding is suppressed). |
| **Split Sub-Meshes** | on | Split multi-material meshes into one Blender object per material. |
| **Triangulate Quads** | on | Triangulate quad faces at parse time. |

About welding: The `.x` format stores per-loop normals as split vertices; without welding,
adjacent faces don't share vertices and smooth shading has nothing to average
across, so the mesh looks blocky. Turn it **off** when exact vertex
preservation matters more than shading (e.g. bit-comparing an export against a
reference file).

This toggle is always shown, but has no effect on the DirectX 8.0 path:
those meshes are rigidly bound, so welding would only fuse coincident vertices
belonging to different bones (e.g. eyelid pairs) and collapse hard-edge seams.


### Data

| Option | Default | Description |
|---|---|---|
| Import Normals | On | Read per-loop split normals from `MeshNormals` (or `DeclData` on biped-style files); infers sharp edges |
| Smooth Shade from Faces | Off | Apply Blender's shade-smooth pass instead of the file's authored per-loop normals. Useful when the file's normals don't render cleanly |
| Import UVs | On | Read the first UV channel from `MeshTextureCoords` or `DeclData` (V-flip corrected) |
| Import Materials | On | Create Principled BSDF materials from `Material` blocks |
| Import Textures | On | Link image textures from `TextureFileName`. Searches the model folder, common subfolders (`Textures/`, `tex/`), parent dirs, and tries `.png`, `.jpg`, `.tga`, `.dds`, `.bmp`, `.tif`, `.webp`. If the file isn't on disk a placeholder image is created so the texture path stays visible and round-trips on export |
| Use Diffuse Alpha | Off | Wire the material's diffuse alpha into the shader (many `.x` files carry face-color alpha that is not meant as transparency) |

Materials do not import as glossy chrome. The format pre-dates PBR and most
exporters write `shininess=128` as a placeholder, so the importer maps that to
a moderate roughness rather than a mirror finish.

### Armature & Animation

| Option | Default | Description |
|---|---|---|
| Import Armature | On | Build a Blender armature from the `Frame` hierarchy |
| Import Weights | On | Assign vertex groups from `SkinWeights` blocks |
| Import Animation | On | Create F-curve actions from `AnimationSet` / `AnimationKey` blocks |
| Set Scene Frame Range | On | Sets `frame_start` / `frame_end` to match the animation in the file |

### Scene

| Option | Default | Description |
|---|---|---|
| Group in Collection | On | Put everything from one file into its own collection named after the file |

### How character binding works

Skinned models bind using the mesh's own bind-pose data (the `SkinWeights`
offset matrices, where `inv(offset)` is the bind pose) as the source of truth
for where each bone binds, while the `Frame` hierarchy sets the rest
orientation. Bones that have no skin data fall back to the frame hierarchy.
This produces geometrically correct joint positions matching what the artist
authored.

---

## Export Options

### Include

| Option | Default | Description |
|---|---|---|
| Selected Only | Off | Export only selected objects; off exports the full scene |
| Apply Modifiers | On | Bake mesh modifiers before export |

### Transform

| Option | Default | Description |
|---|---|---|
| Scale | 1.0 | Global scale |
| Forward / Up Axis | Z / Y | Coordinate system for the output file. Defaults match the importer for round-trip identity. |

### DirectX Format

All DirectX-specific output settings live in one box: the file format below,
plus **Write Template Declarations** (on by default; strict loaders such as
FPS Creator / Dark Basic and Virtual Sailor NG reject files whose objects use
undeclared templates), **Animation Keys** (quaternion TRS vs 4x4 matrix),
**High-precision animation ticks** (the 4800-ticks-per-second biped
convention), **Inline Materials in Mesh** and **Write Zero UVs When Missing**
(both matching native Virtual Sailor output).

| Format | Extension | Description |
|---|---|---|
| **Text .x** | `.x` | DirectX text format (human-readable) |
| **Binary .x** | `.x` | DirectX binary token format (smaller, faster to parse) |
| **Compressed Text .x** | `.x` | Text format wrapped in MSZIP (`tzip`) |
| **Compressed Binary .x** | `.x` | Binary tokens wrapped in MSZIP (`bzip`) |

Type a filename without an extension and the dropdown decides the format;
typing `model.x` always exports as `.x`.

### Data

| Option | Default | Description |
|---|---|---|
| Export Normals | On | Write per-loop split normals to `MeshNormals` |
| Export UVs | On | Write the active UV layer to `MeshTextureCoords` (V-flip applied) |
| Export Materials | On | Write `Material` blocks with diffuse, shininess, specular, and emissive values |
| Export Textures | On | Write `TextureFileName` inside each material block |
| Use Original Material Data | Off | Export the values stored at import time (the `_x_*` custom properties) rather than the current Blender material state |
| Unweld on Export | On | Split face-loops back into separate vertices to restore the original file's vertex count for round-trip fidelity |

A panel in the export dialog lists every material that has a stored texture
path and lets you edit those paths before export without opening the material
editor.

### Armature & Animation

| Option | Default | Description |
|---|---|---|
| Export Armature | On | Write `Frame` hierarchy from the armature bones |
| Export Weights | On | Write `SkinWeights` blocks from vertex groups |
| Export Animation | On | Bake pose-bone keyframes into `AnimationSet` / `AnimationKey` blocks (sparse keyframes preserved: only frames with actual F-curve keys are emitted) |
| Animation Keys | Quaternion | How animation is written; see below |
| High-precision Animation Ticks | Off | Write `AnimTicksPerSecond = 4800` and scale ticks accordingly, for high-precision biped consumers. Also emits the default DirectX template declarations for engines that require them |
| FPS | 30 | Written as `AnimTicksPerSecond` when High-precision is off |
| Frame Start / End | 1 / 250 | Animation range to bake |

### Animation Keys: Quaternion vs Matrix

| Mode | Description |
|---|---|
| **Quaternion** (default) | Three separate tracks per bone: rotation as a quaternion (keyType 0), scale (keyType 1), position (keyType 2). The most widely compatible option, accepted by the broadest range of importers. |
| **Matrix** | One 4×4 transform per frame per bone (keyType 4). The most faithful round-trip for files authored with matrix keys (no decomposition step) and preserves transforms a quaternion+scale split can't represent, such as shear. |

### Non-manifold warning

If any exported mesh contains non-manifold edges (edges shared by anything
other than exactly two faces), a `WARN` is printed to the log listing the
affected edge indices. Non-manifold geometry typically produces broken normals
and bad skinning in DirectX consumers, so it's worth resolving before export.

---

## Round-trip notes

- **Import → re-export** is rotation-identity. Default axes (Forward = Z,
  Up = Y on both sides) compose with the addon's internal axis-fix so the model
  lands back in the same orientation.
- **UV seams** are preserved by duplicating vertices that carry multiple
  distinct UVs across their loops, with skin weights replicated onto every
  duplicate so animation keeps deforming correctly.
- **Passthrough preservation.** When importing a text-format `.x` file, blocks
  the exporter doesn't natively reproduce (top-level template declarations,
  auxiliary frames, per-mesh `DeclData` and `XSkinMeshHeader`, and Animation
  entries for non-bone frames) are stashed as custom properties and re-emitted
  verbatim on export. Sparse animation keyframe density is preserved on
  round-trip.
- **Damaged-file recovery.** Compressed `.x` files whose MSZIP stream is
  corrupted on disk no longer lose all their animation. As much as possible is
  salvaged channel-by-channel, with only the genuinely unreadable parts falling
  back to rest pose. Healthy files are unaffected.

---

## Supported `.x` Blocks

| Block | Import | Export |
|---|---|---|
| `xof 0303txt` / `xof 0303bin` | Yes | Yes |
| `xof 0303tzip` / `xof 0303bzip` (MSZIP-compressed) | Yes | Yes |
| `AnimTicksPerSecond` | Yes | Yes |
| `Frame` + `FrameTransformMatrix` | Yes | Yes |
| `Mesh` (vertices + N-gon faces) | Yes | Yes (triangulation optional) |
| `MeshNormals` | Yes | Yes |
| `MeshTextureCoords` (V-flipped) | Yes | Yes |
| `DeclData` (per-vertex normals/tangents/UVs) | Yes | Preserved verbatim on round-trip |
| `VertexElement` (DeclData element table) | Yes | Yes |
| `MeshMaterialList` | Yes | Yes |
| `Material` (diffuse / shininess / specular / emissive) | Yes | Yes |
| `TextureFileName` / `TextureFilename` | Yes, both spellings | Yes |
| `XSkinMeshHeader` | Yes | Yes |
| `SkinWeights` (indices + weights + offset matrix) | Yes | Yes |
| `AnimationSet` / `Animation` | Yes | Yes |
| `AnimationKey` type 0 (rotation quaternion) | Yes | Yes |
| `AnimationKey` type 1 (scale) | Yes | Yes |
| `AnimationKey` type 2 (position) | Yes | Yes |
| `AnimationKey` type 4 (full 4x4 matrix) | Yes | Yes (Matrix mode) |

---

## File Layout

```
io_directx_x/
├── __init__.py   # Blender operator registration and UI panels
├── parser.py     # Tokenizer and recursive-descent parser → XNode tree
│                   (handles .x text, binary, and MSZIP-compressed)
├── importer.py   # XNode tree → Blender objects, armature, and animation
└── exporter.py   # Blender scene → .x file
```

`parser.parse_x_file(path)` is the single entry point. It auto-detects the
`.x` sub-format (text / binary / compressed) by magic bytes.


## Known limitations

- **HUD.X-style baked-Group constructs** re-export with a ~0.9-unit residual
  (the import itself is exact); the construct is pathological and no other
  corpus file uses it.
- **Blender's "Rest Position" toggle** can show a small seam offset on a few
  neck vertices of characters whose meshes bind the same bones with different
  per-mesh offsets. The animated result, which is what the engine shows, is
  exact;
  one shared armature cannot make both displays exact when the offsets
  disagree.
- **Unnamed source meshes split by material don't re-merge on export**
  (nothing exists to key the merge); geometry and materials are still correct,
  simply as separate objects.
- The brace scanner used for round-trip passthrough does not skip braces
  inside quoted strings; a `{` or `}` inside a texture filename would miscount.
  No known file does this.

## Version

Current release: **1.4.0**. Development history lives in the repository's
commit log rather than this file.
