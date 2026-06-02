# io_directx_x — Blender DirectX `.x` Importer / Exporter

A full-featured Blender addon for the DirectX `.x` file format. Imports and
exports geometry, materials, textures, armatures, skin weights, and keyframe
animation in both directions, across the text, binary, and MSZIP-compressed
variants of the format.

It handles `.x` files from a wide range of sources — general exporters as well
as game-tool exports such as MilkShape 3D, Ultimate Unwrap 3D, Game Guru, and
FPS Creator — plus a dedicated path for DirectX 8.0 rigid-skinned characters.

> **Note:** `.xcache` cache files are no longer handled by this addon. They are
> supported by a separate, dedicated cache addon. This addon is focused purely
> on DirectX `.x`.

---

## Compatibility

| Source | Status |
|---|---|
| General `.x` exporters (text format) | ✅ Full support |
| MilkShape 3D / Ultimate Unwrap 3D `.x` | ✅ |
| Game Guru / FPS Creator `.x` | ✅ |
| DirectX 8.0 rigid-skinned characters | ✅ Dedicated import option |
| High-precision biped `.x` (`AnimTicksPerSecond = 4800`) | ✅ Round-trip preserves DeclData, auxiliary frames, templates |
| DirectX text format (`txt `) | ✅ |
| DirectX binary format (`bin `) | ✅ |
| MS-ZIP compressed text (`tzip`) | ✅ import + export |
| MS-ZIP compressed binary (`bzip`) | ✅ import + export |
| 32-bit and 64-bit float variants | ✅ |
| Blender 3.x / 4.x / 5.x | ✅ |

---

## Installation

1. **Edit → Preferences → Add-ons → Install…**
2. Select the `io_directx_x` folder (zip it first if needed).
3. Enable **"DirectX X Format (.x)"** in the list.

The addon then appears at:

- **File → Import → DirectX X (.x)**
- **File → Export → DirectX X (.x)**

---

## Import Options

### Import Type

A dropdown selects the import path so the right conversion is used rather than
guessed:

| Type | Description |
|---|---|
| **Standard .x** | General DirectX `.x` files from common exporters. The original, well-tested path. |
| **DirectX 8.0 (rigid skin)** | A single skinned mesh with `SkinWeights` / `matrixOffset` bind matrices and rigid (one-bone-per-vertex) binding. |

### Transform

| Option | Default | Description |
|---|---|---|
| Scale | 1.0 | Global scale multiplier applied to all geometry and bone positions |
| Forward Axis | Z | Which Blender axis maps to DX +Z (forward) |
| Up Axis | Y | Which Blender axis maps to DX +Y (up) |
| Apply Transform | On | Bake the root Frame transform matrix into mesh data |

### Data

| Option | Default | Description |
|---|---|---|
| Import Normals | On | Read per-loop split normals from `MeshNormals` (or `DeclData` on biped-style files); infers sharp edges |
| Smooth Shade from Faces | Off | Apply Blender's shade-smooth pass instead of the file's authored per-loop normals. Useful when the file's normals don't render cleanly |
| Import UVs | On | Read the first UV channel from `MeshTextureCoords` or `DeclData` (V-flip corrected) |
| Import Materials | On | Create Principled BSDF materials from `Material` blocks |
| Import Textures | On | Link image textures from `TextureFileName`. Searches the model folder, common subfolders (`Textures/`, `tex/`), parent dirs, and tries `.png`, `.jpg`, `.tga`, `.dds`, `.bmp`, `.tif`, `.webp`. If the file isn't on disk a placeholder image is created so the texture path stays visible and round-trips on export |

Materials no longer import as glossy chrome by default — the format pre-dates
PBR and most exporters write `shininess=128` as a placeholder, so the importer
maps that to a moderate roughness rather than a mirror finish.

### Weld Duplicate Vertices

| Option | Default | Description |
|---|---|---|
| Weld Duplicate Vertices | On | Collapse vertices that sit at the same position into one and blend their bone weights |

The `.x` format stores per-loop normals as split vertices; without welding,
adjacent faces don't share vertices and smooth shading has nothing to average
across, so the mesh looks blocky. Turn it **off** when exact vertex
preservation matters more than shading (e.g. bit-comparing an export against a
reference file).

This toggle is always shown, but has **no effect on the DirectX 8.0 path** —
those meshes are rigidly bound, so welding would only fuse coincident vertices
belonging to different bones (e.g. eyelid pairs) and collapse hard-edge seams.

### Armature & Animation

| Option | Default | Description |
|---|---|---|
| Import Armature | On | Build a Blender armature from the `Frame` hierarchy |
| Import Weights | On | Assign vertex groups from `SkinWeights` blocks |
| Import Animation | On | Create F-curve actions from `AnimationSet` / `AnimationKey` blocks |
| Set Scene Frame Range | On | Sets `frame_start` / `frame_end` to match the animation in the file |

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

### Format

| Format | Extension | Description |
|---|---|---|
| **Text .x** | `.x` | DirectX text format (human-readable) |
| **Binary .x** | `.x` | DirectX binary token format (smaller, faster to parse) |
| **Compressed Text .x** | `.x` | Text format wrapped in MSZIP (`tzip`) |
| **Compressed Binary .x** | `.x` | Binary tokens wrapped in MSZIP (`bzip`) |

Type a filename without an extension and the dropdown decides the format;
typing `model.x` always exports as `.x`.

### Armature & Animation

| Option | Default | Description |
|---|---|---|
| Export Armature | On | Write `Frame` hierarchy from the armature bones |
| Export Weights | On | Write `SkinWeights` blocks from vertex groups |
| Export Animation | On | Bake pose-bone keyframes into `AnimationSet` / `AnimationKey` blocks (sparse keyframes preserved — only frames with actual F-curve keys are emitted) |
| Animation Keys | Quaternion | How animation is written — see below |
| High-precision Animation Ticks | Off | Write `AnimTicksPerSecond = 4800` and scale ticks accordingly, for high-precision biped consumers. Also emits the default DirectX template declarations for engines that require them |
| FPS | 30 | Written as `AnimTicksPerSecond` when High-precision is off |
| Frame Start / End | 1 / 250 | Animation range to bake |

### Animation Keys: Quaternion vs Matrix

| Mode | Description |
|---|---|
| **Quaternion** (default) | Three separate tracks per bone — rotation as a quaternion (keyType 0), scale (keyType 1), position (keyType 2). The most widely compatible option, accepted by the broadest range of importers. |
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
- **Passthrough preservation** — when importing a text-format `.x` file, blocks
  the exporter doesn't natively reproduce (top-level template declarations,
  auxiliary frames, per-mesh `DeclData` and `XSkinMeshHeader`, and Animation
  entries for non-bone frames) are stashed as custom properties and re-emitted
  verbatim on export. Sparse animation keyframe density is preserved on
  round-trip.
- **Damaged-file recovery** — compressed `.x` files whose MSZIP stream is
  corrupted on disk no longer lose all their animation. As much as possible is
  salvaged channel-by-channel, with only the genuinely unreadable parts falling
  back to rest pose. Healthy files are unaffected.

---

## Supported `.x` Blocks

| Block | Import | Export |
|---|---|---|
| `xof 0303txt` / `xof 0303bin` | ✅ | ✅ both |
| `xof 0303tzip` / `xof 0303bzip` (MSZIP-compressed) | ✅ | ✅ both |
| `AnimTicksPerSecond` | ✅ | ✅ |
| `Frame` + `FrameTransformMatrix` | ✅ | ✅ |
| `Mesh` (vertices + N-gon faces) | ✅ | ✅ (triangulation optional) |
| `MeshNormals` | ✅ | ✅ |
| `MeshTextureCoords` (V-flipped) | ✅ | ✅ |
| `DeclData` (per-vertex normals/tangents/UVs) | ✅ | ✅ verbatim on round-trip |
| `VertexElement` (DeclData element table) | ✅ | ✅ |
| `MeshMaterialList` | ✅ | ✅ |
| `Material` (diffuse / shininess / specular / emissive) | ✅ | ✅ |
| `TextureFileName` / `TextureFilename` | ✅ both spellings | ✅ |
| `XSkinMeshHeader` | ✅ | ✅ |
| `SkinWeights` (indices + weights + offset matrix) | ✅ | ✅ |
| `AnimationSet` / `Animation` | ✅ | ✅ |
| `AnimationKey` type 0 — rotation quaternion | ✅ | ✅ |
| `AnimationKey` type 1 — scale | ✅ | ✅ |
| `AnimationKey` type 2 — position | ✅ | ✅ |
| `AnimationKey` type 4 — full 4×4 matrix | ✅ | ✅ (Matrix mode) |

---

## File Layout

```
io_directx_x/
├── __init__.py   — Blender operator registration and UI panels
├── parser.py     — Tokenizer and recursive-descent parser → XNode tree
│                   (handles .x text, binary, and MSZIP-compressed)
├── importer.py   — XNode tree → Blender objects, armature, and animation
└── exporter.py   — Blender scene → .x file
```

`parser.parse_x_file(path)` is the single entry point — it auto-detects the
`.x` sub-format (text / binary / compressed) by magic bytes.
