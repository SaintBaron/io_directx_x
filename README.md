# io_directx_x — Blender DirectX `.x` and Bugsnax `.xcache` Importer / Exporter

A full-featured Blender addon for the DirectX `.x` file format, with first-class
support for the **Horsepower engine `.xcache`** files used by Bugsnax. Validated
against fragMOTION-exported `.x` files (`xof 0303txt 0032`) and Bugsnax SEMS
`.xcache` files. Supports geometry, materials, textures, armatures, skin
weights, and keyframe animation in both directions.

---

## Compatibility

| Source | Status |
|---|---|
| fragMOTION exports (text format) | ✅ Full support |
| Bugsnax / Horsepower `.xcache` (SEMS) | ✅ Full round-trip (mesh + skeleton + animation + textures) |
| Project Zomboid / 3DS Max biped (`.x`) | ✅ Round-trip preserves DeclData, Translation_Data, templates, AnimTicksPerSecond = 4800 |
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
- **File → Import → DirectX X (.x)** — handles both `.x` and `.xcache`
- **File → Export → DirectX X (.x)** — produces any of five formats (see below)

---

## Import Options

The importer auto-detects the file format by magic bytes (`xof ` for DirectX,
`SEMS` for `.xcache`), so a single import action handles both.

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
| Import Normals | On | Read per-loop split normals from `MeshNormals` (or `DeclData` on PZ / 3DS Max biped files); infers sharp edges |
| Smooth Shade from Faces | Off | Apply Blender's shade-smooth pass instead of using the file's authored per-loop normals. Useful when the file's normals don't render cleanly |
| Import UVs | On | Read the first UV channel from `MeshTextureCoords` or `DeclData` (V-flip corrected) |
| Import Materials | On | Create Principled BSDF materials from `Material` blocks |
| Import Textures | On | Link image textures from `TextureFileName`. Searches the xcache folder, common subfolders (`Textures/`, `tex/`), parent dirs, and tries `.png`, `.jpg`, `.tga`, `.dds`, `.bmp`, `.tif`, `.webp` extensions. If the file isn't on disk a placeholder image is created so the texture path is still visible in Blender and round-trips on export |

`.x` and `.xcache` materials no longer import as glossy chrome by default —
the format pre-dates PBR and most exporters write `shininess=128` as a
placeholder, so the importer now maps that to a moderate roughness rather than
mirror.

### Armature & Animation
| Option | Default | Description |
|---|---|---|
| Import Armature | On | Build a Blender armature from the `Frame` hierarchy |
| Import Weights | On | Assign vertex groups from `SkinWeights` blocks |
| Import Animation | On | Create F-curve actions from `AnimationSet` / `AnimationKey` blocks |
| Rest Pose Source | Bind Pose | Where bone rest positions come from — see below |
| Animation FPS | 0 (auto) | 0 reads `AnimTicksPerSecond` from the file; any other value overrides it |
| Set Scene Frame Range | On | Sets `frame_start` / `frame_end` to exactly match the animation data in the file |

### Rest Pose Source

Two modes control how bone rest matrices are computed:

**Bind Pose** (default) — uses the `SkinWeights` offset matrices
(`inv(offset) = bind pose`). Joint positions are geometrically correct and match
what the artist authored. Some exporters (including fragMOTION's Maya plugin)
bake a 180° root rotation into the offset matrix, which can make the armature
face the wrong direction in edit mode — animation playback is unaffected.

**Frame Hierarchy** — uses the accumulated `FrameTransformMatrix` chain,
matching fragMOTION's own viewport behaviour. The armature and mesh face the
same direction in edit mode. For files where the FTM encodes an animated pose
rather than the bind pose, limbs may appear crunched at rest; mesh vertices are
automatically rebound to compensate.

### Diagnostics
| Option | Default | Description |
|---|---|---|
| Verbose Logging | Off | Prints per-bone and per-mesh DEBUG output to the terminal Blender was launched from; INFO / WARN / ERROR always appear |

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
| Use Original Material Data | Off | When on, exports the values stored at import time (the `_x_*` custom properties) rather than the current Blender material state |

A panel in the export dialog lists every material that has a stored texture
path and lets you edit those paths before export without opening the material
editor.

### Format

The format dropdown has five options:

| Format | Extension | Description |
|---|---|---|
| **Text .x** | `.x` | DirectX text format (human-readable) |
| **Binary .x** | `.x` | DirectX binary token format (smaller, faster to parse) |
| **Compressed Text .x** | `.x` | Text format wrapped in MSZIP (`tzip`) |
| **Compressed Binary .x** | `.x` | Binary tokens wrapped in MSZIP (`bzip`) |
| **Bugsnax .xcache** | `.xcache` | Horsepower SEMS format — full mesh + skeleton + animation + textures |

The filename you type **overrides** the dropdown when there's a clear
conflict — typing `model.xcache` always exports as `.xcache` regardless of the
dropdown setting, and typing `model.x` always exports as `.x`. Type a filename
without an extension and the dropdown decides.

### Armature & Animation
| Option | Default | Description |
|---|---|---|
| Export Armature | On | Write `Frame` hierarchy from the armature bones |
| Export Weights | On | Write `SkinWeights` blocks from vertex groups |
| Export Animation | On | Bake pose-bone keyframes into `AnimationSet` / `AnimationKey` blocks (sparse keyframes preserved — only frames with actual F-curve keys are emitted) |
| High-precision Animation Ticks | Off | Write `AnimTicksPerSecond = 4800` and scale ticks accordingly. Required for Project Zomboid and other 3DS Max biped consumers. Also emits the default DirectX template declarations for engines that require them |
| FPS | 30 | Written as `AnimTicksPerSecond` when High-precision is off (`.x` only — `.xcache` uses a fixed 25.0) |
| Frame Start / End | 1 / 250 | Animation range to bake |

### Non-manifold warning

If any exported mesh contains non-manifold edges (edges shared by anything
other than exactly two faces — open borders, internal faces, etc.), a `WARN`
is printed to the log listing the affected edge indices. Non-manifold geometry
typically produces broken normals and bad skinning in DirectX consumers and
game engines, so it is worth resolving before export.

---

## Round-trip notes

- **Import → re-export** is rotation-identity for both `.x` and `.xcache`.
  Default axes (Forward = Z, Up = Y on both sides) compose with the addon's
  internal axis-fix so the model lands back in the same orientation.
- **`.xcache` → `.x` round-trip** also works: the importer treats `.xcache` as
  a parsed XNode tree, so any export format works on imported `.xcache`
  content.
- **UV seams** are preserved on `.xcache` export by duplicating vertices that
  carry multiple distinct UVs across their loops. Skin weights are replicated
  onto every duplicate so animation keeps deforming correctly.
- **Passthrough preservation** — when importing a text-format `.x` file,
  blocks the exporter doesn't natively reproduce (top-level template
  declarations, auxiliary frames like Project Zomboid's `Translation_Data`,
  per-mesh `DeclData` and `XSkinMeshHeader`, and Animation entries for
  non-bone frames) are stashed as custom properties and re-emitted verbatim
  on export. Sparse animation keyframe density (e.g. 41/2/33 R/S/T) is also
  preserved on round-trip.

---

## Supported `.x` Blocks

| Block | Import | Export |
|---|---|---|
| `xof 0303txt` / `xof 0303bin` | ✅ | ✅ both |
| `xof 0303tzip` / `xof 0303bzip` (MSZIP-compressed) | ✅ | ✅ both |
| `AnimTicksPerSecond` | ✅ | ✅ |
| `Frame` + `FrameTransformMatrix` | ✅ | ✅ |
| `Mesh` (vertices + N-gon faces) | ✅ | ✅ (triangulated optional) |
| `MeshNormals` | ✅ | ✅ |
| `MeshTextureCoords` (V-flipped) | ✅ | ✅ |
| `DeclData` (per-vertex normals/tangents/UVs, PZ / 3DS Max biped) | ✅ | ✅ verbatim on round-trip |
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
| `AnimationKey` type 4 — full 4×4 matrix | ✅ | — |

## Supported `.xcache` features

| Feature | Import | Export |
|---|---|---|
| SEMS header + bone-count detection | ✅ | ✅ |
| Bone hierarchy with relative parent indices | ✅ | ✅ |
| Bind-pose 4×4 matrices | ✅ | ✅ |
| FrameTransformMatrix per bone | ✅ | ✅ |
| Animation channels (pos / scale / rot keyframes) | ✅ | ✅ |
| Skin weights per bone | ✅ | ✅ |
| Mesh blocks with vertex/normal/UV streams | ✅ | ✅ |
| Index buffer | ✅ | ✅ |
| Texture path entries (D / N / S maps) | ✅ | ✅ |
| Multi-mesh files | ✅ | ✅ |

---

## File Layout

```
io_directx_x/
├── __init__.py   — Blender operator registration and UI panels
├── parser.py     — Tokenizer and recursive-descent parser → XNode tree
│                   (handles .x text, binary, compressed, and .xcache)
├── importer.py   — XNode tree → Blender objects, armature, and animation
└── exporter.py   — Blender scene → .x or .xcache file
```

`parser.parse_x_file(path)` is the single entry point — it auto-routes to the
right parser by magic bytes (`xof ` vs `SEMS`).
