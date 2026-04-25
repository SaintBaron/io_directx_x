# io_directx_x — Blender DirectX .x Importer / Exporter

A full-featured Blender addon for the DirectX `.x` file format. Validated against
**Bugsnax** models (fragMOTION-exported `.x` files, `xof 0303txt 0032`) and compatible
with the broader fragMOTION ecosystem. Supports geometry, materials, textures,
armatures, skin weights, and keyframe animation on import and export.

---

## Compatibility

| Source | Status |
|---|---|
| fragMOTION exports (text format) | ✅ |
| Bugsnax character models | ✅ |
| DirectX binary format (`bin `) | ✅ |
| MS-ZIP compressed (`tzip` / `bzip`) | ✅ |
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
| Import Normals | On | Read per-loop split normals from `MeshNormals`; infers sharp edges from them |
| Import UVs | On | Read the first UV channel from `MeshTextureCoords` (V-flip corrected) |
| Import Materials | On | Create Principled BSDF materials from `Material` blocks |
| Import Textures | On | Link image textures from `TextureFileName`; tries `.png`, `.jpg`, `.tga`, `.dds` fallbacks |

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

**Bind Pose** (default) — uses the `SkinWeights` offset matrices (`inv(offset) = bind pose`).
Joint positions are geometrically correct and match what the artist authored. Some exporters
(including fragMOTION's Maya plugin) bake a 180° root rotation into the offset matrix, which
can make the armature face the wrong direction in edit mode — animation playback is unaffected.

**Frame Hierarchy** — uses the accumulated `FrameTransformMatrix` chain, matching fragMOTION's
own viewport behaviour. The armature and mesh face the same direction in edit mode. For files
where the FTM encodes an animated pose rather than the bind pose, limbs may appear crunched
at rest; mesh vertices are automatically rebound to compensate.

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
| Forward / Up Axis | −Z / Y | Coordinate system for the output file |

### Data
| Option | Default | Description |
|---|---|---|
| Export Normals | On | Write per-loop split normals to `MeshNormals` |
| Export UVs | On | Write the active UV layer to `MeshTextureCoords` (V-flip applied) |
| Export Materials | On | Write `Material` blocks with diffuse, shininess, specular, and emissive values |
| Export Textures | On | Write `TextureFileName` inside each material block |
| Use Original Material Data | Off | When on, exports the values stored at import time (the `_x_*` custom properties) rather than the current Blender material state |

A panel in the export dialog lists every material that has a stored texture path and lets
you edit those paths before export without opening the material editor.

### Format
| Option | Default | Description |
|---|---|---|
| Binary Format | Off | Write DirectX binary token format (`bin `) instead of text (`txt `). Smaller and faster to parse but not human-readable |

### Armature & Animation
| Option | Default | Description |
|---|---|---|
| Export Armature | On | Write `Frame` hierarchy from the armature bones |
| Export Weights | On | Write `SkinWeights` blocks from vertex groups |
| Export Animation | On | Bake pose-bone keyframes into `AnimationSet` / `AnimationKey` blocks |
| FPS | 30 | Written as `AnimTicksPerSecond` |
| Frame Start / End | 1 / 250 | Animation range to bake |

### Non-manifold warning
If any exported mesh contains non-manifold edges (edges shared by anything other than
exactly two faces — open borders, internal faces, etc.), a `WARN` is printed to the log
listing the affected edge indices. Non-manifold geometry typically produces broken normals
and bad skinning in DirectX consumers and game engines, so it is worth resolving before
export.

---

## Supported .x Blocks

| Block | Import | Export |
|---|---|---|
| `xof 0303txt` / `xof 0303bin` | ✅ | ✅ text + ✅ binary |
| `AnimTicksPerSecond` | ✅ | ✅ |
| `Frame` + `FrameTransformMatrix` | ✅ | ✅ |
| `Mesh` (vertices + N-gon faces) | ✅ | ✅ (triangulated optional) |
| `MeshNormals` | ✅ | ✅ |
| `MeshTextureCoords` (V-flipped) | ✅ | ✅ |
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
| MS-ZIP decompression (`tzip` / `bzip`) | ✅ | — |

---

## File Layout

```
io_directx_x/
├── __init__.py   — Blender operator registration and UI panels
├── parser.py     — Tokenizer and recursive-descent parser → XNode tree
├── importer.py   — XNode tree → Blender objects, armature, and animation
├── exporter.py   — Blender scene → .x text or binary file
```
