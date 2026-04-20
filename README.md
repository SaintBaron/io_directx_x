# io_directx_x — Blender DirectX .x Importer / Exporter

A full-featured Blender addon for the DirectX text-format `.x` file,
built from and validated against **Burger.x** (xof 0303txt 0032).

---

## Feature Matrix

| Feature | Import | Export |
|---|---|---|
| Geometry (vertices, faces) | ✅ | ✅ (triangulated) |
| Split normals | ✅ | ✅ |
| UV coordinates (V-flip corrected) | ✅ | ✅ |
| Materials (diffuse, shininess, specular) | ✅ | ✅ |
| Textures (DDS/PNG/JPG/TGA fallback) | ✅ | ✅ |
| Frame hierarchy → Armature | ✅ | ✅ |
| FrameTransformMatrix → Bone rest pose | ✅ | ✅ |
| SkinWeights → Vertex Groups | ✅ | ✅ |
| Bind-pose offset matrix | ✅ | ✅ |
| AnimationSet / AnimationKey (rot/scale/pos) | ✅ | ✅ |
| AnimTicksPerSecond | ✅ | ✅ |
| Axis conversion (Forward / Up) | ✅ | ✅ |
| Global scale | ✅ | ✅ |
| Apply mesh modifiers on export | — | ✅ |
| Selected-only export | — | ✅ |

---

## Installation

1. In Blender: **Edit → Preferences → Add-ons → Install…**
2. Select the **`io_directx_x`** folder (zip it first if needed).
3. Enable **"DirectX X Format (.x)"** in the list.

The addon then appears at:
- **File → Import → DirectX X (.x)**
- **File → Export → DirectX X (.x)**

---

## Import Options

| Option | Default | Description |
|---|---|---|
| Scale | 1.0 | Global scale multiplier |
| Forward / Up Axis | −Z / Y | Coordinate system |
| Apply Transform | On | Bake root Frame matrix into mesh data |
| Import Normals | On | Custom split-normal data |
| Import UVs | On | First UV channel |
| Import Materials | On | Blender Principled BSDF materials |
| Import Textures | On | Linked image textures |
| Import Armature | On | Frame hierarchy → bones |
| Import Weights | On | SkinWeights → vertex groups |
| Import Animation | On | AnimationKey keyframes |
| Animation FPS | 0 (auto) | 0 = read from AnimTicksPerSecond |

---

## Export Options

| Option | Default | Description |
|---|---|---|
| Selected Only | Off | Only export selected objects |
| Apply Modifiers | On | Bake modifiers before export |
| Scale | 1.0 | Global scale |
| Forward / Up Axis | −Z / Y | Coordinate system |
| Export Normals | On | Per-loop split normals |
| Export UVs | On | Active UV layer |
| Export Materials | On | Principled BSDF → Material block |
| Export Textures | On | Linked image → TextureFileName |
| Export Armature | On | Bones → Frame hierarchy |
| Export Weights | On | Vertex groups → SkinWeights |
| Export Animation | On | Pose bone keyframes |
| FPS | 30 | AnimTicksPerSecond value |
| Frame Start/End | 1 / 250 | Animation range to bake |

---

## File Structure Produced on Export

```
xof 0303txt 0032

AnimTicksPerSecond { 30; }

Material mat_name { … TextureFileName { "…"; } }

Frame BoneName {
    FrameTransformMatrix { … }
    Frame ChildBone { … }
}

Frame MeshName {
    FrameTransformMatrix { … }
    Mesh MeshNameGeo {
        <vertices>
        <faces>
        MeshNormals { … }
        MeshTextureCoords { … }
        MeshMaterialList { … }
        XSkinMeshHeader { … }
        SkinWeights { "BoneName"; … }
    }
}

AnimationSet anim {
    Animation {
        { BoneName }
        AnimationKey { 0; … }   // rotation quaternion
        AnimationKey { 1; … }   // scale
        AnimationKey { 2; … }   // position
    }
}
```

---

## File Layout

```
io_directx_x/
├── __init__.py   — Blender operator registration + UI panels
├── parser.py     — Tokenizer + recursive-descent parser → XNode tree
├── importer.py   — XNode tree → Blender objects/armature/animation
└── exporter.py   — Blender scene → .x text file
```

---

## Supported .x Tokens

| Block | Notes |
|---|---|
| `xof 0303txt 0032` | Text-format header (binary not supported) |
| `AnimTicksPerSecond` | Sets scene FPS |
| `Frame` | Recursive; becomes armature bone |
| `FrameTransformMatrix` | 16-float row-major matrix |
| `Mesh` | Vertices + N-gon faces (import) / triangles (export) |
| `MeshNormals` | Per-face-corner normals |
| `MeshTextureCoords` | Per-vertex UV (V-flipped) |
| `MeshMaterialList` | Per-face material index |
| `Material` | Diffuse RGBA, shininess, specular, emissive |
| `TextureFileName` | Relative path; tries .dds/.png/.jpg/.tga |
| `XSkinMeshHeader` | Influence count metadata |
| `SkinWeights` | Per-bone vertex index + weight + offset matrix |
| `AnimationSet` | Named animation group |
| `Animation` | Per-bone keyframes with `{ BoneRef }` |
| `AnimationKey` type 0 | Quaternion rotation (w x y z) |
| `AnimationKey` type 1 | Scale (x y z) |
| `AnimationKey` type 2 | Position (x y z) |
| `AnimationKey` type 4 | Full 4×4 matrix key |
