# FRep Designer 4.0 — Tutorial: Build Your First Scene

This tutorial walks through building a complete F-Rep scene from scratch.
By the end you will have created a procedural sculpture that combines
CSG primitives, deformations, custom math expressions, and textured
PBR materials, then exported the result as both a rendered image and
a 3D-printable mesh.

We assume you have built the project (see [README.md](../README.md)).
All steps reference the `frep_designer` GUI; the same operations are
available via JSON scene files for the `frep_multipath --paths gpu_glsl`
command-line workflow.

---

## Step 1 — Start with an empty scene

Launch the GUI from the build directory:

```bash
./build/frep_designer --empty
```

You will see four side-panel tabs (Scene, Render, Expression, Material,
Lights, Node Graph) and a black viewport. The scene starts empty.

## Step 2 — Add a base primitive

From the **Primitives toolbar** at the top, click **Sphere**. A
unit-radius sphere appears in the viewport. In the **Scene tab** the
new object shows up with an auto-generated id like `obj_0`.

> 💡 The toolbar primitives all have sensible defaults. You can tweak
> their parameters by selecting the object in the Scene tab and
> entering values in the parameter list below the object list.

## Step 3 — Carve a hole with CSG

We want to remove a smaller sphere from the middle of the first one.
The cleanest way is the **Node Graph** tab:

1. Click the **Node Graph** tab.
2. Right-click in the empty area → **Add Node → Sphere**. A new
   `Sphere` node appears.
3. Right-click → **Add Node → Difference**.
4. Drag from the first Sphere's output port to the Difference's first
   (left) input, and the second Sphere's output to the Difference's
   second (right) input.
5. Drag the Difference's output to the **Output** node.

The viewport immediately recompiles and renders the carved sphere. The
node graph also has an **Editing:** dropdown at the top — that's the
scene object the graph is currently representing.

> 💡 If you click around in the viewport and lose your editing focus,
> uncheck the **Follow** checkbox in the graph toolbar to lock the
> graph view to your current object.

## Step 4 — Add a twist deformation

Pure-CSG models look like flat geometry. Adding a deformation makes
the scene feel sculpted:

1. In the node graph, right-click → **Add Node → Twist Y**.
2. Re-wire: drop the Twist node *between* the Difference and the
   Output. Right-click an existing edge to delete it, then re-connect.
3. The Twist node has a parameter `k` (twist rate). Set it to about
   `1.0`. The viewport now shows your carved sphere with a vertical
   twist running through it.

## Step 5 — Add a procedural surface via expression

Sometimes the surface you want is mathematically simpler to write
than to construct from primitives. The Expression editor parses
analytic SDF expressions in real time:

1. Switch to the **Expression** tab.
2. Click the **Sample** dropdown and pick **Gyroid** — the editor
   fills in `sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)`.
3. Set the object id to something memorable like `gyroid_lattice`.
4. Click **Add to scene**.

The gyroid surface joins your twisted-carved sphere in the viewport
(positioned at the origin; you may want to translate it).

> 💡 The Expression editor validates syntax live. Errors show up
> immediately, including a column number so you can find typos in
> long expressions. Constants `pi` and `e` are supported, as are
> the standard math functions (`sin`, `cos`, `sqrt`, `pow`, `min`,
> `max`, etc.). See [USER_GUIDE.md](USER_GUIDE.md) for the full list.

## Step 6 — Apply a texture

The default material is a flat grey solid. Let's give the gyroid a
wood-grain texture:

1. With `gyroid_lattice` selected in the Scene tab, switch to the
   **Material** tab.
2. Change **Pattern** from `Solid` to `Texture`.
3. Click **Browse…** next to Texture, navigate to
   `examples/textures/wood.bmp`, and select it.
4. The viewport now shows the gyroid surface wrapped in wood texture.

> 💡 The bundled textures are procedurally generated at build time —
> see `examples/textures/`. Drop any PNG or BMP in there and the file
> picker will find it.

## Step 7 — Tune the lighting

Default scenes have a single warm sun light. For a more sculptural
look, add a cool fill:

1. Switch to the **Lights** tab.
2. Click **Add light**. A new light appears at `(0, 6, 4)` with
   default white color.
3. Click **Color** next to the new light and pick a desaturated blue
   (e.g. `#6080A0`).
4. Reduce its intensity to about `0.4`.
5. Move the light to `(-3, 4, -2)` to fill the shadow side of your
   sculpture.

## Step 8 — Export the rendered image

From the menu: **File → Export image…**, choose a path, set the
resolution to e.g. `1920×1080`. The export uses the GPU compute path
(if Vulkan is available), so even high-res renders complete in under
a second.

## Step 9 — Export a 3D-printable mesh

For physical 3D printing or external rendering:

1. **File → Export mesh (OBJ/STL)…**
2. Choose `mesh.stl` (printers prefer STL).
3. When prompted for **sampling resolution**, enter `128`. Higher
   resolutions catch finer detail at the cost of larger files.

A progress dialog shows live elapsed time; on a typical scene at
128³ resolution this takes 2–5 seconds. The resulting STL contains
~50k triangles depending on surface complexity.

## Step 10 — Save the scene

**File → Save scene…** writes a JSON file containing every object,
its parameters, materials, textures (by path), lights, and camera.
That same file can be reopened later or fed to the headless
`frep_multipath <file> --paths gpu_glsl` for batch rendering.

---

## Where to go next

- **Plugins** — write your own custom SDF node as a .so/.dll. See
  [PLUGIN_AUTHORING.md](PLUGIN_AUTHORING.md).
- **Performance** — when to use CPU vs. GPU, mesh import strategy.
  See [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md).
- **Architecture deep-dive** — how the three execution back-ends
  share a single AST. See [ARCHITECTURE.md](ARCHITECTURE.md).
- **Examples gallery** — browse `examples/*.json` for ready-made
  scenes you can adapt.
- **API reference** — generate Doxygen HTML with
  `cmake --build build --target docs`.

---

## Troubleshooting

- **Viewport stays black**: check the Render tab — your scene might be
  empty, or the camera might be inside the geometry. The default
  camera is at `(2, 2, 4)` looking at origin; reset via the menu if
  you've moved it.
- **"Vulkan not available"**: the GPU compute path requires Vulkan
  1.3+. On Linux ensure `libvulkan1` is installed and your driver
  exposes a Vulkan device (`vulkaninfo` should list at least one).
  Without Vulkan the CPU JIT path takes over automatically.
- **Plugins not loading**: the GUI searches `./plugins/`,
  `./build/plugins/`, `$exe_dir/plugins/`, and
  `$exe_dir/../lib/frep/plugins/`. Plugin filenames must end in `.so`
  (Linux) or `.dll` (Windows).
- **Expression rejected**: hover the red error message in the
  Expression tab — the column number points at the offending token.
  Common mistakes: missing closing parenthesis, using `*` between
  numbers and variables without a space being required (it isn't,
  but check the operator is `*` not `×`), or referencing variables
  other than `x`, `y`, `z`.
