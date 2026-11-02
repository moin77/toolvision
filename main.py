"""
ToolVision synthetic generator v1.1  (Blender 4.5)

Run inside generator.blend (a COPY of assets.blend).

Quick test in the Blender UI:
    Scripting tab -> Text > Open -> generate_tools.py -> Run   (renders COUNT images)

Full run from Command Prompt (Blender stays responsive, faster):
    "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe" -b generator.blend -P generate_tools.py -- --count 1500 --seed 42

Output:
    OUT_DIR/images/synth_00000.jpg ...
    OUT_DIR/labels/synth_00000.txt ...   (YOLO: class x_center y_center width height, all 0-1)
    OUT_DIR/classes.txt
    OUT_DIR/metadata.json                (config, seed, versions, date, class counts)
"""

import bpy
import os
import sys
import json
import math
import random
import colorsys
import datetime
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# =====================================================================
# 1. CONFIG  -- edit these paths before running
# =====================================================================
OUT_DIR     = r"E:\Sythetic_Data\tool-detector\synthetic_raw"
HDRI_DIR    = r"E:\Sythetic_Data\wrench_demo\assets\hdri"       # folder with Poly Haven .hdr / .exr files
TEXTURE_DIR = r"E:\Sythetic_Data\wrench_demo\assets\textures"    # folder with Poly Haven texture folders

COUNT = 10          # images to render (overridden by --count)
START = 0           # first image index (overridden by --start)
SEED  = 42          # master seed (overridden by --seed)

RES_LONG, RES_SHORT = 1024, 768   # 4:3 like a phone camera
SAMPLES = 64                      # Cycles samples (denoised, so 64 is enough)

CLASS_NAMES = ["wrench", "screwdriver", "hammer", "pliers"]
TOOL_NAMES = [
    "combination_wrench",
    "screwdriver",
    "screwdrivers_02_a",
    "screwdrivers_02_b",
    "wooden_hammer_01",
    "pliers_a",
]

# Randomization ranges (these are the "decisions" you will tune)
TOOLS_PER_IMAGE   = (1, 4)        # min, max tools in one image
SCATTER_RADIUS    = 0.20          # metres: tools land inside +-this square
MIN_TOOL_SPACING  = 0.12          # metres between tool centres (less = more overlap)
TOOL_SCALE        = (0.85, 1.15)  # size variation
FLIP_CHANCE       = 0.5           # chance a tool lies on its other face
CAM_DISTANCE      = (0.30, 0.90)  # metres from target (v1.1: closer, like the real photos)
CAM_ELEVATION_DEG = (40, 85)      # 90 = straight down
CAM_LENS_MM       = 26            # roughly a phone main camera
PORTRAIT_CHANCE   = 0.4
HDRI_STRENGTH     = (0.2, 1.8)
SPOT_CHANCE       = 0.35          # "flashlight in a dark room" images
EXPOSURE          = (-0.3, 0.8)   # v1.1: brighter on average
TEXTURED_FLOOR_CHANCE = 0.75      # otherwise a plain random-colour floor
MIN_BOX_FRACTION  = 0.01          # skip boxes smaller than 1% of image width/height


# =====================================================================
# 2. COMMAND-LINE OVERRIDES  (only used when run with -- --count ...)
# =====================================================================
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

def arg(flag, default, cast):
    if flag in argv:
        return cast(argv[argv.index(flag) + 1])
    return default

COUNT   = arg("--count", COUNT, int)
START   = arg("--start", START, int)
SEED    = arg("--seed", SEED, int)
OUT_DIR = arg("--out", OUT_DIR, str)


# =====================================================================
# 3. HELPERS
# =====================================================================
def find_files(folder, extensions, name_contains=None):
    """Sorted list of files (sorted = same order every run = reproducible)."""
    found = []
    if not os.path.isdir(folder):
        return found
    for root, _, files in os.walk(folder):
        for f in files:
            fl = f.lower()
            if not fl.endswith(extensions):
                continue
            if name_contains and not any(k in fl for k in name_contains):
                continue
            found.append(os.path.join(root, f))
    return sorted(found)


def load_image(path):
    return bpy.data.images.load(path, check_existing=True)


# =====================================================================
# 4. SCENE SETUP  (runs once)
# =====================================================================
scene = bpy.context.scene

# Remove objects/worlds left over from a previous run in the UI
for o in list(bpy.data.objects):
    if o.name.startswith("GEN_"):
        bpy.data.objects.remove(o, do_unlink=True)
for w in list(bpy.data.worlds):
    if w.name.startswith("GEN_"):
        bpy.data.worlds.remove(w)

# Collect the tools and check their IDs
tools = []
for name in TOOL_NAMES:
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Object '{name}' not found in this .blend file")
    if "class_id" not in obj:
        raise RuntimeError(f"Object '{name}' has no class_id property")
    tools.append(obj)

tools_by_class = {cid: [t for t in tools if int(t["class_id"]) == cid]
                  for cid in range(len(CLASS_NAMES))}
for cid, objs in tools_by_class.items():
    if not objs:
        raise RuntimeError(f"No object has class_id {cid} ({CLASS_NAMES[cid]})")

# Everything that is not a tool stays out of the render
for o in scene.objects:
    if o not in tools:
        o.hide_render = True

# --- Render settings ---
scene.render.engine = 'CYCLES'
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.quality = 90
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = 'GPU'
    print("Rendering on GPU (OptiX)")
except Exception as e:
    print("GPU setup failed, rendering on CPU:", e)

# --- Floor ---
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = "GEN_floor"
floor_mat = bpy.data.materials.new("GEN_floor_mat")
floor_mat.use_nodes = True
fnt = floor_mat.node_tree
bsdf = fnt.nodes["Principled BSDF"]
floor_tex = fnt.nodes.new("ShaderNodeTexImage")
floor_map = fnt.nodes.new("ShaderNodeMapping")
floor_coord = fnt.nodes.new("ShaderNodeTexCoord")
fnt.links.new(floor_coord.outputs["UV"], floor_map.inputs["Vector"])
fnt.links.new(floor_map.outputs["Vector"], floor_tex.inputs["Vector"])
floor.data.materials.append(floor_mat)

# --- World (HDRI lighting + background) ---
world = bpy.data.worlds.new("GEN_world")
scene.world = world
world.use_nodes = True
wnt = world.node_tree
background = wnt.nodes["Background"]
env_tex = wnt.nodes.new("ShaderNodeTexEnvironment")
env_map = wnt.nodes.new("ShaderNodeMapping")
env_coord = wnt.nodes.new("ShaderNodeTexCoord")
wnt.links.new(env_coord.outputs["Generated"], env_map.inputs["Vector"])
wnt.links.new(env_map.outputs["Vector"], env_tex.inputs["Vector"])

# --- Camera + target it looks at ---
target = bpy.data.objects.new("GEN_target", None)
scene.collection.objects.link(target)
cam_data = bpy.data.cameras.new("GEN_cam")
cam_data.lens = CAM_LENS_MM
cam = bpy.data.objects.new("GEN_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
track = cam.constraints.new('TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# --- Spotlight ("flashlight") ---
spot_data = bpy.data.lights.new("GEN_spot", 'SPOT')
spot = bpy.data.objects.new("GEN_spot", spot_data)
scene.collection.objects.link(spot)
spot_track = spot.constraints.new('TRACK_TO')
spot_track.target = target
spot_track.track_axis = 'TRACK_NEGATIVE_Z'
spot_track.up_axis = 'UP_Y'

# --- Asset lists ---
hdris = find_files(HDRI_DIR, (".hdr", ".exr"))
floor_textures = find_files(TEXTURE_DIR, (".jpg", ".jpeg", ".png"),
                            name_contains=("_diff", "diffuse", "_col", "albedo"))
print(f"Found {len(hdris)} HDRIs and {len(floor_textures)} floor textures")
if not hdris:
    print("WARNING: no HDRIs found -- lighting will be a plain grey world")
if not floor_textures:
    print("WARNING: no textures found -- floors will be plain colours only")


# =====================================================================
# 5. RANDOMIZERS  (run for every image, each uses that image's rng)
# =====================================================================
def randomize_floor(rng):
    for link in list(fnt.links):
        if link.to_socket == bsdf.inputs["Base Color"]:
            fnt.links.remove(link)
    if floor_textures and rng.random() < TEXTURED_FLOOR_CHANCE:
        floor_tex.image = load_image(rng.choice(floor_textures))
        fnt.links.new(floor_tex.outputs["Color"], bsdf.inputs["Base Color"])
        s = rng.uniform(1.0, 4.0)
        floor_map.inputs["Scale"].default_value = (s, s, s)
        floor_map.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)
    else:
        r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0, 0.6), rng.uniform(0.2, 0.95))
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.3, 0.9)


def randomize_world(rng, dark):
    for link in list(wnt.links):
        if link.to_socket == background.inputs["Color"]:
            wnt.links.remove(link)
    if hdris:
        env_tex.image = load_image(rng.choice(hdris))
        wnt.links.new(env_tex.outputs["Color"], background.inputs["Color"])
        env_map.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)
    else:
        background.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    if dark:
        background.inputs["Strength"].default_value = rng.uniform(0.03, 0.3)
    else:
        background.inputs["Strength"].default_value = rng.uniform(*HDRI_STRENGTH)


def place_tools(rng):
    for t in tools:
        t.hide_render = True

    # Pick a CLASS first, then an object of that class.
    # This keeps the 4 classes balanced even though there are 3 screwdriver models.
    k = rng.randint(*TOOLS_PER_IMAGE)
    chosen = []
    for _ in range(k):
        cid = rng.randrange(len(CLASS_NAMES))
        candidates = [o for o in tools_by_class[cid] if o not in chosen]
        if candidates:
            chosen.append(rng.choice(candidates))

    placed = []
    for t in chosen:
        for _ in range(30):
            x = rng.uniform(-SCATTER_RADIUS, SCATTER_RADIUS)
            y = rng.uniform(-SCATTER_RADIUS, SCATTER_RADIUS)
            if all((x - px) ** 2 + (y - py) ** 2 > MIN_TOOL_SPACING ** 2 for px, py in placed):
                break
        placed.append((x, y))
        flip = rng.random() < FLIP_CHANCE
        t.rotation_euler = (0.0, math.pi if flip else 0.0, rng.uniform(0, 2 * math.pi))
        s = rng.uniform(*TOOL_SCALE)
        t.scale = (s, s, s)
        t.location = (x, y, 0.0)
        t.hide_render = False

    # Drop each tool so its lowest point touches the floor
    bpy.context.view_layer.update()
    for t in chosen:
        min_z = min((t.matrix_world @ Vector(c)).z for c in t.bound_box)
        t.location.z -= min_z
    return chosen


def randomize_camera(rng, chosen):
    cx = sum(t.location.x for t in chosen) / len(chosen)
    cy = sum(t.location.y for t in chosen) / len(chosen)
    target.location = (cx + rng.uniform(-0.05, 0.05), cy + rng.uniform(-0.05, 0.05), 0.0)
    d = rng.uniform(*CAM_DISTANCE)
    el = math.radians(rng.uniform(*CAM_ELEVATION_DEG))
    az = rng.uniform(0, 2 * math.pi)
    cam.location = (
        target.location.x + d * math.cos(el) * math.cos(az),
        target.location.y + d * math.cos(el) * math.sin(az),
        d * math.sin(el),
    )


def randomize_spot(rng, use_spot):
    spot.hide_render = not use_spot
    if use_spot:
        spot_data.energy = rng.uniform(20, 150)
        spot_data.spot_size = math.radians(rng.uniform(30, 70))
        spot_data.spot_blend = rng.uniform(0.1, 0.5)
        spot.location = (target.location.x + rng.uniform(-0.4, 0.4),
                         target.location.y + rng.uniform(-0.4, 0.4),
                         rng.uniform(0.6, 1.5))


# =====================================================================
# 6. LABELS: 3D tool -> 2D YOLO box
# =====================================================================
def yolo_box(obj):
    """Project every vertex of the (modifier-evaluated) mesh into the camera.
    Box = full tool outline, even where another tool covers it,
    clipped to the image edges (same rule as the real-photo labels)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()
    mw = ev.matrix_world
    xs, ys = [], []
    for v in mesh.vertices:
        p = world_to_camera_view(scene, cam, mw @ v.co)
        if p.z > 0:                     # in front of the camera
            xs.append(p.x)
            ys.append(p.y)
    ev.to_mesh_clear()
    if not xs:
        return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    w, h = x1 - x0, y1 - y0
    if w < MIN_BOX_FRACTION or h < MIN_BOX_FRACTION:
        return None                     # outside the image or too tiny to learn from
    # Blender's camera view has (0,0) at the BOTTOM-left; YOLO uses TOP-left.
    return (x0 + x1) / 2, 1.0 - (y0 + y1) / 2, w, h


# =====================================================================
# 7. MAIN LOOP
# =====================================================================
img_dir = os.path.join(OUT_DIR, "images")
lbl_dir = os.path.join(OUT_DIR, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

class_counts = {n: 0 for n in CLASS_NAMES}
start_time = datetime.datetime.now()

for i in range(START, START + COUNT):
    # One rng per image: image 734 is identical every time you regenerate it
    rng = random.Random(SEED * 1_000_003 + i)

    portrait = rng.random() < PORTRAIT_CHANCE
    scene.render.resolution_x = RES_SHORT if portrait else RES_LONG
    scene.render.resolution_y = RES_LONG if portrait else RES_SHORT

    use_spot = rng.random() < SPOT_CHANCE
    randomize_floor(rng)
    randomize_world(rng, dark=use_spot)
    chosen = place_tools(rng)
    randomize_camera(rng, chosen)
    randomize_spot(rng, use_spot)
    scene.view_settings.exposure = rng.uniform(*EXPOSURE)
    # v1.1: phone photos have punchier contrast than Blender's default AgX look
    scene.view_settings.view_transform = rng.choice(["Standard", "Standard", "AgX"])
    bpy.context.view_layer.update()

    name = f"synth_{i:05d}"
    lines = []
    for t in chosen:
        box = yolo_box(t)
        if box:
            cid = int(t["class_id"])
            lines.append(f"{cid} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            class_counts[CLASS_NAMES[cid]] += 1

    scene.render.filepath = os.path.join(img_dir, name + ".jpg")
    bpy.ops.render.render(write_still=True)
    with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    done = i - START + 1
    print(f"[{done}/{COUNT}] {name}: {len(lines)} boxes")

# =====================================================================
# 8. METADATA  (makes the dataset reproducible and documentable)
# =====================================================================
with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
    f.write("\n".join(CLASS_NAMES) + "\n")

metadata = {
    "generator": "ToolVision generate_tools.py v1.1",
    "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "blender_version": bpy.app.version_string,
    "seed": SEED,
    "start_index": START,
    "count": COUNT,
    "resolution": [RES_LONG, RES_SHORT],
    "samples": SAMPLES,
    "classes": CLASS_NAMES,
    "tool_objects": {t.name: int(t["class_id"]) for t in tools},
    "num_hdris": len(hdris),
    "num_floor_textures": len(floor_textures),
    "randomization": {
        "tools_per_image": TOOLS_PER_IMAGE,
        "scatter_radius_m": SCATTER_RADIUS,
        "min_tool_spacing_m": MIN_TOOL_SPACING,
        "tool_scale": TOOL_SCALE,
        "flip_chance": FLIP_CHANCE,
        "cam_distance_m": CAM_DISTANCE,
        "cam_elevation_deg": CAM_ELEVATION_DEG,
        "cam_lens_mm": CAM_LENS_MM,
        "portrait_chance": PORTRAIT_CHANCE,
        "hdri_strength": HDRI_STRENGTH,
        "spot_chance": SPOT_CHANCE,
        "exposure": EXPOSURE,
        "textured_floor_chance": TEXTURED_FLOOR_CHANCE,
    },
    "label_rule": "full tool outline (including parts covered by other tools), clipped to image edge",
    "boxes_per_class": class_counts,
    "render_minutes": round((datetime.datetime.now() - start_time).total_seconds() / 60, 1),
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Done. Boxes per class:", class_counts)