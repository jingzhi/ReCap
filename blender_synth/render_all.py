# A simple script that uses blender to render views of a single object by rotation the camera around it.
import argparse
import json
import os
import pickle
import sys
from math import radians

import bpy
import mathutils
import numpy as np

from blender_utils import delete_all_lights, delete_emissive_materials, setup_pass

if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--blender_file", type=str, default="")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument(
        "--render_layer_key",
        type=str,
        default="RenderLayer",
        help="The key for selecting the active view layer for rendering. Vary for different project.",
    )
    args = parser.parse_args(argv)

render_layer_key = args.render_layer_key  # for [ficus,hotdog,baseball,armadillo]
obj = os.path.splitext(os.path.basename(args.blender_file))[0]
print(f"object name: {obj}")
dump_pose = False
overwrite_render = True

# data gen configuration
RESOLUTION = 800
CAM_LOCATION = (0, 4.0, 0.5)
SAVE_NAME = args.output_dir
blender_project_file = args.blender_file


if args.no_train:
    train_renders = None
else:
    train_renders = {
        "name": "train",
        "num_views": 200,
        "random_views": True,
        "upper_views": True,
        "render_depth": False,
        "render_normal": False,
        "render_albedo": False,
        "circle_fixed_start": [0.0, 0.0, 0.0],  # only used when random views is False
        "circle_fixed_end": [0.7, 0.0, 0.0],  # only used when random views is False
    }
if args.no_val:
    val_renders = None
else:
    val_renders = {
        "name": "val",
        "num_views": 10,
        "random_views": True,
        "upper_views": True,
        "render_depth": False,
        "render_normal": False,
        "render_albedo": False,
        "circle_fixed_start": [0.0, 0.0, 0.0],  # only used when random views is False
        "circle_fixed_end": [0.7, 0.0, 0.0],  # only used when random views is False
    }

if args.no_test:
    test_renders = None
else:
    test_renders = {
        "name": "test",
        "num_views": 200,
        "random_views": False,
        "upper_views": True,
        "render_depth": True,
        "render_normal": True,
        "render_albedo": True,
        "circle_fixed_start": [0.0, 0.0, 0.0],  # only used when random views is False
        "circle_fixed_end": [0.7, 0.0, 0.0],  # only used when random views is False
    }

envs_dict = {
    "interior": "data/Environment_Maps/high_res_envmaps_1k/interior.hdr",
    "courtyard": "data/Environment_Maps/high_res_envmaps_1k/courtyard.hdr",
    "snow": "data/Environment_Maps/high_res_envmaps_1k/snow.hdr",
    "bridge": "data/Environment_Maps/high_res_envmaps_1k/bridge.hdr",
    "forest": "data/Environment_Maps/high_res_envmaps_1k/forest.hdr",
    "fireplace": "data/Environment_Maps/high_res_envmaps_1k/fireplace.hdr",
    "night": "data/Environment_Maps/high_res_envmaps_1k/night.hdr",
    "sunset": "data/Environment_Maps/high_res_envmaps_1k/sunset.hdr",
}
envs_strength_dict = {
    "interior": 1.0,
    "forest": 1.0,
    "fireplace": 1.0,
    "night": 1.0,
    "courtyard": 1.0,
    "bridge": 1.0,
    "snow": 1.0,
    "sunset": 1.0,
}
if obj == "coffee":
    envs_strength_dict["interior"] = 0.3
    envs_strength_dict["courtyard"] = 0.5
    envs_strength_dict["snow"] = 0.6
elif obj == "teapot":
    envs_strength_dict["interior"] = 0.3
    envs_strength_dict["courtyard"] = 0.5
    envs_strength_dict["snow"] = 0.6

output_base = bpy.path.abspath(f"//{SAVE_NAME}")
current_base = bpy.path.abspath(f"//")
print(f"Blender working in {current_base}")
if not os.path.exists(output_base):
    os.makedirs(output_base)


## Helper functions
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


# Setup blender
bpy.ops.wm.open_mainfile(filepath=blender_project_file)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.frame_set(0)  # for proper output numbering
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"  # set output format to .png
scene.render.image_settings.color_mode = "RGBA"
scene.render.use_file_extension = True  # Automaticaaly add file extension when saving

# Background
scene.render.dither_intensity = 0.0
scene.render.film_transparent = True
COLOR_DEPTH = 8
scene.render.image_settings.color_depth = str(COLOR_DEPTH)
# Render Optimizations
scene.render.use_persistent_data = True
# Create collection for objects not to render with background
objs = [
    ob
    for ob in bpy.context.scene.objects
    if ob.type in ("EMPTY") and "Empty" in ob.name
]
with bpy.context.temp_override(selected_objects=objs):
    bpy.ops.object.delete()
# delete any lights ans emissive objects
has_light, lights = delete_all_lights()
has_emissive, emissives = delete_emissive_materials()
if has_light or has_emissive:
    print(
        f"{obj} -- Has light: {has_light}, \n ,{lights}.\n Has emissive: {has_emissive},\n{emissives}\n"
    )
    with open("deletion_log.txt", "a") as f:
        f.write(
            f"{obj} -- Has light: {has_light}, \n ,{lights}.\n Has emissive: {has_emissive},\n{emissives}\n"
        )

# Clean any existing nodes
for node in scene.node_tree.nodes:
    scene.node_tree.nodes.remove(node)

# enable light passes in current active render layer
scene.view_layers[render_layer_key].use_pass_z = True
scene.view_layers[render_layer_key].use_pass_normal = True
scene.view_layers[render_layer_key].use_pass_diffuse_color = True
scene.view_layers[render_layer_key].use_pass_glossy_color = True

# using node to connect outputs
scene.use_nodes = True
tree = scene.node_tree
# New render layer node.
render_layers = tree.nodes.new("CompositorNodeRLayers")
# Depth node
depth_file_output = setup_pass(
    render_layers, tree, "Depth", output_format="OPEN_EXR", use_alpha=False
)
# Normal node
normal_file_output = setup_pass(render_layers, tree, "Normal", color_override=True)

# Albedo node
diff_albedo_file_output = setup_pass(
    render_layers, tree, "DiffCol", color_override=True
)
glossy_albedo_file_output = setup_pass(
    render_layers, tree, "GlossCol", color_override=True
)
albedo_file_output = setup_pass(render_layers, tree, "CombCol", color_override=True)

# World node for changing env map
for node in scene.world.node_tree.nodes:
    scene.world.node_tree.nodes.remove(node)
texture_node = scene.world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
background_node = scene.world.node_tree.nodes.new(type="ShaderNodeBackground")
world_output_node = scene.world.node_tree.nodes.new(type="ShaderNodeOutputWorld")
first_env = next(iter(envs_dict.keys()))
texture_node.image = bpy.data.images.load(envs_dict[first_env])
background_node.inputs["Strength"].default_value = envs_strength_dict[first_env]
scene.world.node_tree.links.new(
    texture_node.outputs["Color"], background_node.inputs["Color"]
)
scene.world.node_tree.links.new(
    background_node.outputs["Background"], world_output_node.inputs["Surface"]
)

# Remove any emission:

# Create empty object to parent the camera
cam = scene.objects["Camera"]
cam.location = CAM_LOCATION
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

for m_render in [train_renders, val_renders, test_renders]:
    if m_render is None:
        continue
    name = m_render["name"]
    num_views = m_render["num_views"]
    upper_views = m_render["upper_views"]
    random_views = m_render["random_views"]
    render_depth = m_render["render_depth"]
    render_normal = m_render["render_normal"]
    render_albedo = m_render["render_albedo"]
    # set rotation to a fixed start, for non-random views
    save_path = os.path.join(output_base, f"{name}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rots_file = os.path.join(save_path, f"{name}_rots.pkl")
    if os.path.exists(rots_file):
        with open(rots_file, "rb") as file:
            print(f"Detected exiting rotation list {rots_file}, loading.")
            rots_list = pickle.load(file)
    else:
        rots_list = []
        # rotation order :XYZ
        circle_fix_start = m_render["circle_fixed_start"]
        circle_fix_end = m_render["circle_fixed_end"]
        vertical_diff = circle_fix_end[0] - circle_fix_start[0]
        rot_fix = circle_fix_start.copy()
        stepsize = 360.0 / num_views
        print(stepsize)
        for i in range(0, num_views):
            # Get a View
            if random_views:  # Random views
                if upper_views:
                    rot = np.random.uniform(0, 1, size=3)
                    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
                    rot[1] = 0
                    rot[2] = rot[2] * 2 * np.pi
                    rots_list.append(rot)
                else:
                    rot = np.random.uniform(0, 2 * np.pi, size=3)
                    rots_list.append(rot)
            else:  # Step through fix views
                rot_fix[0] = (
                    circle_fix_start[0]
                    + (np.cos(radians(stepsize * i)) + 1) / 2 * vertical_diff
                )
                rot_fix[2] += radians(2 * stepsize)
                rots_list.append(rot_fix.copy())
        # dump random rots file
        with open(rots_file, "wb") as file:
            pickle.dump(rots_list, file)
    # Actual renders
    cam_poses = {
        "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
    }
    cam_poses["frames"] = []

    for i in range(0, num_views):
        b_empty.rotation_euler = rots_list[i]
        bpy.context.view_layer.update()
        cam_pose_matrix = listify_matrix(cam.matrix_world)
        for env_idx, (env_name, env_path) in enumerate(envs_dict.items()):
            if render_depth and env_idx == 0:
                depth_file_output.mute = False
                depth_file_output.base_path = save_path
                depth_file_output.file_slots[0].path = f"depth_#{i:03d}"
            else:
                depth_file_output.mute = True
            if render_normal and env_idx == 0:
                normal_file_output.mute = False
                normal_file_output.base_path = save_path
                normal_file_output.file_slots[0].path = f"normal_#{i:03d}"
            else:
                normal_file_output.mute = True
            if render_albedo and env_idx == 0:
                diff_albedo_file_output.mute = False
                glossy_albedo_file_output.mute = False
                albedo_file_output.mute = False
                diff_albedo_file_output.base_path = save_path
                diff_albedo_file_output.file_slots[0].path = f"diffCol_#{i:03d}"
                glossy_albedo_file_output.base_path = save_path
                glossy_albedo_file_output.file_slots[0].path = f"glossCol_#{i:03d}"
                albedo_file_output.base_path = save_path
                albedo_file_output.file_slots[0].path = f"combCol_#{i:03d}"
            else:
                diff_albedo_file_output.mute = True
                glossy_albedo_file_output.mute = True
                albedo_file_output.mute = True
            texture_node.image = bpy.data.images.load(envs_dict[env_name])
            env_strength = envs_strength_dict[env_name]
            background_node.inputs["Strength"].default_value = env_strength
            ## 1.1 rgb render
            frame_name = f"rgba_{i:04d}_{env_name}"
            print(f"Rendering {frame_name}")
            scene.render.filepath = os.path.join(save_path, frame_name)
            frame_data = {
                "file_path": f"./{name}/{frame_name}",
                "env": f"{env_name}",
                "env_strength": env_strength,
                "frame_idx": i,
                "transform_matrix": cam_pose_matrix,
            }
            cam_poses["frames"].append(frame_data)
            if (not overwrite_render) and os.path.exists(
                os.path.join(save_path, f"{frame_name}.png")
            ):
                print(f"{frame_name}.png exists, continue...")
                continue
            else:
                bpy.ops.render.render(write_still=True)  # render still

    # dump pose json file,note each rgb env will have repeated pose
    if dump_pose:
        with open(output_base + "/" + f"transforms_{name}.json", "w") as out_file:
            json.dump(cam_poses, out_file, indent=4)
