import random

from utils.camera_utils import camera_to_JSON

from .dataset_readers import sceneLoadTypeCallbacks


def get_sceneinfo(scene_cfg, shuffle=False):
    """
    class SceneInfo(NamedTuple):
        point_cloud: BasicPointCloud
        train_cameras: list
        test_cameras: list
        nerf_normalization: dict
        ply_path: str
    """
    scene_type = scene_cfg.scene_type

    if scene_type == "colmap":
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            scene_cfg.source_path, scene_cfg.colmap.image_dir, scene_cfg.eval
        )
    elif scene_type == "blender":
        scene_info = sceneLoadTypeCallbacks["Blender"](scene_cfg)
    else:
        assert False, f"Could not recognize scene type! {self.scene_type}"

    if shuffle:
        random.shuffle(
            scene_info.train_cameras
        )  # Multi-res consistent random shuffling
        random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
    return scene_info


def dump_sceneinfo(scene_info, save_path):
    # dump ply
    with open(scene_info.ply_path, "rb") as src_file, open(
        os.path.join(save_path, "input.ply"), "wb"
    ) as dest_file:
        dest_file.write(src_file.read())
    # dump cameras.json
    json_cams = []
    camlist = []
    if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
    if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(save_path, "cameras.json"), "w") as file:
        json.dump(json_cams, file)
