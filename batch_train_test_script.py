import os
import time

envs_dict = {
    "interior": "data/Environment_Maps/high_res_envmaps_1k/interior.hdr",
    "courtyard": "data/Environment_Maps/high_res_envmaps_1k/courtyard.hdr",
    "snow": "data/Environment_Maps/high_res_envmaps_1k/snow.hdr",
    "forest": "data/Environment_Maps/high_res_envmaps_1k/forest.hdr",
    "fireplace": "data/Environment_Maps/high_res_envmaps_1k/fireplace.hdr",
    "night": "data/Environment_Maps/high_res_envmaps_1k/night.hdr",
    "sunset": "data/Environment_Maps/high_res_envmaps_1k/sunset.hdr",
    "bridge": "data/Environment_Maps/high_res_envmaps_1k/bridge.hdr",
}
# scaled envmaps for teapot and coffee
envs_dict_scale = {
    "interior": "data/Environment_Maps/high_res_envmaps_1k/interior_0_3.hdr",
    "courtyard": "data/Environment_Maps/high_res_envmaps_1k/courtyard_0_5.hdr",
    "snow": "data/Environment_Maps/high_res_envmaps_1k/snow_0_6.hdr",
    "forest": "data/Environment_Maps/high_res_envmaps_1k/forest.hdr",
    "fireplace": "data/Environment_Maps/high_res_envmaps_1k/fireplace.hdr",
    "night": "data/Environment_Maps/high_res_envmaps_1k/night.hdr",
    "sunset": "data/Environment_Maps/high_res_envmaps_1k/sunset.hdr",
    "bridge": "data/Environment_Maps/high_res_envmaps_1k/bridge.hdr",
}


input_objs = [
    "toaster",
    "helmet",
    "chair",
    "musclecar",
    "lego",
    "coffee",
    "teapot",
    "drums",
    "ficus",
    "hotdog",
    "ship",
    "material",
    "mic",
]

perform_train = True
perform_render = True  # NVS
perform_relight = True  # render with novel HDR map
perform_eval = True

notes = "baseline"
train_envs = [
    "bridge",
    "courtyard",
]  # must match env names specified in the train json file
train_transform = "transforms_train_bridge_courtyard.json"
for obj in input_objs:
    if obj in ["coffee", "teapot"]:
        m_envs_dict = envs_dict_scale
    else:
        m_envs_dict = envs_dict
    input_dir = f"data/RelightObj/{obj}_relight"
    output_dir = f"output/{obj}_relight_{notes}"
    # train
    if perform_train:
        os.system(
            f" python train.py \
                    --opts \
                    model_dir {output_dir} \
                    scene.white_background True \
                    scene.train_envs {','.join(train_envs)} \
                    scene.blender.train_transform {train_transform} \
                    scene.eval False \
                    scene.source_path {input_dir} "
        )
    # render
    if perform_render:
        for env_name in train_envs:
            test_transform = f"transforms_test_{env_name}.json"
            os.system(
                f" python render.py \
                        --opts \
                        model_dir {output_dir} \
                        checkpoint_path {output_dir}/chkpt30000.pth \
                        scene.source_path {input_dir} \
                        scene.white_background True \
                        scene.blender.test_transform {test_transform} \
                        render.save_name {env_name}_learned \
                        render.light_name {env_name} "
            )
    # relight
    if perform_relight:
        for env_name, env_hdr in m_envs_dict.items():
            test_transform = f"transforms_test_{env_name}.json"
            os.system(
                f" python render.py \
                       --opts \
                       model_dir {output_dir} \
                       checkpoint_path {output_dir}/chkpt30000.pth \
                       scene.source_path {input_dir} \
                       scene.white_background True \
                       scene.blender.test_transform {test_transform}\
                       render.save_name {env_name} \
                       render.light_name {env_name} \
                       render.light_hdr_path {env_hdr} "
            )
    # metric
    if perform_eval:
        for env_name in train_envs:
            os.system(
                f"python eval.py --output_dir {output_dir}/test/{env_name}_learned/renders --gt_dir data/RelightObj/{obj}_relight/test --light_name {env_name}_learned --gt_name {env_name} --white_background"
            )
        for env_name in m_envs_dict.keys():
            os.system(
                f"python eval.py --output_dir {output_dir}/test/{env_name}/renders --gt_dir data/RelightObj/{obj}_relight/test --light_name {env_name} --gt_name {env_name}  --notes {notes} --white_background"
            )
