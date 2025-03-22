from yacs.config import CfgNode as CN

# root config node
cfg = CN()

# Miscellaneous
cfg.model_dir = "output/chair"
cfg.num_workers = 4
cfg.skip_train = False
cfg.skip_test = False
cfg.debug_visual = True
cfg.debug_visual_iterations = [1, 2000, 5000, 10000, 15000, 29000]
cfg.detect_anomaly = True
cfg.quiet = False
cfg.checkpoint_path = None

cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.brdf_lut_path = "renderer/brdf_256_256.bin"
# for NVS & relighting
cfg.render.light_name = ""  # light name
cfg.render.light_hdr_path = ""  # path to ligh map hdr
cfg.render.save_name = ""  # render save name

# Scene
cfg.scene = CN()
cfg.scene.source_path = ""  # train_dir
cfg.scene.resolution = -1
cfg.scene.data_device = "cuda"
cfg.scene.white_background = True
cfg.scene.shuffle = True
cfg.scene.resolution_scales = [1.0]
cfg.scene.train_envs = ["bridge"]
cfg.scene.scene_type = "blender"
cfg.scene.eval = True
## for blender dataset
cfg.scene.blender = CN()
cfg.scene.blender.extension = ".png"
cfg.scene.blender.train_transform = "transforms_train_bridge.json"
cfg.scene.blender.test_transform = "transforms_test_bridge.json"
## for colmap dataset (not tested, kept from original 3DGS code)
cfg.scene.colmap = CN()
cfg.scene.colmap.image_dir = "images"

# Model
cfg.model = CN()
cfg.model.sh_degree = 3

# Envmap
cfg.light = CN()
cfg.light.cubemap_res = 256

# Optimizor
cfg.optim = CN()
cfg.optim.iterations = 30000
cfg.optim.position_lr_init = 0.00016
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
cfg.optim.material_lr_init = 0.025
cfg.optim.material_lr_final = 0.00025
cfg.optim.material_lr_delay_mult = 0.01
cfg.optim.material_lr_max_steps = 30000
cfg.optim.light_lr_init = 0.025
cfg.optim.light_lr_final = 0.00025
cfg.optim.percent_dense = 0.01
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_predicted_normal = 0.1
cfg.optim.lambda_neutral = 0.5
cfg.optim.lambda_con = 0.2
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.init_until_iter = 0
cfg.optim.densify_grad_threshold = 0.0002
