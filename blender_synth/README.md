## Data Generation
The rendering script, `render_all.py`, renders an object under all provided light maps.

Paths to the light maps are configured in line 87 of `render_all.py`.

To render a single object under all lighting conditions, use the following command:

```bash
#for ficus, hotdog
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/<object>.blend" --output_dir "<object>_relight" --render_layer_key "View Layer"

#for other objects
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/<object>.blend" --output_dir "<object>_relight" 
```

To batch render test and train sets for all objects under all lighting conditions, use the following script:

```bash
bash batch_render.sh
```

### Albedo, depth, normal
If you wish to render additional passes for the provided dataset, configure  the `--output_dir` to the existing data path and adjust the render options accordingly (see line 44-85 of `render_all.py`). Since we have included the `train_rots.pkl` and `test_rots.pkl` in the dataset, the rendering script will reload these poses for rendering. Ensure that `overwrite_render` is enabled in line 35.

#### Extra note on Albedo
Note the material for the blender files in Shiny Blender and NeRF Synthetic are not all well-defined, the "albedo" pass is thus not accurate. This script renders the blender "diffCol", "GlossCol" pass and their combination "CombCol". Please use them at your own discresion as "albedo". 

In the paper, we use the following "albedo" choices when reproducing TensoIR:
| Pass | Objects |
|----------|----------|
| diffCol | ficus, teapot, ship, musclecar, mic, hotdog, helmet|
| combCol | material, toaster, drums, chair, coffee, lego |
