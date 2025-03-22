PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/ficus.blend" --output_dir "ficus_relight" --render_layer_key "View Layer" --no_val
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/teapot.blend" --output_dir "teapot_relight" --no_val
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/materials.blend" --output_dir "material_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/mic.blend" --output_dir "mic_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/toaster.blend" --output_dir "toaster_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/hotdog.blend" --output_dir "hotdog_relight" --render_layer_key "View Layer" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/chair.blend" --output_dir "chair_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/helmet.blend" --output_dir "helmet_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/musclecar.blend" --output_dir "musclecar_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/ship.blend" --output_dir "ship_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/coffee.blend" --output_dir "coffee_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/lego.blend" --output_dir "lego_relight" --no_val 
PYTHONPATH=$(pwd) blender --background --python render_all.py -- --blender_file "./blender_files/drums.blend" --output_dir "drums_relight" --no_val 
:w


