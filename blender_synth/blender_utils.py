import bpy


def setup_pass(
    render_layers,
    tree,
    pass_name,
    use_alpha=True,
    color_override=False,
    output_format="PNG",
):
    """
    render_layers: target 'CompositorNodeRLayers'
    tree: scene node tree
    use_alpha: whether to composite scene alpha into output, valid for PNG output
    color_overide: set as true for saving albedo as PNG in linear value
    output_format: str ['PNG','OPEN_EXR']
    """
    output_file_node = tree.nodes.new(type="CompositorNodeOutputFile")
    output_file_node.format.file_format = output_format

    if color_override:
        assert output_format == "PNG"
        output_file_node.format.color_management = "OVERRIDE"
        output_file_node.format.display_settings.display_device = "None"  # sRGB
        output_file_node.format.view_settings.view_transform = "Raw"  # filmic

    if pass_name == "CombCol":
        addition_node = tree.nodes.new(type="CompositorNodeMixRGB")
        addition_node.blend_type = "ADD"
        tree.links.new(render_layers.outputs["DiffCol"], addition_node.inputs[1])
        tree.links.new(render_layers.outputs["GlossCol"], addition_node.inputs[2])
        target_port = addition_node.outputs["Image"]
    else:
        target_port = render_layers.outputs[pass_name]

    if use_alpha:
        assert output_format == "PNG"
        output_file_node.format.color_mode = "RGBA"
        alpha_node = tree.nodes.new(type="CompositorNodeSetAlpha")
        tree.links.new(target_port, alpha_node.inputs[0])
        tree.links.new(render_layers.outputs["Alpha"], alpha_node.inputs[1])
        tree.links.new(alpha_node.outputs["Image"], output_file_node.inputs[0])
    else:
        tree.links.new(target_port, output_file_node.inputs[0])

    return output_file_node


def delete_all_lights():
    """Deletes all light objects in the scene."""
    bpy.ops.object.select_all(action="DESELECT")  # Deselect everything
    lights = []
    light_names = []
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":  # Check if object is a light
            lights.append(obj)
            light_names.append(obj.name)
    with bpy.context.temp_override(selected_objects=lights):
        bpy.ops.object.delete()  # Delete selected objects
    if not light_names:
        has_light = False
    else:
        has_light = True
    return has_light, light_names


def delete_emissive_materials():
    """Finds all objects using emissive materials (including meshes, curves, surfaces)."""
    bpy.ops.object.select_all(action="DESELECT")  # Deselect everything
    emissive_mat = []

    # Object types that can have materials
    valid_types = {"MESH", "SURFACE", "CURVE", "SURFACE", "META"}

    for obj in bpy.data.objects:
        if obj.type in valid_types:
            for slot in obj.material_slots:
                mat = slot.material
                if mat and mat.node_tree:  # Ensure material uses nodes
                    for node in mat.node_tree.nodes:
                        if node.type == "EMISSION":  # Check for emission node
                            print(f"Removing emission node from {mat.name}")
                            emissive_mat.append(mat.name)
                            mat.node_tree.nodes.remove(node)
    if not emissive_mat:
        has_emissive = False
    else:
        has_emissive = True
    return has_emissive, emissive_mat
