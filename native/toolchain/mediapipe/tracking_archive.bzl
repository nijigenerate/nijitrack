"""Archive the selected C API and its compiled transitive closure (Bazel 7)."""
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")

def _impl(ctx):
    toolchain = find_cc_toolchain(ctx)
    features = cc_common.configure_features(ctx = ctx, cc_toolchain = toolchain)
    inputs = depset(transitive = [dep[CcInfo].linking_context.linker_inputs for dep in ctx.attr.deps], order = "topological")
    objects = []
    remaining = []
    flags = []
    for linker in inputs.to_list():
        flags.extend(linker.user_link_flags)
        for library in linker.libraries:
            compiled = library.pic_objects or library.objects
            if compiled:
                objects.extend(compiled)
            elif library.static_library or library.pic_static_library or library.dynamic_library:
                remaining.append(str(linker.owner))
    windows = ctx.target_platform_has_constraint(ctx.attr._windows[platform_common.ConstraintValueInfo])
    archive = ctx.actions.declare_file(ctx.label.name + ".lib" if windows else "lib" + ctx.label.name + ".a")
    variables = cc_common.create_link_variables(cc_toolchain = toolchain, feature_configuration = features,
        output_file = archive.path, is_using_linker = False)
    action = ACTION_NAMES.cpp_link_static_library
    args = ctx.actions.args()
    args.add_all(cc_common.get_memory_inefficient_command_line(feature_configuration = features, action_name = action, variables = variables))
    unique_objects = depset(objects)
    args.add_all(unique_objects)
    args.use_param_file("@%s", use_always = True)
    ctx.actions.run(executable = cc_common.get_tool_for_action(feature_configuration = features, action_name = action),
        arguments = [args], inputs = depset(transitive = [unique_objects, toolchain.all_files]), outputs = [archive],
        env = cc_common.get_environment_variables(feature_configuration = features, action_name = action, variables = variables),
        use_default_shell_env = True, mnemonic = "TrackingStaticArchive")
    linkdeps = ctx.actions.declare_file(ctx.label.name + "_linkdeps.txt")
    linkopts = ctx.actions.declare_file(ctx.label.name + "_linkopts.txt")
    ctx.actions.write(linkdeps, "\n".join(remaining))
    ctx.actions.write(linkopts, "\n".join(flags))
    return [DefaultInfo(files = depset([archive])), OutputGroupInfo(linkdeps = depset([linkdeps]), linkopts = depset([linkopts]))]

tracking_archive = rule(implementation = _impl, attrs = {
    "deps": attr.label_list(providers = [CcInfo]),
    "_windows": attr.label(default = "@platforms//os:windows"),
}, toolchains = use_cc_toolchain(), fragments = ["cpp"])
