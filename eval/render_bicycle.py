import os
import torch
import torchvision
import numpy as np

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from argparse import ArgumentParser

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import get_combined_args
from utils.general_utils import safe_state

from eval.openclip_encoder import OpenCLIPNetwork


def compute_gaussian_mask(
    gaussians,
    clip_model,
    prompt,
    threshold=0.4,
    num_knn=10,
):
    """
    Compute a binary Gaussian mask for a text prompt using
    VALA language features and KNN smoothing.
    """

    clip_model.set_positives([prompt])

    gs_lang_feat = gaussians.get_language_feature()

    print("Language feature shape:", tuple(gs_lang_feat.shape))

    # Safety check:
    # every Gaussian must have exactly one language feature vector.
    num_gaussians = gaussians.get_xyz.shape[0]
    num_features = gs_lang_feat.shape[0]

    if num_gaussians != num_features:
        raise RuntimeError(
            f"Gaussian / language-feature mismatch: "
            f"{num_gaussians} Gaussians vs "
            f"{num_features} language features"
        )

    # --------------------------------------------------
    # CLIP relevance for every Gaussian
    # --------------------------------------------------

    valid_map_3d = clip_model.get_max_across_3d(gs_lang_feat)

    # For one prompt this should become [N].
    if valid_map_3d.ndim == 0:
        valid_map_3d = valid_map_3d.unsqueeze(0)

    if valid_map_3d.ndim == 2:
        relv_1d = valid_map_3d[0]
    else:
        relv_1d = valid_map_3d

    print(
        "Raw relevance:",
        "min =", relv_1d.min().item(),
        "max =", relv_1d.max().item(),
        "mean =", relv_1d.mean().item(),
    )

    # --------------------------------------------------
    # KNN smoothing
    # --------------------------------------------------

    gs_xyz_np = gaussians.get_xyz.detach().cpu().numpy()

    print("Building KNN graph...")

    nbrs = NearestNeighbors(
        n_neighbors=num_knn
    ).fit(gs_xyz_np)

    _, indices = nbrs.kneighbors(gs_xyz_np)

    indices = torch.from_numpy(
        indices
    ).to(relv_1d.device)

    neighbors_vals = relv_1d[indices]
    neighbors_avg = neighbors_vals.mean(dim=1)

    relv_smoothed = 0.5 * (
        relv_1d + neighbors_avg
    )

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------

    output = relv_smoothed

    output = output - torch.min(output)

    output = output / (
        torch.max(output) + 1e-9
    )

    output = output * 2.0 - 1.0

    output = torch.clip(
        output,
        0,
        1,
    )

    gs_mask = output > threshold

    print(
        "Normalized relevance:",
        "min =", output.min().item(),
        "max =", output.max().item(),
        "mean =", output.mean().item(),
    )

    print(
        "Selected Gaussians:",
        gs_mask.sum().item(),
        "/",
        gs_mask.numel(),
        f"({100.0 * gs_mask.float().mean().item():.2f}%)",
    )

    return gs_mask


def render_views(
    views,
    gaussians,
    pipeline,
    background,
    mask,
    output_dir,
    num_views,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    views = views[:num_views]

    for idx, view in enumerate(
        tqdm(
            views,
            desc="Rendering selected views",
        )
    ):
        # --------------------------------------------------
        # Original RGB
        # --------------------------------------------------

        result_original = render(
            view,
            gaussians,
            pipeline,
            background,
        )

        rgb = result_original["render"]

        # --------------------------------------------------
        # Masked RGB
        # --------------------------------------------------

        result_masked = render(
            view,
            gaussians,
            pipeline,
            background,
            mask=mask,
        )

        masked_rgb = result_masked["render"]

        # --------------------------------------------------
        # Alpha
        # --------------------------------------------------

        alpha = result_masked["render_alpha"]

        print(
            f"View {idx}: "
            f"rgb={tuple(rgb.shape)}, "
            f"masked_rgb={tuple(masked_rgb.shape)}, "
            f"alpha={tuple(alpha.shape)}"
        )

        torchvision.utils.save_image(
            rgb,
            os.path.join(
                output_dir,
                f"{idx:05d}_original.png",
            ),
        )

        torchvision.utils.save_image(
            masked_rgb,
            os.path.join(
                output_dir,
                f"{idx:05d}_masked.png",
            ),
        )

        # render_alpha is normally [1, H, W, 1]
        # Convert to [1, H, W].

        if alpha.ndim == 4:
            alpha = alpha.squeeze(0).squeeze(-1)

        elif (
            alpha.ndim == 3
            and alpha.shape[-1] == 1
        ):
            alpha = alpha.squeeze(-1)

        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(0)

        if alpha.ndim != 3:
            raise RuntimeError(
                "Unexpected render_alpha shape "
                f"after conversion: {tuple(alpha.shape)}"
            )

        torchvision.utils.save_image(
            alpha,
            os.path.join(
                output_dir,
                f"{idx:05d}_mask.png",
            ),
        )


def main():
    parser = ArgumentParser(
        description=(
            "Qualitative text segmentation of a "
            "Mip-NeRF360 scene using VALA language features"
        )
    )

    model = ModelParams(
        parser,
        sentinel=True,
    )

    pipeline = PipelineParams(parser)

    optimization = OptimizationParams(parser)

    parser.add_argument(
        "--iteration",
        default=30000,
        type=int,
    )

    parser.add_argument(
        "--prompt",
        default="bicycle",
        type=str,
    )

    parser.add_argument(
        "--mask_thresh",
        default=0.4,
        type=float,
    )

    parser.add_argument(
        "--num_knn",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--num_views",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    args = get_combined_args(parser)

    safe_state(args.quiet)

    device = "cuda"

    print("========================================")
    print("VALA qualitative segmentation")
    print("Scene:", args.model_path)
    print("Prompt:", args.prompt)
    print("Feature level:", args.feature_level)
    print("Threshold:", args.mask_thresh)
    print("KNN:", args.num_knn)
    print("========================================")

    # ==================================================
    # Load Gaussian model
    # ==================================================

    gaussians = GaussianModel(
        args.sh_degree
    )

    scene = Scene(
        model.extract(args),
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        include_feature=True,
    )

    print("Initial Gaussian model loaded.")
    print("Loaded iteration:", scene.loaded_iter)
    print(
        "Initial number of Gaussians:",
        gaussians.get_xyz.shape[0],
    )

    # ==================================================
    # Load VALA language checkpoint
    # ==================================================

    checkpoint_path = os.path.join(
        args.model_path,
        "none",
        (
            f"chkpnt{args.iteration}"
            f"_langfeat_{args.feature_level}"
            f"_stochastic_gate.pth"
        ),
    )

    print("Loading language checkpoint:")
    print(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Language checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # This is the same loading mechanism used by VALA.
    model_params, first_iter = torch.load(
        checkpoint_path,
        map_location=device,
    )

    print(
        "Checkpoint loaded. "
        f"Checkpoint iteration: {first_iter}"
    )

    # IMPORTANT:
    #
    # restore_language_features() replaces
    # _xyz, _features_dc, _features_rest,
    # _scaling, _rotation, _opacity and
    # _language_feature with the contents
    # of the language checkpoint.
    #
    # The language checkpoint therefore contains
    # the PRUNED Gaussian model itself.

    gaussians.restore_language_features(
        model_params,
        optimization.extract(args),
    )

    # ==================================================
    # Verify restored model
    # ==================================================

    num_gaussians = gaussians.get_xyz.shape[0]

    language_features = (
        gaussians.get_language_feature()
    )

    num_features = language_features.shape[0]

    print("----------------------------------------")
    print("Restored language Gaussian model.")
    print(
        "Number of Gaussians:",
        num_gaussians,
    )
    print(
        "Language feature shape:",
        tuple(language_features.shape),
    )
    print("----------------------------------------")

    if num_gaussians != num_features:
        raise RuntimeError(
            "FATAL: Gaussian / language-feature mismatch!\n"
            f"Gaussians: {num_gaussians}\n"
            f"Language features: {num_features}"
        )

    print(
        "Gaussian / language-feature count matches."
    )

    # ==================================================
    # Load OpenCLIP
    # ==================================================

    print("Loading OpenCLIP...")

    clip_model = OpenCLIPNetwork(
        device
    )

    print("OpenCLIP loaded.")

    # ==================================================
    # Compute Gaussian mask
    # ==================================================

    with torch.no_grad():
        gaussian_mask = compute_gaussian_mask(
            gaussians,
            clip_model,
            args.prompt,
            threshold=args.mask_thresh,
            num_knn=args.num_knn,
        )

    ply_path = os.path.join(args.output_dir, "segmented_gaussians.ply")
    save_segmented_gaussians_ply(gaussians, gaussian_mask, ply_path)

    # ==================================================
    # Background
    # ==================================================

    bg_color = (
        [1, 1, 1]
        if args.white_background
        else [0, 0, 0]
    )

    background = torch.tensor(
        bg_color,
        dtype=torch.float32,
        device=device,
    )

    # ==================================================
    # Output directory
    # ==================================================

    if args.output_dir is None:
        output_dir = os.path.join(
            args.model_path,
            "qualitative",
            args.prompt.replace(" ", "_"),
            f"level_{args.feature_level}",
            f"threshold_{args.mask_thresh}",
        )

    else:
        output_dir = args.output_dir

    print("Output directory:")
    print(output_dir)

    # ==================================================
    # Select cameras
    # ==================================================

    test_views = scene.getTestCameras()

    print(
        "Number of test cameras:",
        len(test_views),
    )

    if len(test_views) == 0:
        print(
            "No test cameras found. "
            "Using training cameras."
        )

        test_views = scene.getTrainCameras()

    print(
        "Rendering",
        min(len(test_views), args.num_views),
        "views."
    )

    # ==================================================
    # Render
    # ==================================================

    render_views(
        test_views,
        gaussians,
        pipeline.extract(args),
        background,
        gaussian_mask,
        output_dir,
        args.num_views,
    )

    print("========================================")
    print("Done.")
    print("Results:", output_dir)
    print("========================================")

def save_segmented_gaussians_ply(gaussians, mask, output_path):
    from plyfile import PlyData, PlyElement
    import numpy as np

    mask = mask.detach().cpu().bool()

    xyz = gaussians.get_xyz.detach().cpu()[mask].numpy()

    normals = np.zeros_like(xyz)

    features_dc = (
        gaussians._features_dc
        .detach()
        .cpu()[mask]
        .numpy()
        .reshape(xyz.shape[0], -1)
    )

    features_rest = (
        gaussians._features_rest
        .detach()
        .cpu()[mask]
        .numpy()
        .reshape(xyz.shape[0], -1)
    )

    opacity = (
        gaussians._opacity
        .detach()
        .cpu()[mask]
        .numpy()
        .reshape(xyz.shape[0], -1)
    )

    scaling = (
        gaussians._scaling
        .detach()
        .cpu()[mask]
        .numpy()
        .reshape(xyz.shape[0], -1)
    )

    rotation = (
        gaussians._rotation
        .detach()
        .cpu()[mask]
        .numpy()
        .reshape(xyz.shape[0], -1)
    )

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]

    dtype += [
        (f"f_dc_{i}", "f4")
        for i in range(features_dc.shape[1])
    ]

    dtype += [
        (f"f_rest_{i}", "f4")
        for i in range(features_rest.shape[1])
    ]

    dtype += [("opacity", "f4")]

    dtype += [
        (f"scale_{i}", "f4")
        for i in range(scaling.shape[1])
    ]

    dtype += [
        (f"rot_{i}", "f4")
        for i in range(rotation.shape[1])
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)

    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]

    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]

    for i in range(features_dc.shape[1]):
        elements[f"f_dc_{i}"] = features_dc[:, i]

    for i in range(features_rest.shape[1]):
        elements[f"f_rest_{i}"] = features_rest[:, i]

    elements["opacity"] = opacity[:, 0]

    for i in range(scaling.shape[1]):
        elements[f"scale_{i}"] = scaling[:, i]

    for i in range(rotation.shape[1]):
        elements[f"rot_{i}"] = rotation[:, i]

    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)

    print(f"Saved {xyz.shape[0]} segmented Gaussians to:")
    print(output_path)


if __name__ == "__main__":
    main()
