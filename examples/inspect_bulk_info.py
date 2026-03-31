"""
examples/inspect_bulk_info.py

Load the evolution history from a completed JETSCAPE-FNO run and inspect it
as numpy arrays and PyTorch tensors.  Produces matplotlib plots of energy
density, temperature, and velocity field slices at selected proper-time steps.

Usage (two modes):

  1. Run a fresh simulation (Mode A, XML-driven), then inspect:
       conda activate js_fno
       python examples/inspect_bulk_info.py \\
           --main config/jetscape_main.xml \\
           --user config/jetscape_user_AA_dukeTune.xml

  2. Pass in a pre-computed bulk_info numpy file (saved by a previous run):
       python examples/inspect_bulk_info.py --load bulk_info.npy

Optional flags:
    --main      Path to main XML config (default: config/jetscape_main.xml)
    --user      Path to user XML config (default: config/jetscape_user_AA_dukeTune.xml)
    --load      Load a previously saved .npy file instead of running a sim
    --save      Save the extracted numpy array to this path (e.g. bulk_info.npy)
    --n-feat    Number of features to extract (default: 3)
    --outdir    Directory for output plots (default: inspect_bulk_info_out/)
    --no-show   Do not open interactive plot windows (always saves to disk)
    --events    Number of events for the in-process simulation (default: 1)
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make sure the python package is importable ────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--main",   default=os.path.join(_REPO_ROOT, "config", "jetscape_main.xml"))
    p.add_argument("--user",   default=os.path.join(_REPO_ROOT, "config", "jetscape_user_AA_dukeTune.xml"))
    p.add_argument("--load",   default=None,
                   help="Load bulk_info from a previously saved .npy file.")
    p.add_argument("--save",   default=None,
                   help="Save extracted bulk_info numpy array to this file.")
    p.add_argument("--n-feat", type=int, default=3, dest="n_features",
                   help="Number of fluid features to extract (3 or 4).")
    p.add_argument("--outdir", default="inspect_bulk_info_out",
                   help="Directory to write plot images.")
    p.add_argument("--no-show", action="store_true",
                   help="Never open interactive plot windows.")
    p.add_argument("--events", type=int, default=1)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data acquisition
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(args: argparse.Namespace):
    """
    Run an in-process JETSCAPE simulation (Mode A) and return the hydro module.

    Returns
    -------
    js : JetScape
        The framework object after Finish().
    hydro : FluidDynamics
        The hydro module from which bulk_info can be extracted.
    """
    from jetscape.run_jetscape import run_automatic
    from jetscape.pyjetscape_core import FluidDynamics

    js = run_automatic(args.main, args.user)

    # Walk the task list to find the FluidDynamics module
    tasks = js.GetTasks()
    hydro = None
    for t in tasks:
        if isinstance(t, FluidDynamics):
            hydro = t
            break

    return js, hydro


def extract_bulk_numpy(hydro, n_features: int) -> "np.ndarray":
    """
    Extract EvolutionHistory as a numpy array.

    Returns
    -------
    np.ndarray, shape (ntau, nx, ny, n_features), dtype float32
    """
    from jetscape.utils import bulk_info_to_numpy
    bulk = hydro.get_bulk_info()
    arr  = bulk_info_to_numpy(bulk, n_features)
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_matplotlib(no_show: bool):
    import matplotlib
    if no_show or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_energy_density_slices(
    arr: "np.ndarray",
    dtau: float,
    tau_min: float,
    outdir: str,
    plt,
) -> None:
    """
    Plot energy density (feature 0) at τ_start, τ_mid, τ_end.

    Parameters
    ----------
    arr     : (ntau, nx, ny, n_features)
    dtau    : proper-time step in fm/c
    tau_min : starting proper time in fm/c
    outdir  : output directory
    plt     : matplotlib.pyplot module
    """
    import numpy as np

    ntau, nx, ny, _ = arr.shape
    tau_values = tau_min + dtau * np.arange(ntau)

    step_indices = [0, ntau // 4, ntau // 2, 3 * ntau // 4, ntau - 1]
    step_indices = sorted(set(max(0, min(ntau - 1, s)) for s in step_indices))

    n_cols = len(step_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, k in zip(axes, step_indices):
        ed = arr[k, :, :, 0]   # energy density
        vmin, vmax = 0.0, arr[:, :, :, 0].max()
        im = ax.imshow(
            ed.T, origin="lower", cmap="inferno",
            vmin=vmin, vmax=vmax,
            extent=[-nx / 2, nx / 2, -ny / 2, ny / 2],
        )
        fig.colorbar(im, ax=ax, label="e  [GeV/fm³]", shrink=0.85)
        ax.set_title(f"τ = {tau_values[k]:.2f} fm/c")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

    fig.suptitle("Energy density evolution", fontsize=14)
    fig.tight_layout()

    path = os.path.join(outdir, "energy_density_slices.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_temperature_profile(
    arr: "np.ndarray",
    dtau: float,
    tau_min: float,
    outdir: str,
    plt,
) -> None:
    """
    Plot temperature (feature 1) integrated over the transverse plane vs τ.
    """
    if arr.shape[3] < 2:
        print("  [!] n_features < 2 — skipping temperature plot.")
        return

    import numpy as np

    ntau  = arr.shape[0]
    tau_v = tau_min + dtau * np.arange(ntau)
    T_max = arr[:, :, :, 1].max(axis=(1, 2))   # peak temperature vs τ
    T_avg = arr[:, :, :, 1].mean(axis=(1, 2))  # spatial average vs τ

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tau_v, T_max, label="T_max(τ)",  lw=2)
    ax.plot(tau_v, T_avg, label="T_avg(τ)",  lw=2, ls="--")
    ax.axhline(0.136, color="red", ls=":", label="T_freeze = 0.136 GeV")
    ax.set_xlabel("τ  [fm/c]")
    ax.set_ylabel("Temperature  [GeV]")
    ax.set_title("Temperature evolution (spatial max and mean)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    path = os.path.join(outdir, "temperature_profile.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_velocity_field(
    arr: "np.ndarray",
    dtau: float,
    tau_min: float,
    tau_step: int,
    outdir: str,
    plt,
) -> None:
    """
    Plot vx, vy velocity field (features 2, 3) at a single τ slice using quiver.
    """
    if arr.shape[3] < 4:
        print("  [!] n_features < 4 — skipping velocity quiver plot.")
        return

    import numpy as np

    ntau      = arr.shape[0]
    tau_step  = max(0, min(ntau - 1, tau_step))
    tau_val   = tau_min + dtau * tau_step

    vx = arr[tau_step, :, :, 2]
    vy = arr[tau_step, :, :, 3]
    ed = arr[tau_step, :, :, 0]

    # Downsample for quiver readability
    skip = max(1, arr.shape[1] // 20)
    X    = np.arange(0, arr.shape[1], skip)
    Y    = np.arange(0, arr.shape[2], skip)
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(ed.T, origin="lower", cmap="inferno", alpha=0.6)
    fig.colorbar(im, ax=ax, label="e  [GeV/fm³]", shrink=0.85)
    ax.quiver(
        Xg, Yg,
        vx[::skip, ::skip], vy[::skip, ::skip],
        scale=10.0, color="white", alpha=0.8,
    )
    ax.set_title(f"Velocity field at τ = {tau_val:.2f} fm/c")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.tight_layout()

    path = os.path.join(outdir, "velocity_field.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_torch_tensor_summary(bulk_info, n_features: int, outdir: str, plt) -> None:
    """
    Quick sanity check: convert to torch.Tensor and print/plot its statistics.
    """
    try:
        import torch
        from jetscape.utils import bulk_info_to_tensor
    except ImportError:
        print("  [!] PyTorch not available — skipping tensor summary.")
        return

    t = bulk_info_to_tensor(bulk_info, n_features)
    print(f"\n  torch.Tensor  shape={list(t.shape)}  dtype={t.dtype}")

    feat_names = ["energy_density", "temperature", "vx", "vy",
                  "entropy_density", "pressure"][:n_features]

    means  = t[0].mean(dim=(1, 2, 3)).tolist()
    stddev = t[0].std (dim=(1, 2, 3)).tolist()

    fig, ax = plt.subplots(figsize=(7, 3))
    xs = range(n_features)
    ax.bar(xs, means, yerr=stddev, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(feat_names, rotation=20, ha="right")
    ax.set_ylabel("Spatial & temporal mean")
    ax.set_title("Tensor feature statistics (mean ± std over all grid points)")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(outdir, "tensor_feature_stats.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def print_summary(arr: "np.ndarray", n_features: int) -> None:
    """Print a concise summary of the bulk_info array to stdout."""
    import numpy as np

    ntau, nx, ny, _ = arr.shape
    feature_names = ["energy_density", "temperature", "vx", "vy",
                     "entropy_density", "pressure"]

    print(f"\n{'='*55}")
    print(f"  bulk_info summary")
    print(f"{'='*55}")
    print(f"  Array shape : {arr.shape}  (ntau, nx, ny, n_features)")
    print(f"  dtype       : {arr.dtype}")
    print(f"  Memory      : {arr.nbytes / 1024**2:.2f} MB")
    print()
    for f in range(n_features):
        data = arr[:, :, :, f]
        name = feature_names[f] if f < len(feature_names) else f"feat_{f}"
        print(f"  {name:<20}  min={data.min():+.4f}  "
              f"max={data.max():+.4f}  mean={data.mean():+.4f}")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    import numpy as np

    os.makedirs(args.outdir, exist_ok=True)

    # ── Acquire data ──────────────────────────────────────────────────────────
    bulk_info_obj = None
    dtau    = 0.1    # fm/c — will be overridden if available from run
    tau_min = 1.0    # fm/c — will be overridden if available from run

    if args.load:
        print(f"Loading bulk_info from {args.load} ...")
        arr = np.load(args.load)
        print(f"  Loaded array shape: {arr.shape}")
    else:
        print("Running JETSCAPE simulation (Mode A) ...")
        js, hydro = run_simulation(args)

        if hydro is None:
            print("[!] Could not find a FluidDynamics module in the task list.")
            print("    Either use --load <file> or ensure your user XML includes a hydro module.")
            sys.exit(1)

        bulk_info_obj = hydro.get_bulk_info()

        # Try to read grid metadata
        try:
            dtau    = bulk_info_obj.dtau
            tau_min = bulk_info_obj.tau_min
        except AttributeError:
            pass  # use defaults above

        print(f"  Extracting {args.n_features} features from bulk_info ...")
        arr = extract_bulk_numpy(hydro, args.n_features)

    # ── Save raw array ────────────────────────────────────────────────────────
    if args.save:
        np.save(args.save, arr)
        print(f"  Saved numpy array → {args.save}")

    # ── Text summary ──────────────────────────────────────────────────────────
    print_summary(arr, args.n_features)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt = _setup_matplotlib(args.no_show)

    print("Generating plots ...")
    plot_energy_density_slices(arr, dtau, tau_min, args.outdir, plt)
    plot_temperature_profile(arr, dtau, tau_min, args.outdir, plt)
    plot_velocity_field(arr, dtau, tau_min, tau_step=arr.shape[0] // 2,
                        outdir=args.outdir, plt=plt)

    if bulk_info_obj is not None:
        plot_torch_tensor_summary(bulk_info_obj, args.n_features, args.outdir, plt)

    print(f"\nAll outputs written to: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
