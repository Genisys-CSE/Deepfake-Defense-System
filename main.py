"""
DeepShield — Anti-Deepfake Protection System v2.0

CLI entry point.  Protects images against deepfake creation by applying
multi-model adversarial perturbations, DCT frequency noise, and texture
disruption — all invisible to the human eye.

Usage
-----
    # Protect with balanced preset (recommended)
    python main.py --input photo.jpg --out protected.jpg

    # Maximum protection
    python main.py --input photo.jpg --out protected.jpg --preset maximum

    # Stealth (near-invisible)
    python main.py --input photo.jpg --out protected.jpg --preset stealth

    # Force CPU (if CUDA causes issues)
    python main.py --input photo.jpg --out protected.jpg --device cpu

    # Compare two images (e.g. original vs deepfake)
    python main.py --eval-only --input original.jpg --compare-img deepfake.jpg
"""

import argparse
import sys
import os
import time

# ── Redirect all model downloads to project-local cache ─────────────
# This MUST run before any torch/torchvision imports.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_PROJECT_DIR, 'model_cache')
os.environ.setdefault('TORCH_HOME', os.path.join(_CACHE_DIR, 'torch'))
os.environ.setdefault('FACENET_CACHE', _CACHE_DIR)

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from config import PRESETS
from pipeline import ProtectionPipeline


BANNER = r"""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║     ██████╗ ███████╗███████╗██████╗                  ║
║     ██╔══██╗██╔════╝██╔════╝██╔══██╗                 ║
║     ██║  ██║█████╗  █████╗  ██████╔╝                 ║
║     ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝                  ║
║     ██████╔╝███████╗███████╗██║                      ║
║     ╚═════╝ ╚══════╝╚══════╝╚═╝                     ║
║          ███████╗██╗  ██╗██╗███████╗██╗     ██████╗  ║
║          ██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗ ║
║          ███████╗███████║██║█████╗  ██║     ██║  ██║ ║
║          ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║ ║
║          ███████║██║  ██║██║███████╗███████╗██████╔╝ ║
║          ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ ║
║                                                      ║
║     Anti-Deepfake Protection System  v2.0            ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(
        description="DeepShield — Protect images against deepfake creation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  maximum   High disruption, slight artifacts acceptable
  balanced  Good disruption with minimal visual impact (default)
  stealth   Near-invisible perturbation

Examples:
  python main.py --input photo.jpg --out protected.jpg
  python main.py --input photo.jpg --out protected.jpg --preset maximum
  python main.py --eval-only --input original.jpg --compare-img deepfake.jpg
        """,
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--out", type=str, default="protected.jpg",
                        help="Path to save protected image (default: protected.jpg)")
    parser.add_argument("--preset", type=str, default="balanced",
                        choices=list(PRESETS.keys()),
                        help="Protection preset (default: balanced)")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Force a specific device (default: auto-detect)")

    # Overrides
    parser.add_argument("--steps", type=int, default=None,
                        help="Override PGD iteration steps")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override L-inf epsilon (0–1 scale, e.g. 0.016)")

    # Eval mode
    parser.add_argument("--eval-only", action='store_true',
                        help="Compare two images without applying protection")
    parser.add_argument("--compare-img", type=str,
                        help="Second image path for --eval-only comparison")

    # Toggles
    parser.add_argument("--no-freq", action='store_true',
                        help="Disable DCT frequency perturbation")
    parser.add_argument("--no-texture", action='store_true',
                        help="Disable gram-matrix texture disruption")
    parser.add_argument("--no-adversarial", action='store_true',
                        help="Disable adversarial attack (run frequency-only)")

    args = parser.parse_args()

    print(BANNER)

    # ── Validate inputs ─────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"  ✗ Input file not found: {args.input}")
        sys.exit(1)

    if args.eval_only and not args.compare_img:
        print("  ✗ --compare-img is required with --eval-only")
        sys.exit(1)

    if args.eval_only and args.compare_img and not os.path.isfile(args.compare_img):
        print(f"  ✗ Comparison file not found: {args.compare_img}")
        sys.exit(1)

    # ── Load preset config ──────────────────────────────────────────────
    config = PRESETS[args.preset]
    print(f"  Preset : {args.preset}")
    print(f"  ε      : {config.epsilon * 255:.1f}/255  ({config.epsilon:.4f})")
    print(f"  Steps  : {config.steps}")
    print(f"  EOT    : {config.eot_samples} augmentations/step")

    # Apply overrides
    if args.steps is not None:
        config.steps = args.steps
        print(f"  ↳ steps overridden to {args.steps}")
    if args.epsilon is not None:
        config.epsilon = args.epsilon
        print(f"  ↳ epsilon overridden to {args.epsilon}")
    if args.no_freq:
        config.use_frequency = False
        print("  ↳ DCT frequency perturbation disabled")
    if args.no_texture:
        config.use_texture = False
        print("  ↳ Gram-matrix texture disruption disabled")
    if args.no_adversarial:
        config.use_adversarial = False
        print("  ↳ Adversarial attack disabled (frequency-only mode)")

    # ── Run ─────────────────────────────────────────────────────────────
    start_time = time.time()
    pipeline = ProtectionPipeline(config, device=args.device)

    if args.eval_only:
        pipeline.evaluate(args.input, args.compare_img)
    else:
        pipeline.protect(args.input, args.out)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n⏱  Total time: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
