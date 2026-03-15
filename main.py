"""
DeepShield — Anti-Deepfake Protection System

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

    # Compare two images (e.g. original vs deepfake)
    python main.py --eval-only --input original.jpg --compare-img deepfake.jpg
"""

import argparse
import sys
import os

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
        """,
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--out", type=str, default="protected.jpg",
                        help="Path to save protected image (default: protected.jpg)")
    parser.add_argument("--preset", type=str, default="balanced",
                        choices=list(PRESETS.keys()),
                        help="Protection preset (default: balanced)")

    # Overrides
    parser.add_argument("--steps", type=int, default=None,
                        help="Override PGD steps")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override L-inf epsilon (in 0-1 scale, e.g. 0.016)")

    # Eval mode
    parser.add_argument("--eval-only", action='store_true',
                        help="Compare two images without applying protection")
    parser.add_argument("--compare-img", type=str,
                        help="Second image for --eval-only comparison")

    # Toggles
    parser.add_argument("--no-freq", action='store_true',
                        help="Disable DCT frequency perturbation")
    parser.add_argument("--no-texture", action='store_true',
                        help="Disable gram-matrix texture disruption")
    parser.add_argument("--no-adversarial", action='store_true',
                        help="Disable adversarial attack (freq-only mode)")

    args = parser.parse_args()

    print(BANNER)

    # ── Load preset config ──────────────────────────────────────────────
    config = PRESETS[args.preset]
    print(f"  Preset: {args.preset}")

    # Apply overrides
    if args.steps is not None:
        config.steps = args.steps
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.no_freq:
        config.use_frequency = False
    if args.no_texture:
        config.use_texture = False
    if args.no_adversarial:
        config.use_adversarial = False

    # ── Run ─────────────────────────────────────────────────────────────
    pipeline = ProtectionPipeline(config)

    if args.eval_only:
        if not args.compare_img:
            print("  ✗ --compare-img is required with --eval-only")
            sys.exit(1)
        pipeline.evaluate(args.input, args.compare_img)
    else:
        pipeline.protect(args.input, args.out)


if __name__ == "__main__":
    main()
