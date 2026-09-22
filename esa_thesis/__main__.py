"""Command-line entry point; analysis tools are imported only when requested."""
import argparse
import importlib
import sys

COMMANDS = {'combine-specialist': 'esa_thesis.combine_specialist', 'forecast-develop': 'esa_thesis.forecast_development', 'develop': 'esa_thesis.development', 'compare-scoring': 'esa_thesis.evaluation.compare_scoring', 'recalibrate': 'esa_thesis.evaluation.recalibrate', 'doctor': 'esa_thesis.doctor', 'correlation': 'esa_thesis.analysis.correlation', 'importance': 'esa_thesis.analysis.importance', 'representatives': 'esa_thesis.analysis.representatives', 'score': 'esa_thesis.evaluation.legacy_scores', 'evaluate': 'esa_thesis.evaluation.legacy_evaluation', 'evaluate-ensemble': 'esa_thesis.evaluation.mlp_ensemble', 'reselect': 'esa_thesis.evaluation.reselect', 'subset-v1': 'esa_thesis.evaluation.subset_v1', 'subset': 'esa_thesis.evaluation.subset', 'coverage': 'esa_thesis.evaluation.coverage', 'evt': 'esa_thesis.calibration.evt', 'dspot': 'esa_thesis.calibration.dspot'}

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        historical = {'score', 'evaluate', 'evaluate-ensemble', 'reselect', 'subset-v1', 'subset', 'coverage', 'evt', 'dspot'}
        if argv[0] in historical and '--help' not in argv and '-h' not in argv:
            if '--allow-legacy-metric' not in argv:
                raise SystemExit('Historical evaluator: use recalibrate for corrected ESA scores. To reproduce old results only, pass --allow-legacy-metric.')
            argv.remove('--allow-legacy-metric')
            print('WARNING: historical evaluation; outputs are not official ESA scores.', file=sys.stderr)
        module = importlib.import_module(COMMANDS[argv[0]])
        previous = sys.argv
        try:
            sys.argv = [COMMANDS[argv[0]], *argv[1:]]
            return module.main()
        finally:
            sys.argv = previous
    parser = argparse.ArgumentParser(description="ESA satellite telemetry thesis experiments")
    subs = parser.add_subparsers(dest="command", required=True)
    train = subs.add_parser("train", help="Train or resume the configured experiments")
    train.add_argument("--dry-run", action="store_true", help="Show configuration without training")
    train.add_argument("--runs", nargs="+", help="Run names from config.RUN_SPECS")
    train.add_argument("--output-root", help="Separate experiment output directory")
    for name in COMMANDS:
        subs.add_parser(name, help="Run " + name + " (use --help for options)")
    args = parser.parse_args(argv)
    if args.command == "train":
        from . import config
        specs = config.RUN_SPECS
        if args.runs:
            missing = set(args.runs) - {s["run"] for s in specs}
            if missing:
                parser.error("Unknown runs: " + ", ".join(sorted(missing)))
            specs = [s for s in specs if s["run"] in args.runs]
        from .paths import project_path
        output = project_path(args.output_root) if args.output_root else config.OUT_ROOT
        if args.dry_run:
            print("Device:", config.DEVICE)
            print("Train:", config.TRAIN_FILE)
            print("Test:", config.TEST_FILE)
            print("Output:", output)
            for spec in specs:
                print(spec["run"], "features=", len(spec["features"]))
            return
        from . import training
        training.RUN_SPECS = specs
        training.OUT_ROOT = output
        training.main()

if __name__ == "__main__":
    raise SystemExit(main())
