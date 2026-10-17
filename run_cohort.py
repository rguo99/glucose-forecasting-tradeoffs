#!/usr/bin/env python3
"""
run_cohort.py
=============
Fit one model for every user in the analysed cohort (or one shard of it).

Each user is fitted in its own subprocess (real_data_evaluation.py), one at a
time per listed GPU index. Results go to
    <EXPERIMENTS_DIR>/batch_<model>[_scale<tag>][_shard<k>of<N>]_<timestamp>/

Usage:
  python run_cohort.py --model global --gpus 0                     # Bayesian Loop, all users
  python run_cohort.py --model global --gpus 0 --shard 3/395       # one user (Slurm array task)
  python run_cohort.py --model global --gpus 0 --prior-scale 0.10  # prior sensitivity
  python run_cohort.py --model lstm_ensemble --gpus 0,1 --shard 1/2
  python run_cohort.py --model global --gpus 0 --only-uids 829,1012 --tag smoke

Models: global (Bayesian Loop), ar1_pure, ar1_rls, lstm, lstm_ensemble,
lstm_conformal, transformer, transformer_ensemble, transformer_conformal.
Bayesian Loop, Pure AR1 and AR1-RLS run on CPU (export CUDA_VISIBLE_DEVICES="");
the six neural models run on GPU.
"""
import os
import re
import glob
import argparse

# GPU determinism for the neural models: cuBLAS must use a fixed workspace,
# and this has to be set before the per-user subprocess initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Transformer-MC training protocol used in the paper: early stopping disabled,
# checkpoint chosen on a held-out validation fold. Read only by
# models/transformer_mc.py; Transformer-Ensemble/-Conformal and the LSTM models
# use their own defaults (patience 100, no validation selection).
os.environ.setdefault("TRANSFORMER_PATIENCE_OVERRIDE", "100000")
os.environ.setdefault("TRANSFORMER_USE_VAL_SELECTION_OVERRIDE", "1")

import config  # noqa: E402
import eval_config  # noqa: E402
import run_experiments as re_mod  # noqa: E402


def completed_users(model, prior_tag):
    """User ids that already have a result for this model and prior scale in any
    batch_<model><prior_tag>[_shard<k>of<N>]_<timestamp> directory."""
    pat = re.compile(rf"^batch_{re.escape(model)}{re.escape(prior_tag)}(_shard\d+of\d+)?_\d{{4}}-\d\d-\d\d_")
    done = set()
    for d in glob.glob(os.path.join(config.EXPERIMENTS_DIR, "batch_*")):
        if pat.match(os.path.basename(d)):
            for p in glob.glob(os.path.join(d, "*", "results", "user_*_results.pkl")):
                done.add(os.path.basename(p)[len("user_"):-len("_results.pkl")])
    return done


MODEL_CMD_FNS = {
    "global":                 (re_mod.global_cmd, "Global"),
    "lstm":                   (re_mod.lstm_cmd, "LSTM_MC"),
    "lstm_ensemble":          (re_mod.lstm_ensemble_cmd, "LSTM_Ensemble"),
    "transformer_ensemble":   (re_mod.transformer_ensemble_cmd, "Transformer_Ensemble"),
    "lstm_conformal":         (re_mod.lstm_conformal_cmd, "LSTM_Conformal"),
    "transformer_conformal":  (re_mod.transformer_conformal_cmd, "Transformer_Conformal"),
    "transformer":            (re_mod.transformer_cmd, "Transformer_MC"),
    "ar1_pure":               (re_mod.ar1_baseline_cmd, "AR1_Pure"),
    "ar1_rls":                (re_mod.rls_ar1_baseline_cmd, "AR1_RLS"),
}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--model", required=True, choices=list(MODEL_CMD_FNS.keys()))
    p.add_argument("--gpus", type=str, default="0",
                   help="Comma-separated local GPU indices (one concurrent user per index). "
                        "For CPU-only models pass a single index, e.g. 0.")
    p.add_argument("--shard", type=str, default=None, help="k/N, 1-indexed")
    p.add_argument("--prior-scale", type=float, default=eval_config.MAIN_PRIOR_SCALE,
                   help="Log-space scale of the Bayesian Loop ISF/CR priors "
                        "(0.30 main analysis; 0.10 and 0.50 for prior sensitivity).")
    p.add_argument("--only-uids", type=str, default=None,
                   help="Comma-separated user ids to restrict to (smoke tests).")
    p.add_argument("--tag", type=str, default=None,
                   help="Write to <EXPERIMENTS_DIR>/<tag>_<model>_<timestamp> instead of a "
                        "batch_* dir, so test runs are never picked up by the evaluation.")
    p.add_argument("--force", action="store_true",
                   help="Refit users that already have results for this model and prior scale "
                        "(by default they are skipped, so a resubmitted or requeued shard only "
                        "fits the missing users and never creates duplicate results).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]

    # Divide the CPU allocation across the concurrent per-user subprocesses.
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 20))
    re_mod.NUM_CPUS = max(2, total_cpus // len(gpus))
    print(f"[cpu] total={total_cpus} gpus={len(gpus)} -> NUM_CPUS={re_mod.NUM_CPUS} per subprocess")

    data_dir, curation_dir = config.data_dir(), config.filter_dir()
    user_ids = eval_config.load_eligible_users(data_dir, curation_dir)

    if args.only_uids:
        keep = set(args.only_uids.split(","))
        user_ids = [u for u in user_ids if str(u) in keep]
        print(f"--only-uids -> {user_ids}")

    if args.shard:
        k_str, nsh_str = args.shard.split("/")
        k, nsh = int(k_str), int(nsh_str)
        assert 1 <= k <= nsh
        total = len(user_ids)
        lo = (total * (k - 1)) // nsh
        hi = (total * k) // nsh
        user_ids = user_ids[lo:hi]
        print(f"SHARD {k}/{nsh}: users [{lo}:{hi}] of {total} -> {len(user_ids)} users "
              f"({user_ids[0] if user_ids else 'none'}..{user_ids[-1] if user_ids else 'none'})")

    prior_tag = (f"_scale{eval_config.scale_tag(args.prior_scale)}"
                 if args.prior_scale != eval_config.MAIN_PRIOR_SCALE else "")
    if not args.tag and not args.force:
        done = completed_users(args.model, prior_tag)
        skipped = [u for u in user_ids if str(u) in done]
        if skipped:
            user_ids = [u for u in user_ids if str(u) not in done]
            print(f"Skipping {len(skipped)} users with existing results (--force to refit); "
                  f"{len(user_ids)} to fit")
        if not user_ids:
            print("Nothing to fit.")
            return
    shard_tag = f"_shard{args.shard.replace('/', 'of')}" if args.shard else ""
    name = f"batch_{args.model}{prior_tag}{shard_tag}_{re_mod.RUN_TIMESTAMP}"
    if args.tag:
        name = f"{args.tag}_{args.model}{prior_tag}{shard_tag}_{re_mod.RUN_TIMESTAMP}"
    batch_dir = os.path.join(config.EXPERIMENTS_DIR, name)
    print(f"Data        : {data_dir}")
    print(f"Curation    : {curation_dir}")
    print(f"Prior scale : {args.prior_scale}")
    print(f"Batch dir   : {batch_dir}")
    print(f"GPUs        : {gpus}")

    cmd_fn_raw, model_name = MODEL_CMD_FNS[args.model]
    cmd_fn = lambda uid: cmd_fn_raw(uid, data_dir, curation_dir, batch_dir)  # noqa: E731

    _, failed = re_mod.run_queue(user_ids, cmd_fn, model_name, batch_dir, gpus,
                                 dry_run=args.dry_run, prior_scale=args.prior_scale)
    if failed:
        print(f"[WARN] {len(failed)} users failed: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
