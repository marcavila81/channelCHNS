# AGENTS.md — channelCHNS project rules

## Where things run (READ FIRST, EVERY SESSION)

- **This repo has two homes. Never confuse them.**
  - **Marc's Mac (local clone)**: editing, commits, git work, and ALL Python
    analyses (numpy, matplotlib, pandas, compare_YL.py, plotting, CSV/YL
    reference work). Python runs locally on the Mac. The Mac has NO NVIDIA
    GPU: never compile with nvcc here, never run wnCH_NS here, never suggest
    running either locally.
  - **The cluster (ZARM, B200 node)**: ALL builds (make/nvcc) and ALL wnCH_NS
    runs happen there, over SSH. Connection details are in AGENTS.local.md
    (gitignored). READ IT at the start of any session that builds or runs.
    If it is missing, ASK Marc — do not guess, and do not run on the Mac.
    Runs are long (hours): launch with nohup or inside tmux, then poll;
    never block a session on a live run.
- Flow for changes needing both: edit + commit locally, git pull on the
  cluster, build and run there, scp outputs (ycg.txt, diag_v_int.txt,
  pln_*.bin, *.png) back to the Mac, analyse locally.
- Validation is sacred: never modify or delete golden-case outputs; compare
  with tolerance, not bitwise; the oracle-validated tag is the reference.

## Conventions
- Develop on main, no feature branches.
- case.in = runtime inputs; Rad/rho_r/mu_r/Re/dt are compile-time via
  make FLAGS="-DRad=... -Drho_r=..." (#ifndef-guarded in head.h).
- Nondimensionalisation: U=sqrt(gR), Fr=1, the carrier (liquid) is the
  reference phase, rho_r<=1, internal Weber slot We = Bo/(1-rho_r).
- SECURITY: this repository is PUBLIC. Never commit hostnames, usernames,
  SSH details, credentials or cluster-specific absolute paths — they belong
  in AGENTS.local.md only. Before committing docs or scripts, scan them.
