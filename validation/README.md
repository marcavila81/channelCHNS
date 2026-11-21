# Validation and sanity checks for channelCHNS

This directory lets anyone who downloads the code check their build/run
against reference results, and gives fast smoke tests for solver changes.

## Layout

    case1_static_cap_th50_Bo0.json   full validation case (oracle + reference metrics)
    sanity_th50_Bo0_coarse.json      fast smoke test (coarse grid, larger Cn, short run)
    run_case.py                      json -> case.in -> run binary -> analysis
    analysis.py                      metrics + oracle check + plots (writes metrics.json + PNGs)
    reference/                       committed reference outputs of case1
                                     (metrics.json, PNGs, ycg.txt, diag_v_int.txt, run_info.txt)

## Validation case 1 (full)

Requires a GPU build of the code (`make -C ../code`). Then:

    cd validation
    python run_case.py case1_static_cap_th50_Bo0.json --workdir /path/to/run/dir

This writes `case.in`, runs `../code/wnCH_NS case.in` in the workdir
(outputs land in `<workdir>/output/`), then runs `analysis.py`, which
prints each metric with PASS/FAIL against the oracle and the delta vs the
reference values, and writes `metrics.json` + plots into the output dir.

Exit code 0 = oracle PASS. Reference outputs from the original run are in
`reference/` for comparison (do not commit the big .bin fields; they are
reproducible from the JSON and are git-ignored).

## Sanity case (fast, for solver tweaks)

    cd validation
    python run_case.py sanity_th50_Bo0_coarse.json --workdir /path/to/sanity \
        --build-flags "-DNY=65 -DNX=64 -DNZ=33 -DCn=0.05 -Ddt=1e-4"

The `build_flags` rebuild the code with a coarse grid (NY=65, NX=64, NZ=33),
larger Cahn number (Cn=0.05) and larger time step (dt=1e-4); the run itself is
short (t_tot=0.02), so the whole check takes seconds. Its oracle is loose on
purpose: it catches crashes, NaNs, missing files and large regressions.

Typical workflow when tweaking the solver: run the sanity case before and
after the change and compare `output_sanity/metrics.json` (or run the full
case1 for accuracy-critical changes).

## Case JSON format

    name / description           free text
    binary                       path to wnCH_NS relative to this file
    build_flags                  optional Makefile FLAGS for a special build
    runtime_input                keys written to case.in (th_s, Bo, t_tot,
                                 dt_mp, dt_rms, dt_pln, dt_bin, outdir)
    compile_time                 the head.h setup the case assumes (grid, Cn, ...)
    oracle                       metric -> {expected, abs_tol} or {max}
    hard_checks                  files that must exist, all_finite flag
    reference                    reference metric values (case1)
