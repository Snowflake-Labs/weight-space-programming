# Car-policy weight-edit results

Each arm uses the same reconstructed Kinetix CPU evaluator, 1,024 paired episodes, and seed `20261105`.

| Arm | Successes | Red collisions | Timeouts |
|---|---:|---:|---:|
| Original policy | 0 | 0 | 1,024 |
| Astra, unguided one-shot | 0 | 39 | 985 |
| Published repair control | 1,024 | 0 | 0 |
| Astra, guided one-shot | 1,024 | 0 | 0 |

The guided arm added one generic instruction asking Astra to consider state-dependent control and check its numerical edits. Astra received no published answer, published weights, or control results. Its first response was frozen before evaluation.

The upstream prompt, policy weights, and published answer are not redistributed here because the source repository does not declare a license. See `RikiyaT/lm_understand-logs` at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

This is a reconstruction of the paper's evaluator, not a bit-exact reproduction, and one successful generation does not establish reliability.
