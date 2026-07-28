# Official VBench custom-input comparison

Evaluated with the official `Vchitect/VBench` implementation (version 0.1.5 source), using the same 8 prompts and videos for every method.

| Method | Subject consistency | Background consistency | Motion smoothness | Dynamic degree | Aesthetic quality | Imaging quality | Custom 6-dim mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.964307 | 0.947326 | 0.986766 | 0.750000 | 0.625441 | 0.665321 | 0.823194 |
| SLA | 0.951465 | 0.947240 | 0.980382 | 0.875000 | 0.612709 | 0.666179 | 0.838829 |
| SLA + Q 2:4 | 0.970070 | 0.959052 | 0.992890 | 0.000000 | 0.597919 | 0.609991 | 0.688320 |

The custom 6-dimension mean is an unweighted arithmetic mean created for this experiment. It is not the official 16-dimension VBench Total Score. Official VBench custom-input mode supports these six dimensions. `dynamic_degree` is a binary decision per video, so with 8 videos its aggregate changes in increments of 0.125.

The SLA + Q 2:4 result has high frame/subject consistency and motion smoothness but a dynamic-degree score of zero for all 8 clips. This indicates temporal/static collapse rather than a quality win: smoother-looking frames are not moving enough under the VBench motion threshold.
