# Claim 4 source audit

Primary HTML:

- URL: `https://ar5iv.labs.arxiv.org/html/2605.31261`
- Retrieved: `2026-07-23T06:09:31Z`
- SHA-256: `e6ed4e6be5e80ab9f65eac02be852add36872fc660fa780cfcf1e68ac0f94250`
- Section 6.2 anchor: `S6.SS2`
- Appendix E.2.1 anchor: `A5.SS2.SSS1`
- Appendix E.2.2 anchor: `A5.SS2.SSS2`

Released arXiv e-print:

- URL: `https://export.arxiv.org/e-print/2605.31261`
- Retrieved with explicit audit user agent on `2026-07-23`
- SHA-256: `f85693bad9adf7d13c2ae827178bbb052f54b7460dcfd0261802f9bf6eac214c`
- Contains TeX and raster/PDF figures, but no training code or raw CSV/JSON curves.

The source fixes 12 states, four actions, four beacons, epsilon 0.05,
goal states 2/3, trap states 6/11/12, PPO Table 1 settings, S5 settings,
30 million timesteps, and eight seeds. It does not numerically specify episode
length `K`, beacon angles, the eight seed values, or full `Q(a)` entries for
CW2/CCW1/CCW2. It provides only the CW1 entries as a sample.

The author publication page was retrieved on `2026-07-23T07:19:43Z`
(SHA-256 `7b10cc8790d1f3e78647d6ddf311b3d1055ae9e3252e3d92b3c64735114475f1`).
Unlike nearby publications, this paper has no Code link. Exact-title GitHub
repository search returned zero repositories; searches for `"RingWorld"
"Deep ALF"` returned zero code results. The cited
`https://github.com/luchris429/s5rl` implements the earlier S5/PPO work, not
this paper's RingWorld or Deep ALF.
