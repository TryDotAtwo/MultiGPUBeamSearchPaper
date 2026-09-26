# Bilingual editorial release QA

## Accepted changes

- Preserved and verified F1–F6 from the side audit: Cube4 96 six-class positions → 56 piece tokens + CLS = 57 tokens, input LayerNorm and ordered blocks; unchanged 3,383,064 parameters and 8×H200 usage. DeepCubeA weight 0.6 and batches 10000/10000, no invented beam. Empty Top_B and b>0 restriction. Key-based owner routing. Three scalar implementations and corrected collaborator wording.
- Architecture/inventory tables now use wrapped columns, not resizebox shrinking. Boundary tables have compact headers and explicit unit/status legends. Scalar comparison increased from scriptsize to small. Removed repeated capacity recaps. Float placement is relaxed, bibliography starts only after all results; raggedbottom avoids stretched paragraph gaps. All eleven tables retained.
- C1/C2 captions and provenance distinguish historical comparison from boundary runs. Important additional finding: C1 is the internal solver timer; C2 wraps executable initialization as well as search, excluding build/export. Both divide by eight completed native steps, not last-depth duration. Neither values nor ratios were recomputed or pooled. Added raw C1 records/logs and C2 point log plus SCALAR_COHORTS.md to the publication package.

## Verification

- Final EN and RU compiled twice with existing pdflatex; pass transcripts retained. EN 10 pages, RU 11 pages. Both final logs have zero undefined references, overfull boxes, LaTeX errors, fatal stops or emergency stops. MiKTeX emits an external log-directory permission/update notice; compilation nevertheless succeeds and writes the PDFs.
- Visually inspected every final page: EN 1–10 and RU 1–11, including all tables, three diagrams and bibliography. No clipped text, overlapping cells or missing glyphs observed. Dense prior-work tables remain readable at PDF zoom; no further font reduction used. Wide float tables still occupy dedicated result pages; not claimed to have identical pagination across languages.
- Structural checks: 22 ordered labels, 18 citation commands, 14 references, eight equation/align environments and eleven numeric table sequences match across languages. The checker normalizes English 'six-class' to the digit 6 only for this parity comparison. Per-language table digits are unchanged from the pre-layout staged sources. Semantic review covered all changed passages and retained limitations.
- Publication tests: 14 passed. SHA inventory regenerated with POSIX paths; actual digest verification performed before publication. Canonical .tex/.bib exactly match Prism staging and ZIP; all five canonical paper files exactly match publication staging. checks.json records accepted hashes.

## Scope and remaining limitation

No subagents, Computer Use, local GPU work, new experiments or solver-code edits. No stronger priority, exactness or scaling claims. Online Prism was not synchronized or compiled: there is no verified permitted connector for that project. The local import ZIP is ready, not evidence of remote synchronization. Git publication state is recorded separately in publication.txt.
