# Source and QA notes

- Technical report required structure: title, technical summary, findings, scope/definitions, methodology, limitations/robustness, recommendations, further questions.
- Chart map: stability grouped bar; raw-vs-difference edge scatter; reconstruction grouped bar; candidate graph and suspect-edge audit tables.
- Pearson analyses use the full chronological build data. Spearman uses a deterministic evenly spaced sample.
- Test split is not read by this script.
- Portable report validation and packaging passed. Browser verification is structural-only because no compatible Chromium headless-shell was available; semantic chart tables remain embedded.
