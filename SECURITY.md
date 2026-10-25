# Security

## Scope

This repository trains and runs models locally. It ships no server, no network
service and no credentials. The realistic risks are:

- a malicious corpus: the data path executes nothing, but a crafted JSONL can
  exhaust memory (see the size limits in `docs/BENCH.md`);
- a malicious checkpoint: `.pt` files are loaded with `weights_only=False`
  because they carry the config. Only load checkpoints you produced or trust.

## Reporting

Use GitHub's private vulnerability reporting (Security tab -> "Report a
vulnerability") instead of a public issue. Include the commit, the file and a
minimal reproduction. There is no bounty; expect an answer within a week.

## Not in scope

- Model outputs: they are not a security boundary. A completion can be wrong,
  offensive or insecure code; treat every generated snippet as untrusted.
- The Hugging Face datasets: their licences and gates are managed upstream.
