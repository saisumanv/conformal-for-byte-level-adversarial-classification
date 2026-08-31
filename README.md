# Conformal Prediction for Byte-Level CPS Traffic Classification under Adversarial Attacks
Artifact for COMNET-D-26-00609.

## Datasets
Not redistributed here. Obtain from the sources cited in Table 1:
4SICS, IEC60870-IEC IDS, IEC-61850-GOOSE, RICSel21-IEC104.

## Regenerating byte-level inputs
Edit the pcap path in `utils.generateRawBytes()` and run it once per capture.
Outputs land in `data/raw_bytes/` as `raw_bytes_<dataset>.csv`.

## Running
1. Set the target CSV in `train.py` (`input_raw = ...`).
2. `python train.py`
Hyperparameters, splits, and seeds are documented in Appendix A of the paper.

## Environment
CPU only; no GPU required. See requirements.txt.

## Attribution
`attack.py` and `cw.py` are adapted from the torchattacks library.
`clustered_class_conformal_utils.py` and `clustering_utils.py` are adapted from
the empirical-bayes-conformal repository.
