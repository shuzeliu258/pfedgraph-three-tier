# pFedGraph Three-Tier Extension

This repository provides a PyTorch-based implementation of a hierarchical
personalized federated learning system, extending **pFedGraph** from its original
single-tier (client-only) formulation to a **three-tier architecture** with
client, edge, and global servers.

The project focuses on personalized aggregation under realistic distributed
system settings, including hierarchical communication and heterogeneous clients.

---

## Key Features

- Extension of pFedGraph from a single-tier client setting to a three-tier (client–edge–global) architecture
- Hierarchical personalized aggregation based on client similarity
- Support for non-IID data distributions
- Experimental evaluation under heterogeneous client settings
- Separation of code and large experimental artifacts for clean repository management

---

## Experimental Data (GitHub Release)

The experimental artifacts used in the experiments (client partition pickle files)
are generated data and are not tracked in the Git repository due to GitHub’s
100MB file size limit.

You can download them from the GitHub Release page:

https://github.com/shuzeliu258/pfedgraph-three-tier/releases/tag/data-v1

Files Included

30_trainer_datast_list_CIFAR_hete_V6.pickle

30_trainer_datast_list_CIFAR_hete_V7.pickle

30_trainer_datast_list_CIFAR_hete_V8.pickle

30_trainer_datast_list_CIFAR_hete_V9.pickle

After downloading, unzip the files and place them in the project root directory
(or under the path specified in your configuration files).

---

## How to Run

Use the provided shell script to run the three-tier pFedGraph experiments:

bash run_pfedgraph-three-tier_3_8_rounds.sh

You may need to modify paths or configuration parameters depending on your
local environment and dataset locations.

---

## Notes on Reproducibility

The released pickle files contain generated client partition artifacts used in experiments

Different random seeds or configurations may result in different client partitions

The codebase is structured to allow easy extension to other datasets or hierarchy depths

---

## Background

pFedGraph is a personalized federated learning approach that leverages client
similarity to guide collaboration. The original algorithm assumes a flat,
single-tier client-server setting.

This project extends pFedGraph to a hierarchical federated learning system,
enabling personalized aggregation across client, edge, and global levels to
better reflect realistic deployment scenarios.

---

## Disclaimer

This code is intended for research and system prototyping purposes.
It is not optimized for production deployment.

---

## License

MIT License
