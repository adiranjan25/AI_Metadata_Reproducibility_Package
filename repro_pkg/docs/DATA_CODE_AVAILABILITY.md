# Data and Code Availability Statement

The public case-study evidence is available from the New York City Taxi and Limousine Commission (NYC TLC) Trip Record Data portal and its public data dictionaries. The study uses public schema metadata and the documented addition of `cbd_congestion_fee` from 2025 onward; it does not redistribute TLC trip-level Parquet files in this package.

The controlled semantic corpus is generated from the field names and descriptions documented in the manuscript methodology. The 10K–100K retrieval workload is explicitly synthetic-scale and is seeded from the TLC-derived metadata vocabulary. It must not be represented as a native NYC TLC catalog of that size.

All scripts needed to regenerate the controlled corpus, classification metrics, retrieval benchmark, raw machine-readable summaries, and manuscript charts are included in this package. Benchmark latency is environment-dependent. The package records software versions; a replication intended for archival publication should additionally record CPU model, allocated cores, RAM, and container/VM limits.

Before journal submission, archive this package in a permanent research repository (for example Zenodo), obtain a DOI, and replace the manuscript's provisional code/data availability language with the DOI and release version.
