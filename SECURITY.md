# Security

Please report vulnerabilities privately through GitHub's "Report a vulnerability" button on this
repository, or by email to admin@rushlabs.ai. We aim to reply within a few days.

Rush Small is a training and evaluation recipe that runs on Modal. Every model and dataset is
pinned to a commit in `pipeline.py`, the Python environment is hash-locked in `requirements.txt`,
llama.cpp is pinned by commit and rerankd by release checksum.
