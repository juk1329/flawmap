"""The inspection service: a seller uploads a capture, the platform server processes it, and the
seller and buyers open the result.

* ``store``  - SQLite table of inspections and the folder layout under ``data/service/``
* ``intake`` - saving uploads, safe zip extraction, finding and checking the Stray Scanner folder
* ``worker`` - the queue: one job at a time in its own process (one GPU)
* ``job``    - ``python -m flawmap.service.job <data_dir> <id>``: runs the pipeline for one inspection
"""
