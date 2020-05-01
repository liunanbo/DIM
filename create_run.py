"""
Script to upload a run to the database

Usage:
python create_run.py path/to/config.json
"""

import sys
import json
from viola.db.models import Run
from viola.config import config
from mongoengine import connect


if __name__ == "__main__":
    conf_path = sys.argv[1]
    with open(conf_path, "r") as f:
        conf = json.load(f)
    connect(**config["mongodb"])

    if Run.objects.count() == 0:
        run_nb = 1
    else:
        run_nb = max(Run.objects.scalar("pk")) + 1

    Run(id=run_nb, config=conf).save()

    print("Created run with id {}".format(run_nb))
