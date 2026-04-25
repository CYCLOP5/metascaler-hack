import os

import trackio


PROJECT = os.environ.get("TRACKIO_PROJECT", "openenv-dsc-co")

# Trackio auto-generates this bucket name from the Space id. Keeping the default
# here lets the dashboard read the same bucket the training Space writes to.
os.environ.setdefault("TRACKIO_BUCKET_ID", "AceofStades/dsc-co-trackio-bucket")

trackio.show(
    project=PROJECT,
    host="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    open_browser=False,
    block_thread=True,
)
