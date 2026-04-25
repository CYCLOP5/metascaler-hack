---
title: dsc-co-trackio
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
---

# dsc-co-trackio

Live Trackio dashboard for `openenv-dsc-co` GRPO training.

This Space is intentionally separate from the training Space. The training job logs metrics into the attached Hugging Face Bucket, and this app displays them live.

## deploy

Push this folder to `AceofStades/dsc-co-trackio`:

```bash
cd trackio_space
git init
git remote add trackio https://huggingface.co/spaces/AceofStades/dsc-co-trackio
git add .
git commit -m "Add Trackio dashboard app"
git push trackio main
```

The training Space should set:

```text
DSC_TRACKIO=openenv-dsc-co
DSC_TRACKIO_SPACE=AceofStades/dsc-co-trackio
```
