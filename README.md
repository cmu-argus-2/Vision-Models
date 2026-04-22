# NN-models

This repository contains the trained region classification and landmark detection networks.

## Installation

Installation can be done by running the command:

```bash
sh python.sh
```

## Allocating swap memory on the NVMe

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

To make swap persistent across reboots:

```bash
sudo nano /etc/fstab
```

Add this line:

```text
/swapfile none swap sw 0 0
```

Confirm:

```bash
free -h
```

## Jetson Stats

Jtop can be a useful tool to track memory/cpu/gpu usage when running the inference models. Link: https://github.com/rbonghi/jetson_stats

```bash
sudo apt update
sudo apt install python3-pip python3-setuptools -y
sudo pip3 install -U jetson-stats
```

## DVC workflow

This repository now uses **Git + DVC** on the `dvc` branch.

Git stores the code and the lightweight DVC metadata files. The large model and sample-image files are stored in DVC-managed storage and are fetched with `dvc pull`.

### Branch to use

Use the `dvc` branch for the DVC-based workflow.

If you already have a clone of the repository:

```bash
git fetch origin
git switch dvc
git pull
```

If `dvc` does not exist locally yet:

```bash
git fetch origin
git switch --track origin/dvc
```

## First-time setup on a new machine

Clone the repository, switch to the DVC branch, install DVC with SSH support, and then pull the large files:

```bash
git clone https://github.com/cmu-argus-2/Vision-Models.git Vision-Models-DVC
cd Vision-Models-DVC
git switch dvc
pipx install "dvc[ssh]"
dvc pull
```

If DVC is already installed but not on your shell `PATH`, run:

```bash
pipx ensurepath
exec $SHELL -l
```

## Normal update / pull workflow

From an existing clone:

```bash
cd ~/ARGUS_models/Vision-Models-DVC
git switch dvc
git pull
dvc pull
```

Use `git pull` to update the repo metadata and `dvc pull` to download the actual model and image files referenced by that metadata.

## Editing and pushing changes

Make changes inside the DVC-tracked directories:

- `trained-ld/`
- `trained-rc/`
- `sample_images/`

Then update the DVC metadata, upload the data objects, and push the Git commit:

```bash
cd ~/ARGUS_models/Vision-Models-DVC
git switch dvc

dvc add trained-ld
dvc add trained-rc
dvc add sample_images

dvc push

git add .dvc/config .dvc/.gitignore .gitignore trained-ld.dvc trained-rc.dvc sample_images.dvc README.md
git commit -m "Update models and sample images"
git push origin dvc
```

### Important notes

- Keep `trained-ld/`, `trained-rc/`, and `sample_images/` present locally.
- These directories should **not** be tracked directly by Git after migration to DVC.
- Git should track the `.dvc` files and DVC config files instead.
- Run `dvc push` before `git push` so the remote storage already contains the data referenced by the new commit.

## Local directory layout on `argus-workstation`

On `argus@argus-workstation`, the working directories currently live under:

```bash
~/ARGUS_models
```

Current contents:

```text
~/ARGUS_models/dvc-models-storage
~/ARGUS_models/Vision-Models
~/ARGUS_models/Vision-Models-backup
~/ARGUS_models/Vision-Models-DVC
~/ARGUS_models/Vision-Models-mint-backup
```

### Meaning of each directory

- `~/ARGUS_models/Vision-Models-DVC`  
  Main DVC-enabled working repository. This is the directory users should normally enter when working with the DVC branch.

- `~/ARGUS_models/dvc-models-storage`  
  DVC storage location on `argus-workstation`. This is where DVC stores the uploaded model and image objects. It is not a normal checked-out repo and will not look like `trained-ld/`, `trained-rc/`, or `sample_images/`.

- `~/ARGUS_models/Vision-Models`  
  Older checkout of the repository.

- `~/ARGUS_models/Vision-Models-backup` and `~/ARGUS_models/Vision-Models-mint-backup`  
  Backup copies used during migration.

## How the files are stored

There are two separate layers:

1. **Repository checkout**
   - Location: `~/ARGUS_models/Vision-Models-DVC`
   - Contains code, scripts, README, `.dvc` metadata, and the materialized working copies of `trained-ld/`, `trained-rc/`, and `sample_images/`

2. **DVC object storage**
   - Location: `~/ARGUS_models/dvc-models-storage`
   - Contains DVC-managed stored objects used by `dvc push` and `dvc pull`
   - This storage is backend data, not a user-facing project tree

## Quick reference

### Pull latest code and data

```bash
cd ~/ARGUS_models/Vision-Models-DVC
git switch dvc
git pull
dvc pull
```

### Push updated code and data

```bash
cd ~/ARGUS_models/Vision-Models-DVC
git switch dvc
dvc add trained-ld trained-rc sample_images
dvc push
git add .dvc/config .dvc/.gitignore .gitignore trained-ld.dvc trained-rc.dvc sample_images.dvc README.md
git commit -m "Update DVC-tracked artifacts"
git push origin dvc
```

### Test on another machine

```bash
git clone https://github.com/cmu-argus-2/Vision-Models.git Vision-Models-DVC-test
cd Vision-Models-DVC-test
git switch dvc
pipx install "dvc[ssh]"
dvc pull
```
