# Isolated research on DGX Spark

Status: rootless Docker and a small GPU allocation validated on `spark-87dc`.
Both workers have separate `save-water` accounts and rootless daemons. Only
`spark-87dc` has the research image built. The controller and both workers host
inference processes; arrange capacity before a training run.

## Isolation boundaries

- Run as the invoking user's numeric UID/GID, with all capabilities dropped,
  no new privileges, and a read-only container filesystem and repository mount.
- Persist writes only in project-specific `outputs` and `cache` directories.
  Temporary writes go to a bounded private `/tmp` and private shared memory.
- Default to no network and no GPU. Separate overrides enable model downloads
  or GPU access. No host networking, host IPC, privileged mode, SSH credentials,
  home-directory mount, or Docker socket mount is used.
- Set CPU, RAM, swap, process and log bounds. RAM limits are not a reliable GPU
  memory quota. Containers share the host kernel and GPU; they are not VMs or
  protection against all GPU contention, driver failures or hostile code.
- The existing Docker daemon stores image layers in its own storage, commonly
  `/var/lib/docker`. Runtime research outputs stay under the chosen state path.
  Strictly keeping image storage out of system directories requires a separately
  configured rootless daemon; do not move the shared daemon's data directory.

## Host prerequisites

Docker Engine with Compose and NVIDIA Container Toolkit on an ARM64 Spark.
Use `ssh shared-dev` as `save-water` for controller access and a separate
`save-water` login for research work on the workers. The `kip` account is only
for administrator provisioning of that account, as authorized by the user.
The controller's `save-water` account currently cannot access the Docker socket.
Prefer administrator-managed launches or an administrator-reviewed rootless GPU
setup. Docker group membership provides root-level host privileges; running the
container as a non-root UID does not remove that privilege from Docker clients.
Do not change shared daemon/runtime settings as part of a project launch.
With rootless Docker, add `-f containers/rootless.yaml` to every Compose command.
For GPU runs, also add `-f containers/gpu-rootless.yaml` in place of
`containers/gpu.yaml`; the former uses NVIDIA CDI. Generate the CDI spec in
`~/.config/cdi/nvidia.yaml` using `nvidia-ctk cdi generate`, and validate a
small container with `--device nvidia.com/gpu=0` before a research image build.
Container UID 0 maps to `save-water` on the host, so it can write to the project
state directories without granting host root access. Keep the other isolation
settings from the base file. Rootless GPU support and cgroup limits require
separate validation on the chosen worker.

The image starts from NVIDIA's Spark playbook PyTorch image and constrains pip
to its installed torch version. Container dependencies live in
`containers/requirements.txt`; the repo's packaging 26 pin conflicts with
NVIDIA DALI's packaging <=25 requirement. The base digest and added packages
are pinned. Record the image ID and `/opt/nl2logic-packages.txt` for each build.

On 2026-09-22, `spark-87dc` built image
`sha256:c97390c632923357e32935ab77fe70494322268881876c62928df72c6567a1f8`
(about 9.5 GB) from repository revision `cd48202`. `pip check` passed;
PyTorch `2.10.0a0+b558c986e8.nv25.11` with CUDA 13.0 returned 32.0 from a
small GB10 tensor. Eleven training-module tests passed with scratch files in
`/tmp`. The read-only source mount, writable output mount, 12 GiB memory limit,
and 256 process limit were verified. No model training has been run yet.

## Prepare on the chosen worker

Run as `save-water`, from its separate repo checkout, with Docker access:

```bash
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export NL2LOGIC_STATE="$HOME/nl2logic-state"
mkdir -p "$NL2LOGIC_STATE/outputs" "$NL2LOGIC_STATE/cache"
docker compose -f containers/compose.yaml -f containers/rootless.yaml config --quiet
docker compose -f containers/compose.yaml -f containers/rootless.yaml build
docker compose -f containers/compose.yaml -f containers/rootless.yaml run --rm research python -m pip check
```

Keep these exports in the shell used for subsequent commands. Do not use the
existing conda/tmux launcher inside this container; invoke Python directly.

Download the default model without GPU access, then use offline containers:

```bash
docker compose -f containers/compose.yaml -f containers/rootless.yaml -f containers/download.yaml run --rm research python -c "from huggingface_hub import snapshot_download; snapshot_download('google/flan-t5-small')"
```

After capacity is available, verify CUDA and a small allocation:

```bash
docker compose -f containers/compose.yaml -f containers/rootless.yaml -f containers/gpu-rootless.yaml run --rm research python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(32, device='cuda').sum().item())"
```

Run a small training job with an explicitly chosen input and unique output path:

```bash
docker compose -f containers/compose.yaml -f containers/rootless.yaml -f containers/gpu-rootless.yaml run --rm -e HF_HUB_OFFLINE=1 research python src/training/train.py --train-file /workspace/data/training_pairs/doctrine_real_train.jsonl --output-dir /outputs/smoke-001 --epochs 1 --batch-size 1 --limit 32
```

This checks infrastructure, not research validity. Freeze reviewed evaluation
splits and run the project's leakage/preflight checks before a research run.
For tests that create files beside the source, copy `/workspace` into `/tmp`
inside the container and run tests there; the repository mount is intentionally
read-only. Large data and checkpoints may require revised, measured limits.

Start with independent jobs on workers. Multi-node NCCL needs a separately
reviewed network/RDMA configuration; the offline GPU override is single-node.

References:
- https://build.nvidia.com/spark/pytorch-fine-tune
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- https://docs.docker.com/engine/security/rootless/
- https://docs.docker.com/engine/install/linux-postinstall/
