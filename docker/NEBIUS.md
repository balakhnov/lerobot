# Build and run a LeRobot image on Nebius

This guide builds the current LeRobot branch into a GPU-enabled Docker image,
pushes it to Docker Hub, and runs it on a Nebius Container VM. The VM only
pulls the prebuilt image; it does not install LeRobot or its dependencies.

The same image works on both 1-GPU and 8-GPU VMs. GPU visibility is determined
at container runtime rather than being fixed in the image.

## Prerequisites

- Docker with Buildx support.
- A public Docker Hub repository, for example `your-user/lerobot`.
- An SSH key registered in Nebius.
- The branch and all files that should be included in the image checked out
  locally.

Nebius documents the Custom Image Container VM flow for images from public
registries. See the [Nebius custom container VM guide](https://docs.nebius.com/compute/virtual-machines/applications-containers).

## 1. Prepare the source tree

Run these commands from the repository root:

```bash
git branch --show-current
git status --short
git lfs pull
```

`docker build` uses the current working tree, including uncommitted files. For
a reproducible image, commit the intended changes before building and use the
commit hash in the image tag.

The Nebius image is defined by `docker/Dockerfile.nebius`. It:

- installs the CUDA-enabled LeRobot environment;
- does not set `CUDA_VISIBLE_DEVICES`, so the image can use the GPUs exposed by
  Docker;
- creates `/lerobot/outputs` for training artifacts;
- runs `sleep infinity` by default, keeping the container alive for
  `docker exec` sessions.

Do not add Hugging Face, Docker Hub, Weights & Biases, or other access tokens to
the Dockerfile or build context. Authenticate at runtime instead.

## 2. Configure the Buildx builder

List the available builders:

```bash
docker buildx ls
```

If `lerobot-builder` does not exist, create and select it:

```bash
docker buildx create --use --name lerobot-builder
docker buildx inspect --bootstrap
```

If it already exists, select it:

```bash
docker buildx use lerobot-builder
docker buildx inspect --bootstrap
```

The builder may pull `moby/buildkit:buildx-stable-1`. This is expected: Buildx
runs BuildKit in a local container to build for another platform, manage the
layer cache, and push the result. The BuildKit image is not included in the
LeRobot image and is not deployed to Nebius.

## 3. Build and push the image

Authenticate with Docker Hub:

```bash
docker login
```

Set the Docker Hub account and derive an immutable tag from the branch and
commit:

```bash
DOCKERHUB_USER="your-dockerhub-user"
BRANCH_TAG=$(git branch --show-current | tr '/' '-')
REVISION=$(git rev-parse --short=12 HEAD)
IMAGE="docker.io/${DOCKERHUB_USER}/lerobot:${BRANCH_TAG}-${REVISION}"
```

Build and push:

```bash
docker buildx build \
  --platform linux/amd64 \
  --file docker/Dockerfile.nebius \
  --label "org.opencontainers.image.revision=${REVISION}" \
  --tag "${IMAGE}" \
  --push \
  .
```

Use `linux/amd64` even when building on an Apple Silicon Mac. Nebius GPU VMs
run the AMD64 image, and Buildx handles the cross-platform build.

Keep the full value printed by the following command; it is the image name to
enter in Nebius:

```bash
echo "${IMAGE}"
```

## 4. Create the Nebius Container VM

In the Nebius console:

1. Open **Compute > Container VMs**.
2. Select **Create container VM**.
3. Select **Custom Image**.
4. Enter the complete image name, including its commit tag.
5. Select the required 1-GPU or 8-GPU platform and preset.
6. Add the Docker run arguments shown below.
7. Configure disk size, networking, username, and SSH public key.
8. Create the VM.

Use these Docker run arguments for either GPU count:

```text
--restart=always --gpus all --shm-size=16GB -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

With this configuration:

- a 1-GPU VM exposes one GPU to the container;
- an 8-GPU VM exposes all eight GPUs to the same image.

Do not set `CUDA_VISIBLE_DEVICES=0` in the image. That would hide the other
seven GPUs on an 8-GPU VM. To deliberately restrict a particular container,
set the variable at runtime, for example:

```text
-e CUDA_VISIBLE_DEVICES=0,1
```

Making eight GPUs visible does not by itself start distributed training. The
training command must still launch the appropriate number of worker processes.

## 5. Persist caches and training outputs

Container files can be lost when the container or VM is replaced. For a simple
setup, add Docker named volumes to the run arguments:

```text
--restart=always --gpus all --shm-size=16GB -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -v lerobot-hf-cache:/home/user_lerobot/.cache/huggingface -v lerobot-outputs:/lerobot/outputs
```

Named volumes survive container restarts but remain on that VM's disk. For
important or long training runs, use a retained Nebius disk or shared
filesystem for datasets, caches, and checkpoints. When replacing a 1-GPU VM
with an 8-GPU VM, retain and reattach the data disk.

## 6. Connect and verify the container

Copy the VM public IP from the Nebius console and connect to the host:

```bash
ssh your-nebius-user@your-nebius-vm-ip
```

Find the running container:

```bash
sudo docker ps
```

Enter it using the container name from the previous command:

```bash
sudo docker exec -it your-container-name bash
```

Verify the GPU environment inside the container:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())'
which lerobot-train
```

Expected `torch.cuda.device_count()` output is `1` on a 1-GPU VM and `8` on an
8-GPU VM.

Authenticate only after entering the running container:

```bash
hf auth login
wandb login
```

Then run the normal training command, writing durable artifacts under the
mounted output directory:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --output_dir=/lerobot/outputs/act-training
```

Replace the policy, dataset, and remaining options with the desired training
configuration.

## 7. Publish a new branch version

After changing the branch:

1. Commit the changes.
2. Rerun the variables and `docker buildx build` commands.
3. Use the new branch-and-commit image tag when creating or redeploying the
   Container VM.
4. Reattach the persistent disk or volumes containing datasets and outputs.

Buildx reuses cached dependency layers when `pyproject.toml` and `uv.lock` have
not changed, so code-only image updates are normally faster than the first
build.

## Troubleshooting

### The container immediately stops

Confirm that the image was built from `docker/Dockerfile.nebius`. Its default
command is `sleep infinity`, which keeps the container running.

### Only one GPU is visible on an 8-GPU VM

Check that `CUDA_VISIBLE_DEVICES=0` is not baked into the image or supplied in
the Docker run arguments:

```bash
echo "${CUDA_VISIBLE_DEVICES}"
python -c 'import torch; print(torch.cuda.device_count())'
```

### LIBERO fails to initialize EGL

LIBERO's headless MuJoCo renderer requires the NVIDIA OpenGL/EGL driver
libraries in addition to CUDA. If evaluation reports `failed to create dri2
screen` or `Cannot initialize a EGL device display`, recreate the container
with:

```text
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

Setting this variable later inside `docker exec` is too late: the NVIDIA
container runtime decides which host driver libraries to mount when it creates
the container. Images built from `docker/Dockerfile.nebius` include this value
by default, but keeping it explicit in the Container VM run arguments also
makes the requirement visible in the deployment configuration.

### The VM reports an image architecture error

Rebuild with:

```text
--platform linux/amd64
```

### Docker pulls `moby/buildkit`

This is the expected Buildx builder image. It participates only in the local
build and push process.
