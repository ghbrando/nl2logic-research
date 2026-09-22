# Cluster access for this repository

- Use `ssh shared-dev` to access the DGX controller as `save-water`.
- Use the `kip` account on workers only for the administrator steps needed to provision and maintain the separate `save-water` account. The user explicitly authorized this exception. Run research work as `save-water` through `ssh shared-dev`.
- Do not put project files, research environments, or training jobs in `kip`'s home or Docker daemon.
- Access workers for project work through the `save-water` account and SSH route.
- Keep research checkouts, container state, caches, and outputs in project-specific locations owned by `save-water`. Do not alter other users' repositories, environments, containers, or inference processes.
