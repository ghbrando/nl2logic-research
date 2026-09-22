# Cluster access for this repository

- Use `ssh shared-dev` to access the DGX controller as `save-water`.
- Never authenticate as, run commands as, or use files under the `kip` account. This includes the two workers, even if a document lists `kip` credentials or commands.
- Access workers only through a separately authorized `save-water` account and SSH route. If that route is unavailable, stop worker operations and report the access blocker.
- Keep research checkouts, container state, caches, and outputs in project-specific locations owned by `save-water`. Do not alter other users' repositories, environments, containers, or inference processes.
