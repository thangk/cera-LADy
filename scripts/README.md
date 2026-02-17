# cera-LADy Scripts

## run_job.sh

Run LADy aspect detection benchmarks on CERA job output directories.

### Quick Start

```bash
# Enter the container
docker exec -it cera-lady-cli bash

# Run a job (from anywhere inside the container)
./app/scripts/run_job.sh type=cera dir=<job-dir-name> ac=laptops
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `type=` | Yes | | `real`, `cera`, or `heuristic` |
| `dir=` | Yes | | Job dir name (looked up in `/app/jobs/`) or absolute path |
| `ac=` | Yes | | Aspect category file: bare name (`laptops`, `restaurants`, `hotels`) or path to `.csv` |
| `models=` | No | `all` | Comma-separated: `rnd`, `btm`, `ctm`, `bert`, or `all` |
| `targets=` | No | auto-discover | Comma-separated target sizes (e.g., `100,500,1000`) |

### naspects

The number of aspects is **auto-counted** from the category CSV file (rows minus header):

| Category File | Aspects |
|---------------|---------|
| `laptops.csv` | 85 |
| `restaurants.csv` | 13 |
| `hotels.csv` | 36 |

### Examples

```bash
# Full benchmark: all 4 models, all targets, laptop categories
run_job.sh type=cera dir=j9715-rq1-cera-laptops-5-targets ac=laptops

# Single model on specific targets
run_job.sh type=heuristic dir=j972a-rq1-heuristic ac=laptops models=bert targets=100,500

# Real baseline
run_job.sh type=real dir=rq1-real-laptops-5-targets ac=laptops

# Custom categories file
run_job.sh type=cera dir=my-job ac=/app/datasets/categories/custom.csv
```

### XML Discovery

**cera/heuristic**: Searches `datasets/{target}/run{N}/` recursively for XML files. Prefers `*explicit.xml` over `*implicit.xml`.

**real**: Searches `datasets/{target}/` for XML files. Each XML found = one run.

### Output

Results go to `/app/output/{type}/` with collision avoidance (`cera/`, `cera-1/`, etc.).

```
output/{type}/
  target-100/
    run1/           # LADy results (contains {naspects}./{model}/ structure)
    run2/
    aggregate.csv   # Cross-run mean +/- std
  target-500/
    ...
```

### Setup

The job directories must be accessible at `/app/jobs/` inside the container. The `docker-compose.yml` mounts `./jobs:/app/jobs`.

Place or symlink CERA job output folders into `cera-LADy/jobs/`:

```bash
# Copy
cp -r ../../cera/jobs/j9715-rq1-cera-laptops-5-targets jobs/

# Or symlink (only works if target is also mounted in Docker)
ln -s /absolute/path/to/job jobs/job-name
```
