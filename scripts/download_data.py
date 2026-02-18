"""Dataset acquisition script for BoltzKinema training data.

Downloads all raw datasets needed for training:
  - ATLAS: GROMACS trajectories (~30 GB)
  - MISATO: HDF5 protein-ligand complexes (~50 GB)
  - MDposit/DynaRepo: PDB + XTC trajectories (~200 GB)
  - DD-13M: Metadynamics dissociation trajectories (~100 GB)
  - CATH2: CATH domain trajectories (~28 GB compressed)
  - MegaSim: Wildtype + mutant trajectories (~10 GB compressed)
  - Octapeptides: Peptide trajectories (~511 MB compressed)

Usage:
    python scripts/download_data.py --datasets all --output-dir data/raw
    python scripts/download_data.py --datasets atlas,cath2 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress bar, skipping if it already exists."""
    if dest.exists():
        logger.info("Skipping existing file: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            with tqdm(
                total=total, unit="B", unit_scale=True, desc=desc or dest.name
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_zenodo_files(
    record_id: str,
    filenames: list[str] | None,
    output_dir: Path,
) -> None:
    """Download files from a Zenodo record.

    Parameters
    ----------
    record_id : Zenodo record ID.
    filenames : specific filenames to download, or None for all files.
    output_dir : directory to save files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(
        f"https://zenodo.org/api/records/{record_id}", timeout=30
    )
    r.raise_for_status()
    files = {f["key"]: f for f in r.json()["files"]}

    targets = filenames if filenames else list(files.keys())
    for fname in targets:
        if fname not in files:
            logger.warning("File %s not found in Zenodo record %s", fname, record_id)
            continue
        file_info = files[fname]
        dest = output_dir / fname
        _download_file(
            file_info["links"]["self"],
            dest,
            desc=fname,
        )


# ---------------------------------------------------------------------------
# Per-dataset downloaders
# ---------------------------------------------------------------------------


def download_atlas(output_dir: Path) -> None:
    """Download ATLAS dataset (~30 GB).

    Source: https://www.dsimb.inserm.fr/ATLAS/
    ~1,500 protein chains, 3x100ns trajectories each.
    Format: GROMACS .xtc trajectories + .gro topology.
    """
    atlas_dir = output_dir / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ATLAS dataset to %s", atlas_dir)
    logger.info("Using wget for recursive ATLAS download...")
    subprocess.run(
        [
            "wget",
            "-r",
            "-np",
            "-nH",
            "--cut-dirs=1",
            "-P",
            str(atlas_dir),
            "https://www.dsimb.inserm.fr/ATLAS/database/",
        ],
        check=True,
    )


def download_misato(output_dir: Path) -> None:
    """Download MISATO dataset (~50 GB).

    Source: https://github.com/t7morgen/misato-dataset
    ~16,000 protein-ligand complexes, 8ns each (100 frames).
    Format: HDF5 with coordinates + topology.
    """
    misato_dir = output_dir / "misato"
    misato_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MISATO dataset to %s", misato_dir)
    logger.info(
        "MISATO requires its own download tool. "
        "Run: python -m misato.download --output-dir %s",
        misato_dir,
    )


def download_dynarepo(output_dir: Path, max_workers: int = 4) -> None:
    """Download MDposit/DynaRepo dataset (~200 GB).

    Source: DynaRepo (federated MDDB).
    ~930 systems, ~700 unique proteins, 3 replicas x 500ns each.
    Format: PDB + XTC + TPR.
    """
    NODES = [
        ("dynarepo", "https://dynarepo.inria.fr"),
    ]
    FILES = ["structure.pdb", "trajectory.xtc", "topology.tpr"]
    dynarepo_dir = output_dir / "dynarepo"

    def _get_all_projects(base_url: str) -> list[dict]:
        projects: list[dict] = []
        page = 1
        while True:
            r = requests.get(
                f"{base_url}/api/rest/v1/projects",
                params={"page": page},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            batch = data.get("projects", [])
            if not batch:
                break
            projects.extend(batch)
            if len(projects) >= data.get("filteredCount", 0):
                break
            page += 1
        return projects

    def _download_one(
        base_url: str, accession: str, filename: str, md_index: int, dest: Path
    ) -> None:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(
            f"{base_url}/api/rest/v1/projects/{accession}/files/{filename}",
            params={"md": md_index},
            stream=True,
            timeout=60,
        )
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    for node_name, base_url in NODES:
        logger.info("Fetching project list from %s...", node_name)
        projects = _get_all_projects(base_url)
        logger.info("Found %d projects on %s", len(projects), node_name)

        tasks = []
        for proj in projects:
            accession = proj["accession"]
            n_replicas = len(proj.get("mds", ["replica 1"]))
            for md_i in range(n_replicas):
                for fname in FILES:
                    dest = dynarepo_dir / accession / f"replica_{md_i}" / fname
                    tasks.append((base_url, accession, fname, md_i, dest))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_one, *t): t for t in tasks}
            for f in tqdm(
                as_completed(futures), total=len(futures), desc=node_name
            ):
                f.result()


def download_dd13m(output_dir: Path) -> None:
    """Download DD-13M dataset (~100 GB).

    Source: arXiv:2504.18367
    565 complexes, 26,612 metadynamics dissociation trajectories.
    """
    dd13m_dir = output_dir / "dd13m"
    dd13m_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "DD-13M download not yet automated. "
        "Please download manually from the paper's data repository to %s",
        dd13m_dir,
    )


def download_cath2(output_dir: Path) -> None:
    """Download CATH2 domains dataset (~28 GB compressed).

    Source: https://zenodo.org/records/15629740
    ~1,100 CATH domains, 50-200 amino acids, ~1 us each.
    """
    cath2_dir = output_dir / "cath2"
    logger.info("Downloading CATH2 dataset to %s", cath2_dir)
    _download_zenodo_files(
        record_id="15629740",
        filenames=["MSR_cath2.zip"],
        output_dir=cath2_dir,
    )
    # Unzip
    zip_path = cath2_dir / "MSR_cath2.zip"
    if zip_path.exists():
        logger.info("Extracting %s...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cath2_dir)


def download_megasim(output_dir: Path) -> None:
    """Download MegaSim dataset (~10 GB compressed).

    Source: https://zenodo.org/records/15641184
    271 wildtype proteins + 21,458 point mutants.
    """
    megasim_dir = output_dir / "megasim"
    logger.info("Downloading MegaSim dataset to %s", megasim_dir)
    _download_zenodo_files(
        record_id="15641184",
        filenames=[
            "megasim_wildtype_merge.zip",
            "megasim_mutants_allatom.zip",
        ],
        output_dir=megasim_dir,
    )
    # Unzip
    for zip_name in ["megasim_wildtype_merge.zip", "megasim_mutants_allatom.zip"]:
        zip_path = megasim_dir / zip_name
        if zip_path.exists():
            logger.info("Extracting %s...", zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(megasim_dir)


def download_octapeptides(output_dir: Path) -> None:
    """Download Octapeptides dataset (~511 MB compressed, ~78 GB uncompressed).

    Source: https://zenodo.org/records/15641199
    ~1,100 8-residue peptides, 5 x 1 us each.
    """
    octa_dir = output_dir / "octapeptides"
    logger.info("Downloading Octapeptides dataset to %s", octa_dir)
    _download_zenodo_files(
        record_id="15641199",
        filenames=None,  # Download all files
        output_dir=octa_dir,
    )
    # Unzip any zip files
    for zip_path in sorted(octa_dir.glob("*.zip")):
        logger.info("Extracting %s...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(octa_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_DATASETS = {
    "atlas": download_atlas,
    "misato": download_misato,
    "dynarepo": download_dynarepo,
    "dd13m": download_dd13m,
    "cath2": download_cath2,
    "megasim": download_megasim,
    "octapeptides": download_octapeptides,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BoltzKinema training data")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names, or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Root output directory for raw data",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets == "all":
        datasets = list(ALL_DATASETS.keys())
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    for name in datasets:
        if name not in ALL_DATASETS:
            logger.error("Unknown dataset: %s (available: %s)", name, list(ALL_DATASETS.keys()))
            continue
        logger.info("=== Downloading %s ===", name)
        ALL_DATASETS[name](output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
