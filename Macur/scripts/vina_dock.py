import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def check_executable(executable_name: str) -> bool:
    return shutil.which(executable_name) is not None


def run_command(command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command execution failed: {' '.join(command)}\nOutput:\n{result.stdout}")
    return result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_ext(path: Path) -> str:
    return path.suffix.lower()


def convert_to_pdbqt_with_obabel(input_path: Path, output_path: Path, is_ligand: bool) -> None:
    if not check_executable("obabel"):
        raise RuntimeError("obabel not detected, please install Open Babel or prepare *.pdbqt files in advance.")

    ext = detect_ext(input_path)
    if not ext:
        raise RuntimeError(f"Cannot recognize file extension: {input_path}")

    args = ["obabel", f"-i{ext[1:]}", str(input_path), "-opdbqt", "-O", str(output_path)]
    if is_ligand:
        args += ["-p", "7.4"]
    result = run_command(args)
    if not output_path.exists():
        raise RuntimeError(f"obabel conversion failed, file not generated: {output_path}\nOutput:\n{result.stdout}")


def ensure_pdbqt(input_path: Path, work_dir: Path, is_ligand: bool) -> Path:
    if detect_ext(input_path) == ".pdbqt":
        return input_path
    ensure_dir(work_dir)
    out_name = input_path.stem + ".pdbqt"
    out_path = work_dir / out_name
    convert_to_pdbqt_with_obabel(input_path, out_path, is_ligand=is_ligand)
    return out_path


def parse_coords_from_pdb_or_pdbqt(lines: Iterable[str]) -> List[Tuple[float, float, float]]:
    coords: List[Tuple[float, float, float]] = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append((x, y, z))
            except Exception:
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                        coords.append((x, y, z))
                    except Exception:
                        pass
    if not coords:
        raise ValueError("Failed to parse atomic coordinates from PDB/PDBQT.")
    return coords


def parse_coords_from_mol2(lines: List[str]) -> List[Tuple[float, float, float]]:
    coords: List[Tuple[float, float, float]] = []
    in_atom = False
    for line in lines:
        if line.strip().upper().startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.strip().startswith("@<TRIPOS>") and not line.strip().upper().startswith("@<TRIPOS>ATOM"):
            if in_atom:
                break
        if in_atom:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    coords.append((x, y, z))
                except Exception:
                    pass
    if not coords:
        raise ValueError("Failed to parse atomic coordinates from MOL2.")
    return coords


def parse_coords_from_sdf(lines: List[str]) -> List[Tuple[float, float, float]]:
    if len(lines) < 4:
        raise ValueError("Incomplete SDF format.")
    counts = lines[3]
    try:
        num_atoms = int(counts[0:3].strip())
    except Exception:
        parts = counts.split()
        if parts:
            num_atoms = int(parts[0])
        else:
            raise ValueError("Failed to parse SDF atom count.")
    coords: List[Tuple[float, float, float]] = []
    start = 4
    end = start + num_atoms
    for line in lines[start:end]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                coords.append((x, y, z))
            except Exception:
                pass
    if not coords:
        raise ValueError("Failed to parse atomic coordinates from SDF.")
    return coords


def parse_coords(file_path: Path) -> List[Tuple[float, float, float]]:
    ext = detect_ext(file_path)
    text = file_path.read_text(errors="ignore").splitlines()
    if ext in (".pdb", ".pdbqt"):
        return parse_coords_from_pdb_or_pdbqt(text)
    if ext == ".mol2":
        return parse_coords_from_mol2(text)
    if ext in (".sdf", ".mol"):
        return parse_coords_from_sdf(text)
    raise ValueError(f"Unsupported coordinate file format: {ext}")


def compute_box(coords: List[Tuple[float, float, float]], margin: float = 4.0) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0)
    size = ((max_x - min_x) + margin, (max_y - min_y) + margin, (max_z - min_z) + margin)
    return center, size


def parse_best_affinity_from_pdbqt(pdbqt_path: Path) -> Optional[float]:
    try:
        with pdbqt_path.open("r", errors="ignore") as f:
            for line in f:
                if "REMARK VINA RESULT:" in line:
                    m = re.search(r"REMARK VINA RESULT:\s*([-\d\.]+)", line)
                    if m:
                        return float(m.group(1))
    except Exception:
        return None
    return None


_VINA_HELP_CACHE: Optional[str] = None


def get_vina_help_text() -> str:
    global _VINA_HELP_CACHE
    if _VINA_HELP_CACHE is not None:
        return _VINA_HELP_CACHE
    try:
        proc = subprocess.run(
            ["vina", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        _VINA_HELP_CACHE = proc.stdout or ""
    except Exception:
        _VINA_HELP_CACHE = ""
    return _VINA_HELP_CACHE


def vina_supports_flag(flag: str) -> bool:
    help_text = get_vina_help_text()
    return f"--{flag}" in help_text


def clean_receptor_pdbqt(pdbqt_path: Path) -> None:
    if not pdbqt_path.exists():
        return
    try:
        lines = pdbqt_path.read_text(errors="ignore").splitlines()
    except Exception:
        return
    remove_tags = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
    cleaned: List[str] = []
    for line in lines:
        strip = line.lstrip()
        if any(strip.startswith(tag) for tag in remove_tags):
            continue
        cleaned.append(line)
    if len(cleaned) != len(lines):
        try:
            pdbqt_path.write_text("\n".join(cleaned) + "\n")
        except Exception:
            pass


def dock_with_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    out_pdbqt: Path,
    log_path: Path,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: int,
    seed: Optional[int],
    cpu: Optional[int],
) -> None:
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.3f}",
        "--size_y", f"{size[1]:.3f}",
        "--size_z", f"{size[2]:.3f}",
        "--out", str(out_pdbqt),
    ]
    if vina_supports_flag("exhaustiveness"):
        cmd += ["--exhaustiveness", str(exhaustiveness)]
    if vina_supports_flag("num_modes"):
        cmd += ["--num_modes", str(num_modes)]
    if vina_supports_flag("energy_range"):
        cmd += ["--energy_range", str(energy_range)]
    if seed is not None and vina_supports_flag("seed"):
        cmd += ["--seed", str(seed)]
    if cpu is not None and vina_supports_flag("cpu"):
        cmd += ["--cpu", str(cpu)]

    result = run_command(cmd)
    try:
        with log_path.open("w") as f:
            f.write("===== COMMAND =====\n")
            f.write(" ".join(cmd) + "\n")
            f.write("\n===== STDOUT =====\n")
            f.write(result.stdout or "")
    except Exception:
        pass


def collect_ligand_files(path: Path) -> List[Path]:
    if path.is_dir():
        files: List[Path] = []
        for ext in (".pdbqt", ".sdf", ".mol2", ".pdb", ".mol"):
            files.extend(sorted(path.glob(f"*{ext}")))
        if not files:
            raise FileNotFoundError(f"No recognizable ligand files found in directory: {path}")
        return files
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Ligand path not found: {path}")


def is_multimodel_pdbqt(pdbqt_path: Path) -> bool:
    try:
        text = pdbqt_path.read_text(errors="ignore").splitlines()
    except Exception:
        return False
    model_count = sum(1 for line in text if line.strip().upper().startswith("MODEL"))
    return model_count > 1


def split_pdbqt_with_vina_split(input_pdbqt: Path, out_dir: Path, basename: str) -> List[Path]:
    ensure_dir(out_dir)
    if not check_executable("vina_split"):
        return []
    prefix = out_dir / f"{basename}_ligand"
    try:
        run_command(["vina_split", "--input", str(input_pdbqt), "--ligand", str(prefix)])
    except Exception:
        return []
    generated = sorted(out_dir.glob(f"{basename}_ligand_*.pdbqt"))
    return generated


def split_pdbqt_manually(input_pdbqt: Path, out_dir: Path, basename: str) -> List[Path]:
    ensure_dir(out_dir)
    try:
        lines = input_pdbqt.read_text(errors="ignore").splitlines()
    except Exception:
        return []
    chunks: List[List[str]] = []
    current: List[str] = []
    in_model = False
    for line in lines:
        up = line.strip().upper()
        if up.startswith("MODEL"):
            if current:
                chunks.append(current)
                current = []
            in_model = True
            continue
        if up.startswith("ENDMDL"):
            in_model = False
            chunks.append(current)
            current = []
            continue
        current.append(line)
    if current:
        chunks.append(current)
    out_files: List[Path] = []
    for i, chunk in enumerate(chunks, start=1):
        cleaned = [l for l in chunk if not l.strip().upper().startswith(("MODEL", "ENDMDL"))]
        out_path = out_dir / f"{basename}_ligand_{i}.pdbqt"
        try:
            out_path.write_text("\n".join(cleaned) + "\n")
            out_files.append(out_path)
        except Exception:
            pass
    return out_files


def expand_ligand_models(ligand_pdbqt: Path, work_dir: Path) -> List[Path]:
    if not is_multimodel_pdbqt(ligand_pdbqt):
        return [ligand_pdbqt]
    split_dir = work_dir / "split_ligands"
    basename = ligand_pdbqt.stem
    parts = split_pdbqt_with_vina_split(ligand_pdbqt, split_dir, basename)
    if not parts:
        parts = split_pdbqt_manually(ligand_pdbqt, split_dir, basename)
    return parts if parts else [ligand_pdbqt]


def main():
    parser = argparse.ArgumentParser(
        description="Perform molecular docking using AutoDock Vina (supports batch ligands)."
    )
    parser.add_argument("--receptor", required=True, type=str, help="Path to receptor file (.pdb or .pdbqt)")
    parser.add_argument("--ligand", required=True, type=str, help="Path to ligand file or directory (.sdf/.mol2/.pdb/.pdbqt or directory)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: outputs/vina_TIMESTAMP)")

    parser.add_argument("--center_x", type=float, default=None, help="Grid center X")
    parser.add_argument("--center_y", type=float, default=None, help="Grid center Y")
    parser.add_argument("--center_z", type=float, default=None, help="Grid center Z")
    parser.add_argument("--size_x", type=float, default=None, help="Grid size X (Å)")
    parser.add_argument("--size_y", type=float, default=None, help="Grid size Y (Å)")
    parser.add_argument("--size_z", type=float, default=None, help="Grid size Z (Å)")

    parser.add_argument("--box_ligand", type=str, default=None, help="Reference ligand file to define binding site box (center and size are auto-calculated from its coordinates)")
    parser.add_argument("--margin", type=float, default=4.0, help="Margin (Å) added to box calculated from reference ligand bounding box")

    parser.add_argument("--exhaustiveness", type=int, default=8, help="Sampling exhaustiveness (default 8)")
    parser.add_argument("--num_modes", type=int, default=9, help="Number of output conformations (default 9)")
    parser.add_argument("--energy_range", type=int, default=3, help="Energy range (default 3)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--cpu", type=int, default=None, help="CPU number to use (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")

    args = parser.parse_args()

    if not check_executable("vina"):
        print("Error: Cannot find vina command. Please install AutoDock Vina and ensure it is in your PATH.", file=sys.stderr)
        sys.exit(1)

    receptor_path = Path(args.receptor).expanduser().resolve()
    ligand_path = Path(args.ligand).expanduser().resolve()
    if not receptor_path.exists():
        print(f"Error: Cannot find receptor file: {receptor_path}", file=sys.stderr)
        sys.exit(1)
    if not ligand_path.exists():
        print(f"Error: Cannot find ligand path: {ligand_path}", file=sys.stderr)
        sys.exit(1)

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "outputs" / f"vina_{time_tag}")
    ensure_dir(out_dir)
    prep_dir = out_dir / "prepared"
    ensure_dir(prep_dir)
    logs_dir = out_dir / "logs"
    ensure_dir(logs_dir)
    poses_dir = out_dir / "poses"
    ensure_dir(poses_dir)

    try:
        receptor_pdbqt = ensure_pdbqt(receptor_path, work_dir=prep_dir, is_ligand=False)
    except Exception as e:
        print(f"Receptor preparation failed: {e}", file=sys.stderr)
        sys.exit(1)
    if receptor_pdbqt.suffix.lower() == ".pdbqt" and receptor_pdbqt.parent != prep_dir:
        copied = prep_dir / receptor_pdbqt.name
        try:
            shutil.copyfile(receptor_pdbqt, copied)
            receptor_pdbqt = copied
        except Exception:
            pass
    clean_receptor_pdbqt(receptor_pdbqt)

    center: Optional[Tuple[float, float, float]] = None
    size: Optional[Tuple[float, float, float]] = None
    cx, cy, cz = args.center_x, args.center_y, args.center_z
    sx, sy, sz = args.size_x, args.size_y, args.size_z
    if all(v is not None for v in (cx, cy, cz, sx, sy, sz)):
        center = (float(cx), float(cy), float(cz))
        size = (float(sx), float(sy), float(sz))
    elif args.box_ligand:
        box_ref = Path(args.box_ligand).expanduser().resolve()
        if not box_ref.exists():
            print(f"Error: Reference ligand file for auto box calculation not found: {box_ref}", file=sys.stderr)
            sys.exit(1)
        try:
            coords = parse_coords(box_ref)
            center, size = compute_box(coords, margin=float(args.margin))
            print(f"Auto-calculated box: center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})")
        except Exception as e:
            print(f"Failed to parse box from reference ligand: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: No complete center/size values provided and no --box_ligand for automatic box calculation.", file=sys.stderr)
        sys.exit(1)

    try:
        ligand_files = collect_ligand_files(ligand_path)
    except Exception as e:
        print(f"Failed to collect ligands: {e}", file=sys.stderr)
        sys.exit(1)

    results: List[Tuple[str, Optional[float], Path]] = []
    print(f"Start docking, total {len(ligand_files)} ligands ...")

    for idx, lig in enumerate(ligand_files, start=1):
        try:
            lig_pdbqt = ensure_pdbqt(lig, work_dir=prep_dir, is_ligand=True)
        except Exception as e:
            print(f"[{idx}/{len(ligand_files)}] Ligand preparation failed {lig.name}: {e}", file=sys.stderr)
            continue

        model_files = expand_ligand_models(lig_pdbqt, work_dir=prep_dir)
        if len(model_files) > 1:
            print(f"[{idx}/{len(ligand_files)}] Multi-model ligand found, split into {len(model_files)} conformers: {lig.name}")

        for m_i, mfile in enumerate(model_files, start=1):
            suffix = f"_m{m_i}" if len(model_files) > 1 else ""
            out_name = mfile.stem + "_vina_out.pdbqt" if len(model_files) == 1 else (lig_pdbqt.stem + f"{suffix}_vina_out.pdbqt")
            out_pose = poses_dir / out_name
            log_path = logs_dir / ((mfile.stem if len(model_files) == 1 else lig_pdbqt.stem + suffix) + ".log.txt")
            if out_pose.exists() and not args.overwrite:
                aff = parse_best_affinity_from_pdbqt(out_pose)
                print(f"[{idx}/{len(ligand_files)}] Result already exists (skipped): {lig.name}{suffix}  Best affinity: {aff if aff is not None else 'Unknown'} kcal/mol")
                results.append((lig.name + suffix, aff, out_pose))
                continue

            try:
                dock_with_vina(
                    receptor_pdbqt=receptor_pdbqt,
                    ligand_pdbqt=mfile,
                    out_pdbqt=out_pose,
                    log_path=log_path,
                    center=center,  # type: ignore
                    size=size,      # type: ignore
                    exhaustiveness=int(args.exhaustiveness),
                    num_modes=int(args.num_modes),
                    energy_range=int(args.energy_range),
                    seed=args.seed,
                    cpu=args.cpu,
                )
                aff = parse_best_affinity_from_pdbqt(out_pose)
                print(f"[{idx}/{len(ligand_files)}] Completed: {lig.name}{suffix}  Best affinity: {aff if aff is not None else 'Unknown'} kcal/mol")
                results.append((lig.name + suffix, aff, out_pose))
            except Exception as e:
                print(f"[{idx}/{len(ligand_files)}] Docking failed {lig.name}{suffix}: {e}", file=sys.stderr)
                continue

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ligand", "best_affinity_kcal_per_mol", "pose_pdbqt"])
        for lig_name, aff, pose_path in results:
            writer.writerow([lig_name, f"{aff:.3f}" if aff is not None else "", str(pose_path)])

    print(f"\nDone. Results directory: {out_dir}")
    print(f"- Receptor and ligand PDBQT files: {prep_dir}")
    print(f"- Docking logs: {logs_dir}")
    print(f"- Predicted conformations: {poses_dir}")
    print(f"- Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()


