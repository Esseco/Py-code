"""Scalar magnetization readers: legacy tuple API and per-element aggregation."""
import re
from pathlib import Path as p
from monty.io import zopen
from monty.os.path import zpath
from pymatgen.core import Structure

def read_magnetic_moments_outcar(init_dir:p) -> tuple:
    '''
    read_magnetic_moments_outcar 的 Docstring
    
    :param init_dir: vasp_dir contains output
    :type init_dir: p
    :return: elements, magnetic_moments, total_magnetic_moment
    :rtype: tuple
    '''
    init_dir = p(init_dir)
    poscar_file = p(zpath(init_dir/'POSCAR'))
    outcar_file = zpath(init_dir/'OUTCAR')
    structure = Structure.from_file(poscar_file)
    elements = [str(site.specie) for site in structure]
    magnetic_moments = []
    total_magnetic_moment = 0.0 
    current_step = []
    reading_magnetization = False

    try:
        with zopen(outcar_file, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                if re.search(r'magnetization\s*\((x|y|z)\)', line, re.IGNORECASE):
                    reading_magnetization = True
                    current_step = []
                    total_magnetic_moment = 0.0 
                    continue
                    
                if reading_magnetization:
                    if '---' in line or '# of ion' in line:
                        continue
                        
                    match = re.match(r'\s*(\d+)\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\s+([-.\d]+)', line)
                    if match:
                        current_step.append(float(match.group(2))) 
                        continue
                        
                    match_tot = re.match(r'\s*tot\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)', line, re.IGNORECASE)
                    if match_tot:
                        total_magnetic_moment = float(match_tot.group(4)) 
                        
                        if current_step:
                            magnetic_moments = [current_step] 
                        
                        reading_magnetization = False
                        continue
                        
    except FileNotFoundError:
        print(f"OUTCAR not found: {outcar_file}")
        return 0

    if not magnetic_moments or len(magnetic_moments[0]) != len(elements):
        return 0
        
    return elements, magnetic_moments, total_magnetic_moment


def read_element_moments(directory, *, structure=None, parameters=None) -> dict:
    """Aggregate the final local magnetic moments per chemical element.

    Only the last magnetization table (final ionic step) is read, through
    :func:`Process_Vasp.results.magnetic_check.read_dft_magnetic_data`, so
    compressed OUTCAR and SOC handling stay in one place. Units are mu_B.

    Returns ``available``, ``elements`` (symbol -> ``count``/``sum``/``mean``
    plus ``min``/``max`` for collinear runs or ``norm_mean`` for non-collinear
    ones), ``moments`` in atom order, ``noncollinear`` and ``total_local_moment``.
    Missing or invalid data gives ``available=False`` with the reader's error
    instead of zeros. Read-only: no file writes and no VASP execution.
    """
    from .magnetic_check import read_dft_magnetic_data

    raw = read_dft_magnetic_data(directory, structure=structure, parameters=parameters)
    report = {
        "schema": "vasp-element-moments-v1",
        "unit": "mu_B",
        "available": raw.get("status") == "available",
        "noncollinear": bool(raw.get("noncollinear", False)),
        "source": raw.get("source", "OUTCAR_final_local_projection"),
        "elements": {},
    }
    if not report["available"]:
        report["status"] = raw.get("status", "unavailable")
        report["error"] = raw.get("read_error")
        return report

    sites = raw.get("sites") or []
    grouped = {}
    for site in sites:
        grouped.setdefault(site["element"], []).append(site["moment"])

    for element, values in grouped.items():
        block = {"count": len(values), "sum": _sum(values), "mean": _mean(values),
                 "moments": list(values)}
        if report["noncollinear"]:
            block["norm_mean"] = _mean([_norm(value) for value in values])
        else:
            block["min"], block["max"] = min(values), max(values)
        report["elements"][element] = block

    report.update(
        atom_elements=list(raw.get("elements") or []),
        n_atoms=len(sites),
        total_local_moment=raw.get("total_local_moment"),
    )
    return report


def _sum(values):
    if values and isinstance(values[0], (list, tuple)):
        return [sum(component) for component in zip(*values)]
    return sum(values)


def _mean(values):
    if not values:
        return None
    if isinstance(values[0], (list, tuple)):
        return [sum(component) / len(values) for component in zip(*values)]
    return sum(values) / len(values)


def _norm(value):
    if isinstance(value, (list, tuple)):
        return sum(component * component for component in value) ** 0.5
    return abs(value)


