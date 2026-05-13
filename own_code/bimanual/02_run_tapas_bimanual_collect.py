#!/home/nils/tapas310/bin/python

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TAPAS = ROOT / "TAPAS"
RLBENCH = ROOT / "rlbench"
COPPELIA = ROOT / "sim" / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"


def main():
    env = os.environ.copy()
    env["COPPELIASIM_ROOT"] = str(COPPELIA)
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)
    env["PYTHONPATH"] = f"{TAPAS}:{RLBENCH}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{COPPELIA}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        sys.executable,
        "tapas_gmm/collect_data_rlbench.py",
        "-c",
        "conf/collect_data/rlbench_bimanual.py",
        "-t",
        "BimanualDualPushButtons",
    ]

    subprocess.run(cmd, cwd=TAPAS, env=env, check=True)


if __name__ == "__main__":
    main()
