#!/usr/bin/env python3
"""
IAQF 2026 - Project Layer 主入口
复合型架构：Config(Singleton) | Data(ETL) | Features(Pipeline) | Models(Strategy) | Diagnostics(Observer) | Usage(Facade)

运行: 在项目根目录执行 python -m project_layer.main
"""
import argparse
from pathlib import Path
import sys

# 运行方式：cd 项目根 && python -m project_layer.main
_ROOT = Path(__file__).resolve().parent
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

try:
    from project_layer.usage.runner import Runner
except ImportError:
    from usage.runner import Runner


def main():
    parser = argparse.ArgumentParser(description="IAQF 2026 Project - Ensembled Running Entrance")
    parser.add_argument("--etl", action="store_true", help="Run the entire ETL from Tick Data (otherwise start from processed data (1-minute))")
    parser.add_argument("--stages", nargs="+", default=None,
                        help="Stages: data features var diagnostics simulation cmlsi_robustness (otherwise run all)")
    args = parser.parse_args()
    runner = Runner(run_etl_stage=args.etl)
    runner.start(stages=args.stages)

if __name__ == "__main__":
    main()
