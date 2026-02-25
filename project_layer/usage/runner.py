"""
Runner - Facade 门面
用户只需 Runner.start() 即可运行完整流程
"""
from pathlib import Path

from ..config import get_config
from ..data.etl import run_etl
from ..eda import run_eda
from ..features.pipeline import FeaturePipeline
from ..models.figures_runner import run_var_with_figures
from ..diagnostics.observer import DiagnosticsObserver
from ..diagnostics.ar1 import run_ar1_mean_reversion
from ..diagnostics.transaction_cost import run_transaction_cost_analysis
from ..diagnostics.var_hypotheses import run_var_hypothesis_tests
from ..simulation.runner import run_all_scenarios


class Runner:
    """门面：统一入口"""

    def __init__(self, run_etl_stage: bool = False):
        """
        run_etl_stage: 若 True 则从 Tick 重新跑 ETL；否则使用主项目 processed_1min
        """
        self.run_etl_stage = run_etl_stage
        self.cfg = get_config()

    def start(self, stages: list[str] | None = None) -> None:
        """
        执行完整管道
        stages: 若不指定则全流程；可指定 ["data","features","var","diagnostics","simulation"]
        """
        cfg = self.cfg
        cfg.ensure_dirs()
        if stages is None:
            stages = ["data", "features", "var", "diagnostics", "simulation"]
        else:
            stages = [s.lower() for s in stages]

        # 1. Data (ETL)
        processed_dir = None
        if "data" in stages and self.run_etl_stage:
            processed_dir = run_etl()
        elif "data" in stages and not self.run_etl_stage:
            processed_dir = cfg.get_processed_1min_dir()
            if not processed_dir.exists():
                print("[WARN] processed_1min does not exist, please run '01_data_processor' or set run_etl_stage=True")

        # 2. Features (Pipeline)
        if "features" in stages:
            print("\n--- Feature Pipeline ---")
            pipeline = FeaturePipeline()
            pipeline.run_all(processed_dir)

        # 2b. EDA (图01-05, 09, 10)
        if "eda" in stages or "features" in stages:
            run_eda()

        # 3. VAR + 图表
        if "var" in stages:
            print("\n--- VAR Modeling & Figures ---")
            if not cfg.get_master_feature_path().exists():
                print("[ERROR] Master does not exist, pleas go through the 'features' stage first.")
                return
            run_var_with_figures()

        # 4. Diagnostics (Observer + AR1 + 交易成本)
        if "diagnostics" in stages:
            print("\n--- Diagnostics ---")
            if not cfg.get_master_feature_path().exists():
                print("[WARN] Master does not exists, skip 'diagnostics'")
            else:
                obs = DiagnosticsObserver(None, "crisis")
                obs.run_all_regimes_report()
                run_ar1_mean_reversion()
                run_transaction_cost_analysis()
                run_var_hypothesis_tests()

        # 5. Simulation (GENIUS Act)
        if "simulation" in stages:
            print("\n--- GENIUS Act Simulation ---")
            if not cfg.get_master_feature_path().exists():
                print("[WARN] Master does not exist, skip 'simulation'")
            else:
                run_all_scenarios()

        # 6. CMLSI Advanced Robustness
        if "cmlsi_robustness" in stages:
            print("\n--- CMLSI Advanced Robustness ---")
            if not cfg.get_master_feature_path().exists():
                print("[WARN] Master does not exist, skip 'CMLSI robustness analysis'")
            else:
                from ..CMLSI_test_advanced import run_full_cmlsi_robustness
                run_full_cmlsi_robustness()

        print("\n" + "=" * 60)
        print("√ Project Layer pipeline complete.")
        print("=" * 60)
