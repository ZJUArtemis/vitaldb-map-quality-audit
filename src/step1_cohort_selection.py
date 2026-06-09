#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step1_cohort_selection.py | Topic: 10 | Purpose: 队列筛选与纳排标准应用
"""
import os, sys, json, logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import vitaldb
from tqdm import tqdm

# === PATH SAFETY ===
RAW_DATA = Path("/home/lxk/vitaldb/physionet.org")
WORK = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability")

def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)

# === LOGGING ===
def init_log():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = WORK / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lf = log_dir / f"{ts}_step1_cohort_selection.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()]
    )
    return lf

def main():
    log_file = init_log()
    logging.info(f"=== Topic 10 Phase 1 Step 1: 队列筛选 ===")
    logging.info(f"Log file: {log_file}")

    # === Step 1: 加载所有病例临床信息 ===
    logging.info("Loading all VitalDB cases...")
    # 从所有可能的 caseid 集合开始（使用最大的集合）
    # 先获取所有有 propofol 的病例（TIVA 案例）
    all_caseids = sorted(list(vitaldb.caseids_ppf))
    logging.info(f"Total cases with Propofol: {len(all_caseids)}")

    # 批量加载临床数据
    logging.info("Loading clinical data...")
    all_cases = vitaldb.load_clinical_data(caseids=all_caseids)
    logging.info(f"Clinical data loaded: {len(all_cases)} cases")

    # === Step 2: 应用纳入标准 ===
    flowchart = {}
    flowchart['total'] = len(all_cases)

    # 2.1 年龄 >= 18
    eligible = all_cases[all_cases['age'] >= 18].copy()
    flowchart['age_18_plus'] = len(eligible)
    logging.info(f"After age >= 18: {len(eligible)} cases")

    # 2.2 全身麻醉
    eligible = eligible[eligible['ane_type'].str.contains('General', na=False, case=False)]
    flowchart['general_anesthesia'] = len(eligible)
    logging.info(f"After general anesthesia: {len(eligible)} cases")

    # 2.3 排除心脏手术
    eligible = eligible[~eligible['department'].str.contains('Cardiac|Thoracic', na=False, case=False)]
    flowchart['exclude_cardiac'] = len(eligible)
    logging.info(f"After excluding cardiac/thoracic: {len(eligible)} cases")

    # 2.4 排除缺失关键人口统计学信息
    eligible = eligible.dropna(subset=['age', 'sex', 'height', 'weight'])
    flowchart['has_demographics'] = len(eligible)
    logging.info(f"After requiring demographics: {len(eligible)} cases")

    # === Step 3: 批量检查必要的监测轨道 ===
    logging.info("Batch fetching track names for eligible cases...")

    required_tracks = {
        'abp': ['SNUADC/ART', 'SNUADC/FEM', 'Solar8000/ART_MBP'],
        'ppg': ['SNUADC/PLETH'],
        'propofol': ['Orchestra/PPF20_RATE', 'Orchestra/PPF20_CE']
    }

    # 获取所有 eligible caseids
    eligible_ids = eligible['caseid'].tolist()
    # 批量获取 track name 列表（DataFrame: caseid, tnames)
    tracks_df = vitaldb.get_track_names(caseids=eligible_ids)
    # 将 tnames 列展开为 Python list (already list)
    tracks_map = dict(zip(tracks_df['caseid'], tracks_df['tnames']))

    eligible_caseids = []
    track_availability = []

    for caseid in tqdm(eligible_ids, total=len(eligible_ids), desc="Checking tracks"):
        try:
            track_names = tracks_map.get(caseid, [])

            has_abp = any(t in track_names for t in required_tracks['abp'])
            has_ppg = any(t in track_names for t in required_tracks['ppg'])
            has_propofol = any(t in track_names for t in required_tracks['propofol'])

            track_availability.append({
                'caseid': caseid,
                'has_abp': has_abp,
                'has_ppg': has_ppg,
                'has_propofol': has_propofol,
                'all_required': has_abp and has_ppg and has_propofol
            })

            if has_abp and has_ppg and has_propofol:
                eligible_caseids.append(caseid)
        except Exception as e:
            logging.warning(f"Case {caseid}: Error processing tracks - {e}")
            continue

    flowchart['has_required_tracks'] = len(eligible_caseids)
    logging.info(f"After requiring ABP + PPG + Propofol: {len(eligible_caseids)} cases")

    # === Step 4: 保存结果 ===
    # 4.1 保存 flowchart 数据
    flowchart_path = safe_path(WORK / "outputs" / "metrics" / "cohort_flowchart_numbers.json")
    flowchart_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flowchart_path, 'w') as f:
        json.dump(flowchart, f, indent=2)
    logging.info(f"Saved flowchart data: {flowchart_path}")

    # 4.2 保存 eligible caseids
    eligible_df = pd.DataFrame({'caseid': eligible_caseids})
    eligible_path = safe_path(WORK / "outputs" / "metrics" / "eligible_caseids.csv")
    eligible_df.to_csv(eligible_path, index=False)
    logging.info(f"Saved eligible caseids: {eligible_path}")

    # 4.3 保存 track availability 详情
    track_df = pd.DataFrame(track_availability)
    track_path = safe_path(WORK / "outputs" / "metrics" / "track_availability.csv")
    track_df.to_csv(track_path, index=False)
    logging.info(f"Saved track availability: {track_path}")

    # === Step 5: 统计摘要 ===
    logging.info("\n=== COHORT SELECTION SUMMARY ===")
    logging.info(f"Total VitalDB cases: {flowchart['total']}")
    logging.info(f"Age >= 18: {flowchart['age_18_plus']} ({flowchart['age_18_plus']/flowchart['total']*100:.1f}%)")
    logging.info(f"General anesthesia: {flowchart['general_anesthesia']} ({flowchart['general_anesthesia']/flowchart['total']*100:.1f}%)")
    logging.info(f"Exclude cardiac: {flowchart['exclude_cardiac']} ({flowchart['exclude_cardiac']/flowchart['total']*100:.1f}%)")
    logging.info(f"Has demographics: {flowchart['has_demographics']} ({flowchart['has_demographics']/flowchart['total']*100:.1f}%)")
    logging.info(f"Has required tracks: {flowchart['has_required_tracks']} ({flowchart['has_required_tracks']/flowchart['total']*100:.1f}%)")
    logging.info(f"\nFINAL ELIGIBLE COHORT: {len(eligible_caseids)} cases")

    logging.info("\n=== DONE ===")

if __name__ == "__main__":
    main()
