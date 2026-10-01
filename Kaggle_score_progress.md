# Kaggle Competition Score Progress
## Heavy Equipment Selling Price Prediction — IITM BS MLP Project
### Student: 22f2001153

---

## Quick Reference

| Item | Value |
|---|---|
| **Competition Cutoff** | **0.15000** |
| **Classmate Best Score** | **0.13100** |
| **Current Best Score** | **0.18527 (Phase 4)** |
| **Allowed Libraries** | NumPy, Pandas, Scikit-learn, LightGBM, CatBoost |
| **Banned Libraries** | PyTorch, TensorFlow, NLTK, word2vec, textblob, Pickle, AutoML |
| **3-Model Requirement** | ✅ LightGBM-1 + LightGBM-2 + CatBoost |
| **Data** | Train: 138,701 rows × 50 cols, Test: 15,000 rows |

---

## REUSABLE PROMPT — Copy Every Time You Share Results

> Copy and paste exactly this when you run a new notebook and want to continue:

```
Hi Antigravity, resuming the heavy equipment price prediction Kaggle competition.

CURRENT STATE:
- Notebook just run: 22f2001153_Improved_v[XX].ipynb
- New Kaggle score: 0.XXXXX (was 0.XXXXX in previous version)
- Output folder: "D:\Download\AI MLP Project\my notebook\Improve_try_[XX]_Score_0.XXXXX"

Please:
1. Read "D:\Download\AI MLP Project\Kaggle_score_progress.md" for full context
2. Analyze the output folder (fold scores, feature importances, blend weights)
3. Plan + implement the next version to improve my score
4. Update Kaggle_score_progress.md with the new results
5. Verify the new notebook (no leakage, no banned libraries, 3 models)

Competition cutoff: 0.15000 | Classmate best: 0.13100
```

---

## Score History

| Version | OOF RMSLE | Kaggle Score | Kaggle Gain | Key Changes |
|---|---|---|---|---|
| v14 | 0.19542 | 0.18964 | baseline | Leak-fixed baseline (5-fold CV, group stats inside fold loop) |
| v15 | 0.19415 | 0.18847 | **-0.00117** | 7-col OOF TE, dual LGB (seed diversity), region x InventoryGroup TE |
| v16 | 0.19359 | 0.18761 | **-0.00086** | Cyclical sin/cos dates, 3 region interaction TEs, MachineAge², LogAge |
| v17 | 0.19359 | 0.18757 | **-0.00004** | 3 time TEs (sparse, mostly noise) — see root cause below |
| v18 | 0.19326 | 0.18732 | **-0.00025** | 4 dense TEs (AgeBucket/IC/RegionCode/SubClass x SaleYear), num_leaves=1023 |
| v19 | 0.19354 | 0.18777 | **+0.00045** | ❌ REGRESSION: SaleMonth TEs overfit to training seasonality |
| v20 | 0.19352 | 0.18777 | **+0.00045** | ❌ REGRESSION: 13 TEs = estimation noise > signal gain |
| v21 | TBD | TBD | — | Back to v18 exactly (10 TEs) + 3 rank-based features (AgeRank, HoursRank) |
| **Phase 4** | **0.19190** | **0.18527** | **-0.00205** | ✅ NEW BEST: 3-model ensemble (Rich LGB 1023L + Static LGB 511L + XGBoost), leakage-safe nested TEs + AssetID history, segment blend |
| **Phase 5** | TBD | TBD | — | XGBoost fix (5000→15000 iter, lr 0.020→0.012) + ManufactureYear fold-safe imputation |

### Generalization Gap Analysis

| Version | OOF | Kaggle | Gap |
|---|---|---|---|
| v14 | 0.19542 | 0.18964 | 0.00578 |
| v15 | 0.19415 | 0.18847 | 0.00568 |
| v16 | 0.19359 | 0.18761 | 0.00598 |
| v17 | 0.19359 | 0.18757 | 0.00598 |
| v18 | 0.19326 | 0.18732 | 0.00594 |
| v19 | 0.19354 | 0.18777 | 0.00577 |
| v20 | 0.19352 | 0.18777 | 0.00575 |
| Phase 4 | 0.19190 | 0.18527 | 0.00663 |
| **Avg** | — | — | **~0.00590** |

- To reach Kaggle **0.150** (cutoff) → need OOF ~ **0.1559**
- To reach Kaggle **0.131** (classmate) → need OOF ~ **0.1369**

> **v19 root cause:** SaleMonth TEs overfit training seasonality. OOF 0.19326→0.19354.
>
> **v20 root cause:** 13 interaction TEs (vs v18's 10). Each TE runs a nested 5-fold inner-CV — 3 extra TEs = 3 extra sources of estimation variance. AgeBucket×RegionCode (#15), SubClass×AgeBucket (#26), BaseClass×AgeBucket (#24) had real signal but noise outweighed gain. OOF 0.19352 vs v18 0.19326.
>
> **v21 strategy:** Return to exactly v18's 10 TEs. Add 3 **rank-based** features (AgeRank_SubClass_Year corr=-0.374, AgeRank_Config corr=-0.281, HoursRank_Config corr=-0.192). Rank features use only X (not y), so zero estimation noise. Completely orthogonal information to TEs.

---

## Architecture (Fixed Since v14)

### Models — 3 Models (satisfies project requirement)

| Model | Key Params | Role |
|---|---|---|
| **LightGBM-1** | num_leaves=**1023** (v18), lr=0.010, seed=42, colsample=0.70 | Primary predictor |
| **LightGBM-2** | num_leaves=**1023** (v18), lr=0.010, seed=123, colsample=0.65 | Diversity via different seed/colsample |
| **CatBoost** | max_leaves=255, lr=0.010, grow_policy=Lossguide, l2=3.0 | Native categorical handling |

All: n_estimators=15,000, early_stopping patience=300, 5-fold KFold CV.

### Blend Weights History (Auto-searched per run)

| Version | LGB1 | LGB2 | CatBoost |
|---|---|---|---|
| v15 | 0.35 | 0.40 | 0.25 |
| v16 | 0.30 | 0.45 | 0.25 |
| v17 | 0.30 | 0.40 | 0.30 |

---

## Full Feature Engineering Inventory (v17 state — ~121 features total)

### Static Features — inside `preprocess_df()`, no leakage

| Group | Features | First Added |
|---|---|---|
| Date (raw) | SaleYear, SaleMonth, SaleDay, SaleDayOfWeek, SaleDayOfYear, SaleQuarter, SaleWeek | v14 |
| Date (cyclical) | SaleDayOfYear_sin/cos, SaleMonth_sin/cos, SaleDayOfWeek_sin/cos | v16 |
| SaleDay cyclical | SaleDay_sin, SaleDay_cos | **v17** |
| Date flags | IsMonthEnd, IsMonthStart, IsQuarterEnd, IsQuarterStart, IsYearEnd, IsYearStart | v14 |
| Market trend | Years_Since_2000 | **v17** |
| Machine Age | MachineAge, MachineAge_sq, LogAge | v14/v16 |
| Hours | LogHours, HoursPresent, UsagePerYear | v14 |
| Hours × Age | LogHours_x_LogAge | **v17** |
| Asset history | Asset_Prev_Sales_Count, Has_Asset_History, Asset_Prev_Last_Price, Asset_Prev_Mean_Price, Asset_Days_Since_Last_Sale | v15/v16 |
| Ordinal | UtilizationTierOrd, AssetScaleOrd, AgeDecade, IsNewMachine, IsOldMachine | v14 |

### Fold-Level Features — inside CV loop, computed from training fold only

| Group | Details | First Added |
|---|---|---|
| Frequency encoding | ProductConfigID, Spec_FullDescriptor, Spec_BaseClass, VendorPartnerID | v14 |
| Group means + relative | 6 groupers × 2 targets (Hours, MachineAge) = 24 features | v14/v16 |
| Single-col TE (7 cols) | Spec_FullDescriptor, Spec_BaseClass, ProductConfigID, FunctionalClassification, Spec_SubClass, Spec_VariantModifier, RegionCode | v15 |
| Region interaction TE (3) | InventoryGroupDescription×Region, FunctionalClassification×Region, Spec_SubClass×Region | v15/v16 |
| Time interaction TE (3) | ProductConfigID×SaleYear (η²=0.84), Spec_BaseClass×SaleYear (η²=0.52), FunctionalClassification×SaleYear (η²=0.39) | **v17** |

---

## v17 Feature Importance — LGB1 Last Fold (Top 20)

| Rank | Feature | Importance | Notes |
|---|---|---|---|
| 1 | RegionCode | 103,814 | Raw categorical |
| 2 | MachineAge_rel_ProductConfigID | 95,215 | Relative age |
| 3 | ProductConfigID | 83,173 | Raw categorical |
| 4 | SaleDayOfYear_cos | 79,262 | Cyclical (v16) |
| **5** | **ProductConfigID×SaleYear_TE** | **76,770** | **NEW v17 — η²=0.84** |
| 6 | SaleDayOfYear_sin | 76,149 | Cyclical (v16) |
| 7 | SaleDayOfYear | 71,906 | Raw |
| 8 | MachineAge_rel_Spec_BaseClass | 71,842 | |
| 9 | MachineAge_rel_RegionCode | 69,163 | |
| **10** | **SaleDay_sin** | **63,398** | **NEW v17 cyclical** |
| **11** | **Spec_BaseClass×SaleYear_TE** | **63,238** | **NEW v17 — η²=0.52** |
| **12** | **FunctionalClass×SaleYear_TE** | **61,082** | **NEW v17 — η²=0.39** |
| 13 | MachineAge_rel_FunctionalClassification | 60,761 | |
| 14 | Spec_SubClass×RegionCode_TE | 60,375 | v16 |
| 15 | RegionCode_TE | 59,979 | |
| 16 | InventoryGroupDesc×RegionCode_TE | 59,398 | v15 |
| 17 | FunctionalClass×RegionCode_TE | 58,486 | v16 |
| 18 | SaleDay | 58,246 | Raw |
| 19 | MachineAge_rel_VendorPartnerID | 56,489 | |
| 20 | Spec_BaseClass | 52,102 | |

---

## v18 Plan — Density-First Interaction TEs

### Why v17 failed (root cause)

| Interaction (v17) | Avg samples/cell in fold | Result |
|---|---|---|
| ProductConfigID × SaleYear | **6.8/cell** | Near-zero info (collapsed to global mean) |
| Spec_BaseClass × SaleYear | **14.6/cell** | Marginal |
| FunctionalClass × SaleYear | **102/cell** | ✅ This one actually worked |

### v18 Result (CONFIRMED WORKING)

| Interaction | Avg/cell | OOF Rank | Importance |
|---|---|---|---|
| RegionCode × SaleYear | **110/cell** | #1 | 116,945 |
| InventoryGroupCategory × SaleYear | **924/cell** | #3 | 92,179 |
| AgeBucket × SaleYear | **924/cell** | #5 | 87,030 |
| Spec_SubClass × SaleYear | **112/cell** | #10 | 76,306 |

All 4 new dense TEs entered TOP-10. Blend: LGB1=0.25, LGB2=0.45, CAT=0.30.

---

## v19 Plan — Expanding Dense Interaction Space

Ranked by eta² (variance explained by interaction, measured on full train):

| Interaction | eta² | Avg/cell | Verdict |
|---|---|---|---|
| FunctionalClassification × AgeBucket | **0.6841** | 384/cell | ✅ SAFE |
| FunctionalClassification × SaleMonth | 0.3628 | 153/cell | ✅ SAFE |
| InventoryGroupCategory × AgeBucket | 0.3332 | 3,963/cell | ✅ SAFE |
| AgeBucket × RegionCode | 0.1767 | 430/cell | ✅ SAFE |
| AgeBucket × SaleMonth | 0.1499 | 1,849/cell | ✅ SAFE |
| RegionCode × SaleMonth | 0.0377 | 182/cell | ✅ SAFE |

Key: `FunctionalClassification × AgeBucket` has eta²=0.68 (how equipment size-class price varies by age) — strongest untapped signal.

---

## Data Facts

| Item | Value |
|---|---|
| Price range | $7,500 – $142,000 |
| Target transform | log1p(price) → reverse with expm1() |
| Categorical features | ~40 |
| Missing ManufactureYear | 10.6% (MachineAge=NaN, handled by tree models natively) |
| Prior asset history | 14.4% of rows |
| Unique ProductConfigIDs | 3,594 |
| Unique SaleYears | 24 (1990–2013) |

---

## File Locations

| File | Path |
|---|---|
| **This progress file** | `D:\Download\AI MLP Project\Kaggle_score_progress.md` |
| Dataset | `D:\Download\AI MLP Project\Kaggle Dataset\` |
| Best notebook (v18) | `D:\Download\AI MLP Project\22f2001153_Improved_v18.ipynb` |
| v19 Kaggle output | `D:\Download\AI MLP Project\my notebook\Improve_try_19_Score_0.18777\` |
| v20 Kaggle output | `D:\Download\AI MLP Project\my notebook\Improve_try_20_Score_0.18777\` |
| v21 notebook | `D:\Download\AI MLP Project\22f2001153_Improved_v21.ipynb` |
| Analysis scripts | `D:\Download\AI MLP Project\analyze_v*.py` |

---

*Last updated: v20 result=0.18777 (regression, 13 TEs too many) | v21 ready | Best so far: v18=0.18732*
