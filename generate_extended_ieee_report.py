"""Build the rubric-aligned, extended IEEE-style final project report.

Every numerical result is read from the verified local artifacts.  This script
does not train models or calculate new test-set scores.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_ieee_report as layout


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports/final/Solar_Flare_Forecasting_IEEE_Report.pdf"
FIGURES = ROOT / "reports/final"


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def add(flow: list, *paragraphs: str):
    flow.extend(layout.p(text) for text in paragraphs)


def heading(flow: list, number: str, title: str, *paragraphs: str):
    flow.append(layout.sec(number, title))
    add(flow, *paragraphs)


def subheading(flow: list, title: str, *paragraphs: str):
    flow.append(layout.sub(title))
    add(flow, *paragraphs)


def chart_data(labels: dict, leakage: dict):
    leads = [0, 6, 12, 24, 48]
    valid = [labels["per_lead"][f"{h}h"]["valid"] for h in leads]
    positive = [labels["per_lead"][f"{h}h"]["positive"] for h in leads]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.0), layout="constrained")
    ax1.bar([str(h) for h in leads], valid, color="#24749A")
    ax1.set(title="Valid windows by minimum lead", xlabel="Lead (hours)", ylabel="Windows")
    ax1.ticklabel_format(axis="y", style="plain")
    ax2.plot(leads, [100 * p / v for p, v in zip(positive, valid)], "o-", color="#BF652C")
    ax2.set(title="Positive fraction among valid windows", xlabel="Lead (hours)", ylabel="Positive windows (%)")
    ax2.set_xticks(leads)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=.2)
    fig.savefig(FIGURES / "cohort_and_prevalence.png", dpi=210)
    plt.close(fig)

    names = ["Random", "Region", "Chronological"]
    overlap = [100 * leakage["splits"][s]["overlapping_test_fraction"] for s in
               ("random_split", "region_split", "time_split")]
    fig, ax = plt.subplots(figsize=(6.6, 3.1), layout="constrained")
    bars = ax.bar(names, overlap, color=["#BF652C", "#24749A", "#24749A"])
    ax.set(ylabel="Test windows with training observation overlap (%)", ylim=(0, 109))
    for bar, value in zip(bars, overlap):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(FIGURES / "split_overlap.png", dpi=210)
    plt.close(fig)


def metric(rows, phase, lead, split, features, model="xgboost"):
    return layout.metric(rows, phase, lead, split, features, model)


def result_table(flow, title, headers, rows, widths):
    flow.append(layout.p(title, "CaptionI"))
    flow.append(layout.table(headers, rows, widths))


def main():
    layout.build_visuals()
    rows = list(csv.DictReader((ROOT / "reports/phase4/phase4_results.csv").open(
        newline="", encoding="utf-8")))
    verified = read_json("reports/phase4/phase4_results.json")
    labels = read_json("data/phase2/horizon_targets_report.json")
    leakage = read_json("data/leakage_summary.json")
    cleaning = read_json("data/phase1/cleaning_report.json")
    sensitivity = read_json("reports/phase2/validation_sensitivity.json")
    if len(rows) != 33 or verified["verified_run_count"] != 33 or verified["warnings"]:
        raise ValueError("Expected 33 verified runs with no warnings")
    if labels["supplied_24h_label_comparison_on_valid_rows"]["mismatched"] != 0:
        raise ValueError("The supplied H=0 labels do not match")
    chart_data(labels, leakage)
    flow = [
        layout.p("Investigating the Early-Warning Capability and Generalization of Machine Learning Models for Solar Flare Forecasting", "TitleI"),
        layout.p("Harshit Dharmesh Pandya (240968444), Prasiddhi Singhal (240968448), Mohammed Ahsan Bagsaria (240968484)<br/>Department of Data Science and Engineering, Manipal Institute of Technology, Manipal", "AuthorI"),
        layout.p("<b>Abstract-</b> This study evaluates major solar-flare prediction on SWAN-SF while controlling two evaluation problems: overlapping observations and the meaning of forecast lead time. A 12-hour magnetic window is issued at its end. The binary target is an M- or X-class flare peak in the fixed 24-hour interval beginning H = 0, 6, 12, 24, or 48 hours later. We reconstruct targets from full active-region series, compare four model families at H = 0, and test XGBoost across the four positive leads with both baseline and temporal features. Random, active-region, and chronological splits answer different generalization questions. Of 49,815 original random-test windows, 49,758 share observation time with training; independent region and chronological tests have no measured train-test observation overlap. Thirty-three saved runs pass reload and metric verification. Region-test baseline XGBoost average precision falls from 0.413 at 6 hours to 0.259 at 48 hours. Paired region-bootstrap intervals for temporal-minus-baseline average precision all contain zero. These retrospective results describe lead to the recorded flare peak, not lead to onset or operational forecast skill.", "AbstractI"),
        layout.p("<b>Index Terms-</b> solar-flare forecasting, SWAN-SF, sliding-window leakage, active regions, lead time, average precision, XGBoost.", "KeywordI"),
        layout.FrameBreak(),
    ]

    heading(flow, "I", "Introduction and Motivation",
        "Strong solar flares are observed in the evolving magnetic structure of solar active regions. Forecasting a future M- or X-class peak from measurements available now is a useful scientific benchmark because it forces models to extract information from noisy, time-dependent data. Forecasts that anticipate an event earlier may be more useful than those issued shortly before it, but earlier warning changes the prediction task and can reduce discriminative information. The source of every predictor and the time at which it becomes available therefore matter as much as the classifier itself [1], [2].",
        "SWAN-SF supplies active-region magnetic time series and pre-sliced labeled windows. It is particularly convenient for repeatable machine-learning experiments, yet adjacent examples can be nearly the same observation interval. If highly overlapping windows appear on both sides of a random train-test split, a held-out score can overstate performance on genuinely unseen regions or later periods. The data preparation and validation strategy must preserve the temporal unit being forecast, rather than treating every window as an independent case [1], [2].",
        "The team's synopsis proposed baseline models, explicit forecast horizons, temporal summaries, a leakage study, and interpretation. This report documents the experiments actually completed. The achieved study is narrower than a deployed flare-warning system: it evaluates a retrospective binary target based on recorded peak annotations. The work nevertheless makes the target, rows, splits, preprocessing, trained models, and evaluation reproducible from saved artifacts. All numerical claims below refer to the verified local rebuild unless an external citation is attached.",
        "The central question is how the measured performance of standard classifiers and engineered temporal features changes when major-flare peaks are required to lie farther into the future and evaluation prevents train-test observation overlap. We also ask what the resulting scores can and cannot establish about reliable early warning. Average precision (AP) is emphasized because positive windows are uncommon; thresholded measures and false-alarm counts are reported alongside it.")

    heading(flow, "II", "Literature Review",
        "Angryk et al. introduced the SWAN-SF multivariate benchmark and documented its active-region observations and flare-related labels [1]. The benchmark made it easier to compare learning methods on common data, while leaving researchers responsible for stating exactly how samples are split and how future events are labeled. The present study retains the same magnetic-data setting but constructs lead-specific labels directly from full series. We do not assume that a score from a different label definition or test protocol is numerically comparable.",
        "Goodwin et al. studied simulated real-time flare prediction with stationary, rolling, and expanding training windows and considered solar-cycle effects [2]. Their work supports evaluation that respects when information becomes available. Our chronological split is a simpler holdout by dataset partition; it does not simulate recurring retraining or online decisions. Ji and Aydin investigated active-region forecasting with an interval-forest approach [3], illustrating a different route to interpretable temporal subwindows. Our lag and rolling summaries are lower-cost hand-engineered alternatives, not a reproduction of that method.",
        "Pelkum Donahue and Inceoglu applied a transformer to solar-flare forecasting [4]. Wen and Angryk studied augmentation for severe class imbalance [5], and Wen et al. later examined outlier handling [9]. EskandariNasab et al. investigated preprocessing and contrastive-learning choices for multivariate solar-flare series [6]. These papers make clear that architecture, class distribution, cleaning, and sequence representation can all influence results. Our experiment holds a small set of model definitions fixed, preserves potentially meaningful extreme magnetic values, and tests whether the chosen temporal summaries add AP on identical held-out rows.",
        "Recent studies also include transformer-based operational forecasting [7], multiclass convolutional classification [8], and multimodal or sequence-based architectures [11]-[13]. Such methods can address different sensors, flare classes, or prediction windows. They are useful context for future work, but a direct winner ranking would require rerunning them with the same target, eligible windows, and independent splits. Camporeale and Berger emphasize careful forecast verification in an operational context [10]. We cite that perspective when discussing the gap between retrospective window scores and event-level warning performance.")

    heading(flow, "III", "Survey and Comparative Analysis",
        "Table I groups recent research by the comparison it contributes to this report. The survey covers more than ten research papers, primarily published from 2023 through 2026. It is a methodological survey: the purpose is to identify common design choices and gaps, not to compare published headline scores that use different denominators. The cited SWAN-SF dataset paper [1] anchors the data provenance and is older than the survey range.")
    result_table(flow, "TABLE I. RECENT SOLAR-FLARE FORECASTING LITERATURE",
                 ["Study", "Method and relevance"], [
        ["Goodwin et al. [2]", "Simulated real-time training; solar-cycle and window effects."],
        ["Ji and Aydin [3]", "Interval forest; temporal subwindow interpretation."],
        ["Pelkum Donahue and Inceoglu [4]", "Transformer forecasting from active-region series."],
        ["Wen and Angryk [5]", "Augmentation for extreme class imbalance."],
        ["EskandariNasab et al. [6]", "Preprocessing and representation learning."],
        ["Abduallah and Wang [7]", "Deep-learning operational-horizon framing."],
        ["Zhang et al. [8]", "Multiclass one-against-rest CNN."],
        ["Wen et al. [9]", "Outlier filtering and robustness."],
        ["Camporeale and Berger [10]", "Operational forecast verification."],
        ["Riggi et al. [11]", "Transformer models across modalities."],
        ["Azizian Foumani et al. [12]", "CNN with multivariate magnetic series."],
        ["Alatoom and Nikolaou [13]", "Multimodal transformer at active-region level."],
    ], [83, layout.COL - 7 - 83])
    add(flow,
        "Three recurring survey themes inform our protocol. First, class imbalance makes accuracy alone misleading; a model could classify nearly everything as non-flaring and still achieve high accuracy. Second, useful warning needs a precise issue time and event interval, so a nominal '24-hour model' cannot be compared fairly without its label rule. Third, multiple windows from one active region are statistically related. We therefore report prevalence and AP, define a fixed-width target interval after an explicit lead, and retain region identity in both splitting and uncertainty calculations.",
        "The literature also shows a breadth-versus-control trade-off. Sophisticated sequence architectures may improve a task, but their additional capacity does not answer whether evaluation leakage or a target change explains a reported gain. This project begins with classical and compact neural baselines and then isolates one feature-set comparison under identical test rows. Its contribution is primarily a transparent evaluation of the specified local SWAN-SF pipeline, not a claim to introduce a new state-of-the-art architecture.")

    heading(flow, "IV", "Research Gaps, Problem Statement, and Objectives",
        "The gap addressed here is an integrated evaluation of three choices on one saved data contract: how overlapping windows are split, how minimum lead to a flare peak is defined, and whether inexpensive temporal summaries add information to a magnetic-value baseline. Prior work addresses parts of this broader subject [2]-[13], so the gap is not that leakage, imbalance, or temporal modeling has never been studied. The practical concern is that a project can appear to achieve early warning if random near-duplicate windows, mismatched horizons, or a favorable metric are interpreted too broadly.",
        "<b>Problem statement:</b> Given the 12-hour observation window of an active region and only magnetic measurements available by its end, estimate the probability of at least one recorded M/X flare peak in (issue time + H, issue time + H + 24 hours] for H in {0, 6, 12, 24, 48}. Evaluate the probability ranking and validation-selected binary decision on held-out active regions and later dataset partitions. Quantify observation overlap under a random split as a diagnostic. The target describes a future peak, not the onset of a flare.",
        "<b>Objective O1:</b> train reproducible baseline classifiers on the H=0 binary task. <b>O2:</b> construct and compare four minimum-lead tasks with a common eligible cohort. <b>O3:</b> test whether lag, rate, and rolling features improve AP over the baseline on the same held-out rows. <b>O4:</b> measure overlap and compare random, grouped-region, and chronological protocols. <b>O5:</b> describe which features the fitted model uses and investigate class-conditioned observation patterns without assigning causal meaning. <b>O6:</b> verify that saved model artifacts reproduce the reported scores. Section X states which objectives were achieved and their limits.")

    heading(flow, "V", "Methodology",
        "The implemented system has two input branches: pre-sliced SWAN-SF windows for predictors and full active-region series for event annotations. The branches are joined by the Phase 1 row index. The target is checked for future coverage before training rows are selected. The same row order and feature names are then used across model families. Figure 1 is the overall methodology block diagram; Figure 4 records the two-stage modeling architecture. They show operations that the code and saved artifacts actually perform.",
        "The design separates three clocks that are easy to collapse accidentally. The first is the observation interval, from which all predictors must come. The second is the issue time at the end of that interval. The third is the future validity interval used to assign the binary target. This separation permits a direct audit for look-ahead: no event annotation or measurement after issue time appears among the 24 baseline or 120 temporal predictor columns. Future series records are accessed only to determine target status and whether the complete target interval was observed.",
        "The row index is the contract between these branches. Each pre-sliced window has one row in the Phase 1 feature index; its baseline and temporal arrays use exactly that order. The lead-target matrix uses the same order, with one target column for each H. Rather than re-extracting slightly different windows for each model, the experiment chooses rows from this common index. It then intersects target validity with a saved split assignment. This makes model-family and feature-set comparisons within a lead traceable to the same underlying examples.",
        "The two independent splits answer complementary questions. Holding out complete active regions asks whether a model transfers to regions absent from fitting. Holding out later source partitions asks whether it transfers across the dataset's partition chronology. A strong result on one split cannot substitute for the other, because source and class compositions differ. Random assignment is intentionally kept only to demonstrate the overlap problem with a concrete diagnostic score. These choices were fixed in the shared Phase 1 data contract before the later model matrix was evaluated.",
        "The model matrix likewise separates breadth from lead-time depth. Phase 2 compares four common classifier families at the supplied 24-hour task to establish how a linear model, bagged trees, boosted trees, and a compact neural network behave on the shared features. Phase 3 then holds the boosted-tree family fixed while changing lead and feature set. This avoids attributing a lead effect to an architecture switch. The 33 saved runs are not 33 independently invented experiments; they are a structured cross of phase, lead, split, representation, and model family, with one extra random diagnostic.",
        "Finally, every scored run has a validation stage and a separate test stage. Early stopping, where used, and the F1 decision threshold are selected with validation data. AP and ROC-AUC use saved test probabilities; F1, TSS, HSS, and confusion counts use the frozen threshold. A second evaluator reloads the saved model and checks row IDs, hashes, probabilities, and recomputed scores. This final step makes the report a description of recoverable artifacts rather than an isolated notebook output. The next subsections give the exact data, preprocessing, features, splits, model settings, and metric definitions behind Figure 1.")
    flow += layout.wide_visual("reports/final/pipeline.png",
        "Fig. 1. Overall methodology block diagram. Inputs, target construction, splitting, preprocessing, fitting, and verification match the saved implementation. Source: authors' project code and artifacts.")

    subheading(flow, "A. Data collection and description",
        "The team downloaded the seven SWAN-SF Harvard Dataverse archives: SWAN.tar.gz, five partition instance archives, and addenda.tar.gz [1]. They were extracted into one local data directory. The full-series branch contains 4,098 active-region series. The pre-sliced branch contains 331,185 labeled 12-hour windows from five partitions and 3,205 distinct region identifiers. These counts describe the team's extracted files and row mapping; they are not assumed to be universal counts for all versions of SWAN-SF.",
        "There are 6,234 supplied positive FL windows and 324,951 NF windows, for a 1.88% positive fraction. FL corresponds to the supplied major-flare binary task; NF includes windows without an M/X target event under that labeling rule. The full-series tables contain magnetic measurements and flare-related information at nominal 12-minute cadence. The latter information is used for target construction and coverage checks, never as a predictive input. The 24 retained magnetic parameters include TOTUSJH, TOTBSQ, TOTPOT, ABSNJZH, SAVNCPP, USFLUX, and R_VALUE. Their names and order are stored in the feature-columns artifact.",
        "The full-series scan counted 2,879,658 unique observation rows and 6,082,002 missing or nonfinite magnetic cells out of 69,111,792 potential cells, or about 8.8%. A cell-level missing fraction does not imply the same fraction of windows is unusable: imputation treats individual feature values, while future-label validity requires coverage in the target interval. The two missingness questions are kept separate throughout the workflow. Data provenance and the stable row index are recorded in the extraction manifest and Phase 1 feature index.")
    result_table(flow, "TABLE II. LOCAL DATA CONTRACT AND COUNTS",
                 ["Artifact", "Observed content"], [
        ["Full series", "4,098 region series; event annotations and magnetic readings"],
        ["Pre-sliced windows", "331,185 windows; 12 h each; five partitions"],
        ["Region IDs", "3,205 distinct IDs in the window index"],
        ["Original labels", "6,234 FL; 324,951 NF; 1.88% FL"],
        ["Baseline array", "331,185 by 24 magnetic values"],
        ["Temporal array", "331,185 by 120 summaries"],
    ], [83, layout.COL - 7 - 83])

    subheading(flow, "B. Target definition and eligible rows",
        "An example's issue time is the end of its observed 12-hour window. For each lead H, the reconstructed binary target is one if an MFLARE/XFLARE peak annotation occurs after issue + H and no later than issue + H + 24 hours. The fixed 24-hour interval makes H the minimum lead to the recorded peak. It does not make the prediction a statement about whether a flare begins exactly H hours later, nor about events anywhere in the next H hours. The H=0 task corresponds to the supplied 24-hour benchmark on windows for which future data are available.",
        "A label is accepted only if the source series extends through the end of the target interval and at least 114 of the expected 120 twelve-minute samples are present. Missing coverage creates an unknown target, represented separately from negative. This prevents unobserved future time from being silently interpreted as 'no flare.' Across 299,175 eligible H=0 windows, reconstructed and supplied binary labels matched with zero mismatches. This agreement checks code consistency; it does not independently validate all flare annotations.",
        "At longer leads, the number of valid windows falls because more future time must be observed. H=48 retains 210,341 valid windows and excludes 120,844 of the original 331,185. Phase 3 uses the 210,341 windows with valid targets at every positive lead so baseline-versus-temporal comparisons and lead plots share an eligibility cohort. The number of positives can still change with H because each lead shifts the target interval. A long-lead result therefore answers a different label task, even on the same observation rows.")
    target_rows = [[str(h), f"{labels['per_lead'][f'{h}h']['valid']:,}",
                    f"{labels['per_lead'][f'{h}h']['positive']:,}",
                    f"{labels['per_lead'][f'{h}h']['censored']:,}"] for h in (0, 6, 12, 24, 48)]
    result_table(flow, "TABLE III. LEAD-TARGET COVERAGE IN ALL WINDOWS",
                 ["H (h)", "Valid", "Positive", "Unknown"], target_rows,
                 [37, 65, 65, layout.COL - 7 - 167])
    flow.append(layout.fig("reports/final/cohort_and_prevalence.png",
        "Fig. 2. Eligible-window count and positive prevalence by target lead. Source: reconstructed label report.", max_height=190))

    subheading(flow, "C. Preprocessing and missing values",
        "Phase 1 reads magnetic observations within each window in chronological order. It builds the predictor arrays once in a row order shared by targets and split assignments. Missing and nonfinite predictor values remain explicit until each split's training rows determine an imputation median. The same median is then applied to that split's validation and test rows. Training-only means and standard deviations supply the z-score transform. Saved normalization JSON files preserve the statistics for all three split definitions and both feature arrays.",
        "This ordering matters. Computing medians or means over every window before a split would allow the validation and test distributions to influence model inputs. Likewise, selecting features or tuning thresholds on test results would weaken the holdout. The pipeline avoids those steps by storing separate training-fitted statistics and reapplying them at model evaluation. The baseline and temporal arrays are transformed independently, then concatenated for the combined feature set; the order is recorded in each run's configuration.",
        "Phase 1 counted IQR-based outliers descriptively but did not remove or clip them. Large magnetic values can coincide with the phenomena being forecast, so a generic outlier rule could discard useful information. The report therefore does not claim that outlier removal improved any model. No test-time class rebalancing, oversampling, or global scaling was performed. Model-specific class weights address imbalance only during fitting; AP and prevalence are then evaluated on the naturally imbalanced held-out rows.")

    subheading(flow, "D. Feature selection and temporal engineering",
        "The baseline representation contains the final observed value of each of 24 magnetic parameters in the 12-hour window. This is an explicit domain and leakage filter: flare, X-ray, quality, and future columns are excluded. It is not an algorithmic search over arbitrary subsets, and the study does not claim to have discovered an optimal feature set. The fixed baseline allows the incremental value of temporal summaries to be tested without changing the source measurements.",
        "For each retained magnetic parameter, the temporal representation adds its value one and three hours before issue time, a three-hour rate of change, and the final three-hour rolling mean and standard deviation. That yields 24 x 5 = 120 temporal values. The combined matrix has 144 columns after concatenation. Rolling summaries require at least eight valid readings in their local support; otherwise they remain missing for the training-fitted imputer. No feature uses observations after the window end.",
        "Lag, slope, and local variation can represent evolution that a final reading misses. They also expand dimension sixfold and may duplicate much of the last-value information. Therefore a gain is an empirical question, not a methodological guarantee. The main comparison pairs baseline and combined-feature XGBoost on identical split, lead, and held-out row IDs. The paired region bootstrap measures uncertainty in AP difference while preserving within-region window dependence.")

    subheading(flow, "E. Splitting and leakage-control procedure",
        "Three split assignments were prepared with a common feature index. The region split holds all windows from each active-region ID in only one of training, validation, or test. The chronological split trains on source partitions 1-3, validates on partition 4, and tests on partition 5. The random split deliberately mixes windows and is retained for one XGBoost diagnostic run at H=0. Only region and chronological results are used as evidence of generalization; they answer different questions and have different test populations.",
        "The overlap audit compares observation intervals, not only file names. Adjacent same-region windows have a median 91.5% overlap and a median one-hour start-time separation. Of the 49,815 original random-test windows, 49,758 (99.89%) overlap training observations. The corresponding measured train-test observation overlap is zero for the region and chronological assignments. This structural audit establishes the opportunity for leakage; by itself it does not estimate a causal change in model performance.",
        "The random XGBoost AP of 0.551 exceeds its region and time counterparts, but the three test sets have different row membership and prevalence. It would be incorrect to subtract them and call the result an exact leakage penalty. The defensible inference is that the random protocol is unsuitable for an unseen-region or later-partition claim in this dataset. The chronology split also does not imply external deployment validity: the saved source partitions and future solar conditions may still differ from an operational stream.")
    flow.append(layout.fig("reports/final/split_overlap.png",
        "Fig. 3. Fraction of original test windows sharing observation time with training, by split. Source: Phase 1 leakage audit.", max_height=165))

    subheading(flow, "F. Model selection, architecture, and parameters",
        "Phase 2 trains Logistic Regression, Random Forest, XGBoost, and a compact artificial neural network at H=0. Each family uses baseline and combined features under both independent splits, giving 16 primary runs. One random-split baseline XGBoost run adds a leakage diagnostic. Phase 3 fixes the family to XGBoost and crosses four positive leads, two independent splits, and two feature sets for another 16 runs. The total is 33 saved runs, trained sequentially to bound resource use.",
        "Logistic Regression offers a linear, inspectable baseline and uses balanced class weights, the liblinear solver, and a 200-iteration limit. The forest supplies nonlinear interactions with 80 trees, maximum depth 12, minimum leaf size 15, square-root feature sampling, balanced subsample weights, and one worker. XGBoost uses a depth-three CUDA histogram classifier with at most 180 trees, learning rate 0.07, subsample 0.85, column sample 0.85, L2 penalty 3, 64 histogram bins, two workers, and a negative-to-positive class ratio. Validation PR-AUC early stopping has a 20-round patience. The full exact configuration and best iteration are retained with each fitted model.",
        "The ANN uses a 32-unit ReLU layer, 0.15 dropout, a 16-unit ReLU layer, and a sigmoid output. It fits weighted binary cross-entropy with Adam at learning rate 0.001, batches of 2048, at most eight epochs, and early stopping on validation PR-AUC after two unproductive epochs with best weights restored. The installed native Windows TensorFlow trained this model on CPU; XGBoost used the team's RTX 4050. This compact ANN is a baseline, not a broad architecture search or a GPU benchmark.")
    flow += layout.wide_visual("reports/final/architecture.png",
        "Fig. 4. Implemented model architecture and run matrix. The branch labels, feature dimensions, and run counts correspond to saved training configurations. Source: project code and run directory.")
    result_table(flow, "TABLE IV. PRIMARY TRAINING PARAMETERS",
                 ["Family", "Saved specification"], [
        ["Logistic", "liblinear; balanced; 200 max iterations"],
        ["Forest", "80 trees; depth 12; leaf 15; sqrt features"],
        ["XGBoost", "180 trees max; depth 3; eta 0.07; 64 bins; CUDA; PR-AUC stop 20"],
        ["ANN", "32/16 ReLU; dropout 0.15; Adam 0.001; 8 epochs max"],
    ], [62, layout.COL - 7 - 62])

    subheading(flow, "G. Training, validation, and testing procedure",
        "Each run selects valid row IDs for its phase, lead, and split. The training subset fits imputation, scaling, and model parameters; the validation subset supplies early stopping when used and selects a classification threshold that maximizes validation F1. The test subset is evaluated only after these choices are fixed. A probability is produced for each held-out row, and both validation and test probabilities and row IDs are saved with the model, configuration, metrics, and completion marker.",
        "A small validation-only sensitivity study varied XGBoost depth and ANN width across 12 configurations on baseline features in the two main splits. It did not replace the 33-run matrix or select a new test winner. The highest validation AP differed by split; reporting such validation values as held-out performance would be misleading. The implemented parameters reflect a bounded engineering choice rather than an exhaustive grid search. This distinction matters because a single model setting can be suboptimal for one lead or split without invalidating the comparison of the settings actually tested.",
        "The independent evaluator reloads every saved model, checks hashes of the Phase 1 arrays, label file, and normalization settings, reconstructs authoritative held-out row IDs, recomputes metrics, and compares 64 held-out probabilities after reload. All 33 runs passed these checks and no run was rejected. This verifies consistency of the local artifacts and analysis pipeline; it is not proof that the dataset's physical measurements or flare annotations are error-free.")

    subheading(flow, "H. Evaluation measures and uncertainty",
        "AP summarizes the precision-recall ranking and is the main rare-event measure. ROC-AUC is included because it measures ranking across positive-negative pairs, but it can appear favorable when negative windows dominate. At the validation-selected threshold, precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. TSS is recall minus false-positive rate, and HSS adjusts a contingency-table score for chance agreement. Confusion counts make the operationally relevant false-alarm burden visible.",
        "AP is sensitive to positive prevalence and each lead moves its event interval, so a cross-lead AP curve is a description of related tasks, not an isolated estimate of the effect of waiting H more hours. The region and chronological splits also have different held-out populations; their score difference is not a paired test. For baseline-versus-temporal AP, the held-out rows are identical within each lead and split, which supports a paired comparison. The evaluator resamples active regions 500 times and recomputes the AP difference to obtain a 95% region-bootstrap interval. This treats highly overlapping windows as related within regions. It does not fully account for solar-cycle or annotation uncertainty.")

    heading(flow, "VI", "Result Analysis and Performance Evaluation",
        "This section reports held-out results from the 33 verified runs. Every model's probability threshold was chosen on validation data. The tables distinguish AP, which does not use that threshold, from F1 and TSS, which do. Comparisons are interpreted within the tested protocol; they do not imply that the selected architecture is universally best. The full machine-readable results and saved probabilities are retained locally for audit.")
    subheading(flow, "A. H=0 model-family comparison",
        "At H=0, the region test has 45,757 eligible windows and 939 positives; the chronological test has 67,279 and 990 positives. The test cohorts are different because eligibility and split membership intersect. Table V lists all four baseline model families under both independent splits. On unseen regions, ANN has the largest baseline AP among the four, 0.444, followed by XGBoost at 0.421 and logistic regression at 0.393. This ranking concerns AP only; logistic regression has larger region TSS than XGBoost at its selected threshold, 0.469 versus 0.343.",
        "On later partitions, logistic regression has a baseline AP of 0.458 and XGBoost 0.455, a small numerical difference without a reported paired uncertainty comparison between model families. Forest AP increases from 0.264 for the region split to 0.342 for the time split, while ANN AP decreases from 0.444 to 0.391. These observations show that model ranking depends on the held-out population and metric. It would be too strong to name one global winner based on a single score column.")
    model_rows = []
    for model_label, model_name in (("Logistic", "logistic"), ("Forest", "forest"),
                                    ("XGBoost", "xgboost"), ("ANN", "ann")):
        for split_label, split_name in (("Region", "region_split"), ("Time", "time_split")):
            r = metric(rows, "phase2", 0, split_name, "baseline", model_name)
            model_rows.append([model_label, split_label, f"{float(r['ap']):.3f}",
                               f"{float(r['f1']):.3f}", f"{float(r['tss']):.3f}"])
    result_table(flow, "TABLE V. H=0 BASELINE HELD-OUT RESULTS",
                 ["Model", "Split", "AP", "F1", "TSS"], model_rows,
                 [64, 51, 42, 42, layout.COL - 7 - 199])
    add(flow,
        "Adding temporal summaries produces mixed H=0 effects. Region-test XGBoost AP changes only from 0.421 to 0.423, while its TSS falls from 0.343 to 0.310. Chronological XGBoost AP rises from 0.455 to 0.487, while its TSS falls from 0.519 to 0.440 because thresholded decisions change. Chronological logistic AP also reaches 0.487 with temporal features. These differences were not the subject of the paired region bootstrap reported for Phase 3 and should be treated as descriptive H=0 observations.",
        "The random diagnostic XGBoost baseline run reaches AP 0.551 and TSS 0.602 on its own test set. Its high score coincides with 99.89% observation overlap between the original random test windows and training. The overlap mechanism is clear, but the score gap is not an exact measurement of how much overlap inflated performance: region, time, and random test populations differ. The random run is therefore excluded from claims about unseen-region or future-period behavior.")

    subheading(flow, "B. Four minimum-lead experiments",
        "Table VI shows baseline XGBoost AP and TSS at 6, 12, 24, and 48 hours. The four Phase 3 leads use a common 210,341-window eligible cohort before split assignment. The region test contains 32,144 eligible windows for each lead; the chronological test contains 46,395. Positive counts vary with the shifted target interval. Region baseline AP declines across the four tasks from 0.413 to 0.373, 0.321, and 0.259. This is evidence that the tested model ranks long-lead peaks less effectively in the unseen-region cohort, subject to changed labels and prevalences.",
        "The chronological baseline AP sequence is 0.409, 0.367, 0.347, and 0.359. The slight increase from 24 to 48 hours is a reminder that performance need not be monotone when the target interval moves and the event composition changes. The corresponding TSS values are threshold dependent; on region tests they are 0.333, 0.294, 0.344, and 0.235. At H=24, a higher region TSS than H=12 does not overturn the lower AP: the metrics answer different questions and use different decision rules.")
    lead_rows = []
    for h in (6, 12, 24, 48):
        reg = metric(rows, "phase3", h, "region_split", "baseline")
        tim = metric(rows, "phase3", h, "time_split", "baseline")
        lead_rows.append([str(h), str(reg["positive"]), f"{float(reg['ap']):.3f}",
                          f"{float(reg['tss']):.3f}", f"{float(tim['ap']):.3f}",
                          f"{float(tim['tss']):.3f}"])
    result_table(flow, "TABLE VI. BASELINE XGBOOST ACROSS LEADS",
                 ["H", "Reg.+", "Reg.AP", "Reg.TSS", "Time.AP", "Time.TSS"],
                 lead_rows, [25, 38, 42, 47, 46, layout.COL - 7 - 198])
    add(flow,
        "Positive prevalence in the region-test cohort changes from approximately 2.49% at H=6 to 2.14% at H=48. Since AP depends on prevalence, the change from 0.413 to 0.259 combines changes in ranking, class composition, and target definition. One cannot infer from that difference alone that the same physical flare became less predictable precisely 42 hours earlier. A controlled event-level experiment would track fixed events and issue times, then evaluate event alerts rather than only window labels.")

    subheading(flow, "C. Temporal features and paired uncertainty",
        "The primary feature question is whether 120 engineered summaries improve AP relative to the 24-value baseline, conditional on the same rows, lead, split, and XGBoost family. Table VII gives the paired AP difference, combined minus baseline, and a 95% interval from 500 bootstrap resamples of active regions. The interval includes zero in all eight split-lead combinations. Some point estimates are positive, notably +0.0225 at H=24 for the chronological split, and some are negative, notably -0.0174 at H=48 for the same split. The supported conclusion is that these runs do not establish a reliable AP gain from the specified summaries.",
        "A confidence interval containing zero does not prove the feature sets are identical or that temporal information has no predictive value. It means the observed sample and resampling procedure do not isolate a nonzero improvement at the chosen interval level. The 120 summaries may be redundant with the final magnetic state, may need a different architecture, or may help in subsets not resolved by this sample. Conversely, selecting only the best positive point estimate would ignore the other seven comparisons and the uncertainty interval.")
    comparisons = sorted(verified["paired_comparisons"], key=lambda x: (x["lead_hours"], x["split"]))
    comparison_rows = [[str(x["lead_hours"]), "Region" if x["split"] == "region_split" else "Time",
                        f"{x['difference_ap']:+.3f}",
                        f"[{x['ci95_region_bootstrap'][0]:+.3f}, {x['ci95_region_bootstrap'][1]:+.3f}]"
                       ] for x in comparisons]
    result_table(flow, "TABLE VII. TEMPORAL MINUS BASELINE AP",
                 ["H", "Split", "Delta AP", "95% region CI"], comparison_rows,
                 [25, 52, 47, layout.COL - 7 - 124])
    add(flow,
        "The H=24 chronological temporal model illustrates why ranking and decisions both matter. Its AP is 0.369, higher than its baseline counterpart's 0.347. At the validation-selected threshold, recall is 0.485 and precision is 0.338. The confusion counts are 427 true positives, 838 false positives, 454 false negatives, and 44,676 true negatives on 46,395 held-out windows. The model detects some positive windows but raises roughly two false alarms for each true positive at this threshold. A deployed system would require calibrated probabilities and an alert-cost policy; the present study did neither.")

    subheading(flow, "D. Validation sensitivity and artifact checks",
        "The separate 12-configuration validation-only sensitivity check varied XGBoost depth 2, 3, and 4 and ANN hidden widths 16/8, 32/16, and 64/32 on baseline features for the two main splits. Region validation AP was highest for XGBoost depth 2 (0.448) and ANN 64/32 (0.367); time validation AP was highest for XGBoost depth 3 (0.550) and ANN 16/8 (0.594). These are not held-out test values and were not used to choose a new winner for Tables V-VII. They show that parameter sensitivity exists and that a single architecture should not be called universally optimal.",
        "Saved files include 21 XGBoost JSON models, eight scikit-learn joblib models, four Keras models, one configuration and one metrics file per run, and held-out row/probability arrays. Each configuration records the lead, target width, split, feature set, feature order, seed, and training-data hashes. The evaluator checked all 33 model reloads and recomputed metrics without a warning. Reproducibility is local to an environment with the extracted source data and models; the gitignored data directory is not embedded in this PDF.")

    heading(flow, "VII", "Visualization of Results",
        "Figures 5 and 6 plot AP and TSS across the four positive leads for region and chronological tests. They are produced from the verified Phase 3 result matrix. The AP chart shows baseline and temporal curves close together for most lead-split pairs, matching the small paired differences in Table VII. The region curves trend downward as the lead interval moves out, while the chronological curves do not strictly decline. The TSS chart illustrates that a fixed ranking picture does not specify the operating point: validation-selected thresholds produce visibly different thresholded skill trajectories.")
    flow.append(layout.fig("reports/rebuild/phase3/ap_by_lead.png",
        "Fig. 5. Held-out AP by target lead and split. The curves compare different future intervals, not repeated forecasts of a fixed event. Source: verified Phase 3 results.", max_height=190))
    flow.append(layout.fig("reports/rebuild/phase3/tss_by_lead.png",
        "Fig. 6. Held-out TSS at validation-selected thresholds. Source: verified Phase 3 results.", max_height=190))
    add(flow,
        "Figure 7 shows mean absolute TreeSHAP contributions for the H=24 chronological temporal XGBoost model. R_VALUE three-hour mean, TOTUSJH three-hour mean and one-hour lag, and ABSNJZH three-hour variation rank prominently in the saved interpretation sample. This is a model association: it describes which standardized inputs move the fitted tree output on selected rows. Correlated magnetic inputs can share or exchange attribution, and a large contribution is not a causal mechanism for a flare.",
        "Figure 8 displays class-conditioned 12-hour trajectories for R_VALUE, TOTUSJH, and ABSNJZH. The sampling uses one observation window per region to avoid a plot dominated by repeated windows from one active region. The 881 positive H=24 chronological test windows arise from only 20 positive regions, so apparent pattern differences rely on a small set of independent positive groups. The figure supports exploratory description of the saved cohort, not a claim to have discovered a universal pre-flare signature.")
    flow.append(layout.fig("reports/rebuild/phase4/feature_importance.png",
        "Fig. 7. TreeSHAP feature associations for the H=24 chronological temporal XGBoost model. Source: verified Phase 4 interpretation.", max_height=195))
    flow.append(layout.fig("reports/rebuild/phase4/preflare_patterns.png",
        "Fig. 8. Sampled observation-window trajectories by class. Curves are descriptive and use one window per region. Source: verified Phase 4 analysis.", max_height=195))

    heading(flow, "VIII", "Inference from Results",
        "The strongest inference from the leakage study is procedural. A random split of these pre-sliced windows almost always tests observation intervals already represented in training. It therefore cannot support a claim about an unseen active region. The random diagnostic's AP is numerically higher, consistent with that structural risk, but no causal inflation amount is estimated. For project reporting, the region split is the appropriate evidence for new-region generalization and the chronological split is evidence for later-partition performance.",
        "The lead experiment supports a qualified early-warning statement: the tested baseline XGBoost retains above-prevalence AP on unseen regions at all four positive leads, but its AP is lower for the H=48 target than for H=6. The positive fraction at H=48 is about 2.14% in that region test; AP 0.259 exceeds that prevalence baseline but is not itself a calibrated event-warning success rate. Each H defines a different 24-hour event interval, and later leads lose more source windows to coverage requirements. We therefore describe the observed task performance rather than asserting the maximum reliable warning time for a real operator.",
        "The temporal-feature experiment has a more direct paired design. Baseline and combined inputs are evaluated on the same rows, with region-resampled intervals for AP difference. All eight intervals cross zero. This justifies a conservative inference: the added lag and rolling summaries did not deliver a statistically clear AP gain under this model, cohort, and bootstrap procedure. It does not justify saying that all temporal structure is useless. More flexible sequence models, different feature definitions, or larger independent event samples could reach another conclusion.",
        "Thresholded results explain a separate practical issue. The validation F1 threshold is an experimental convention, not necessarily the best threshold for an alert service. At H=24 in the chronological temporal run, 838 false positives accompany 427 true positives. A system with a very high cost of missed events might accept that balance; a system with a high false-alarm cost might not. This report lacks a specified utility function, probability calibration study, or prospective alert evaluation, so it cannot choose an operational threshold.")

    heading(flow, "IX", "Findings and Discussion",
        "<b>Finding 1:</b> Sample dependence is material. The median adjacent-window overlap is 91.5%, and the random test shares training observation time for 99.89% of original windows. Any performance claim that treats those random rows as independent active regions would misdescribe the actual data geometry. This finding arises from direct interval intersection, independent of model family or metric.",
        "<b>Finding 2:</b> Class imbalance shapes interpretation. Only 1.88% of original pre-sliced windows are positive. At H=0, the region-test XGBoost AP 0.421 is meaningful relative to its local prevalence, but it does not translate to 42.1% of future flares detected. AP is a ranking summary across thresholds. Recall, precision, and confusion counts are needed for any selected operating point. Reporting AP together with TSS, F1, and false alarms avoids implying that one number captures all forecast costs.",
        "<b>Finding 3:</b> No single model dominates across protocol and metric. ANN has the largest H=0 baseline AP on unseen regions among the four implemented families, while logistic regression has the largest region TSS at its selected threshold. On the chronological test, logistic and XGBoost baseline AP are similar. These are observed point estimates, not formal pairwise model-family significance tests. The study's purpose was a controlled benchmark rather than exhaustive hyperparameter optimization.",
        "<b>Finding 4:</b> The tested longer-lead target is harder for baseline XGBoost on held-out regions in AP terms. The 6-to-48-hour AP sequence falls 0.413, 0.373, 0.321, 0.259. Because the positive event interval moves and target coverage shrinks, this is a descriptive trend over four tasks. The chronological curve differs, which reinforces the importance of specifying the deployment population before making a general early-warning claim.",
        "<b>Finding 5:</b> The extra 120 temporal summaries do not have a supported AP advantage over the 24 final-state values in the eight Phase 3 paired comparisons. The largest favorable point estimate is H=24 chronological, but its 95% region-bootstrap interval includes zero. The apparently favorable H=24 result remains useful for model inspection, yet it should not be presented as a proven improvement.",
        "<b>Finding 6:</b> Interpretation identifies what the fitted model uses, not what causes a flare. Rolling R_VALUE and TOTUSJH summaries and ABSNJZH variation receive high sampled TreeSHAP attribution. One-window-per-region trajectory plots reduce repeated-window dominance, but the 20 positive chronological H=24 regions leave substantial uncertainty about recurring patterns. The report therefore uses the language of associations and candidate signals rather than a physical precursor discovery.")

    heading(flow, "X", "Objective Achievement",
        "Table VIII maps the synopsis objectives to observed outputs and states what remains unproven. 'Achieved' means the specified experiment or artifact was completed, not that the resulting model is operationally adequate. One objective concerning reliable early warning is only partly answered because the study is retrospective and peak-based. The distinction prevents a completed analysis from being mistaken for a deployed warning capability.")
    result_table(flow, "TABLE VIII. OBJECTIVES, EVIDENCE, AND STATUS",
                 ["Objective", "Evidence and judgement"], [
        ["O1 Baselines", "Achieved: four H=0 families, both independent splits and feature sets; Table V."],
        ["O2 Leads", "Achieved as peak-target tasks: 6/12/24/48 h, common eligible cohort; Table VI."],
        ["O3 Temporal value", "Tested, no clear AP gain: all eight paired CIs include zero; Table VII."],
        ["O4 Leakage", "Achieved structurally: interval overlap audit and three split protocols; Fig. 3."],
        ["O5 Patterns", "Exploratory only: TreeSHAP and region-sampled curves; Figs. 7-8."],
        ["O6 Reproducibility", "Achieved locally: 33 model reload and metric checks, zero rejected."],
        ["Early warning", "Partly answered: retrospective lead-to-peak ranking; operational utility untested."],
    ], [82, layout.COL - 7 - 82])
    add(flow,
        "The synopsis also anticipated comparison of random, time-based, and active-region-based splits. This was completed, but the random result is intentionally interpreted as a diagnostic. The aim to study warning-time trade-offs was addressed with lead-specific AP, TSS, and prevalence, while an event-level cost trade-off was not. The aim to find recurring temporal patterns was pursued through exploratory plots and attributions; their limited region count and correlational character prevent a definitive physical claim.")

    heading(flow, "XI", "Limitations",
        "<b>Target timing:</b> Event annotations refer to the recorded flare peak. A forecast issued H hours before the target interval begins does not guarantee H hours before the physical onset. A flare may start before its peak, and the project did not construct onset labels. The paper therefore avoids the phrase 'predicts a flare H hours before it starts.' A future onset-based event study would need dependable onset records and distinct temporal matching rules.",
        "<b>Window dependence and uncertainty:</b> Region grouping removes direct train-test observation overlap in the audited assignments, but many test windows within a region remain correlated. The paired bootstrap resamples regions rather than windows for the temporal comparison, which improves uncertainty handling but depends on having enough independent positive regions. The H=24 chronological positive windows represent only 20 regions. Intervals may still be unstable under solar-cycle shift or if rare events cluster in a few regions.",
        "<b>Coverage and selection:</b> Long leads require source observations far into the future. At H=48, 120,844 original windows have unknown targets under the stated rule and are excluded. Eligible rows can differ from excluded rows in ways not captured by the reported metrics. The common Phase 3 cohort makes within-study comparisons clearer but does not remove this selection effect. A prospective stream would need explicit handling for incomplete monitoring and event ascertainment.",
        "<b>Model and search scope:</b> Phase 3 uses only XGBoost; it does not establish that XGBoost is best at each lead. The 12-configuration sensitivity study is validation-only and small. The ANN is compact and trained on CPU, and the tested feature summaries are hand engineered. The study includes no probability calibration, no external dataset, no recurrent model trained under the same target, and no independent event-level alert test. Those omissions bound the conclusion rather than invalidate the reported run matrix.",
        "<b>Metric interpretation:</b> AP changes with prevalence and reflects ranking across thresholds, while TSS and F1 depend on the selected threshold. Using validation F1 as a single operating rule is reproducible but not cost optimized. The study does not estimate actual disruption avoided, lead to onset, or real-time availability of all magnetic measurements. Neither high ROC-AUC nor a visually separated trajectory can substitute for those missing evaluations.")

    heading(flow, "XII", "Future Work",
        "The first extension is an onset- and event-based target. It should associate forecasts with unique flare events, specify alert lead to onset, count repeated alerts once, and evaluate missed events and false alerts per unit time. This would address the main semantic limitation of the present peak-based windows. A rolling chronological experiment across solar cycles would better approximate future deployment than a single later-partition test. It should preserve source provenance and avoid selecting alert rules on the final evaluation interval.",
        "A second extension is uncertainty and calibration. The model's probability estimates should be calibrated on training/validation information, then assessed with reliability diagrams and proper scoring rules on independent regions and periods. Alert thresholds can be selected against an explicit cost or utility model. The number of unique positive regions should be reported for every split and lead. Region-level or event-level resampling, repeated temporal holdouts, and sensitivity to the target-coverage rule would show how robust the current conclusions are.",
        "A third extension is controlled model development. Sequence learners, attention architectures, and alternative temporal summaries can be compared under exactly the same lead labels and row IDs. Feature selection must use training folds only. The 24-value baseline should remain in every comparison so added complexity has a transparent reference. Any proposed web application can load the saved models and normalization statistics, but it must recreate feature order from a complete 12-hour observation window, disclose the peak-window target, and avoid presenting retrospective AP as a live reliability guarantee.")

    heading(flow, "XIII", "Conclusion",
        "This project built and verified a SWAN-SF forecasting pipeline with an explicit issue time, fixed 24-hour target interval, and minimum leads of 0, 6, 12, 24, and 48 hours to recorded M/X flare peaks. It saved 33 models and their configurations, held-out rows, probabilities, and metrics. The overall methodology and model branches are documented in Figures 1 and 4. Region and chronological splits give interpretable generalization tests; the random split exposes severe observation overlap and is retained only as a diagnostic.",
        "In the tested unseen-region cohort, baseline XGBoost AP declines from 0.413 at H=6 to 0.259 at H=48. Adding the specified lag and rolling features does not yield a clear AP advantage in any of the eight paired region-bootstrap comparisons. Interpretation plots identify associations within the fitted model but do not establish causal pre-flare signatures. The objectives of building, comparing, and auditing the retrospective models were completed; operational early-warning reliability remains to be established with onset-based, event-level, prospective validation.")

    heading(flow, "APPENDIX A", "Complete Verified Run Matrix",
        "Tables IX-XI preserve the full 33-run result matrix at compact precision. They allow readers to see configurations that were not singled out in the main text, including models whose thresholded and ranking measures move in different directions. These are held-out results, not validation sensitivity scores. 'B' denotes the 24-value baseline and 'B+T' denotes 24 baseline plus 120 temporal values. The source CSV retains additional metrics, counts, row paths, and full precision.")
    phase2_rows = []
    for r in rows:
        if r["phase"] != "phase2":
            continue
        phase2_rows.append([
            {"region_split":"Reg", "time_split":"Time", "random_split":"Rand"}[r["split"]],
            "B+T" if r["features"] == "baseline_temporal" else "B",
            {"logistic":"LR", "forest":"RF", "xgboost":"XGB", "ann":"ANN"}[r["model"]],
            f"{float(r['ap']):.3f}", f"{float(r['precision']):.3f}",
            f"{float(r['recall']):.3f}", f"{float(r['tss']):.3f}",
        ])
    result_table(flow, "TABLE IX. ALL PHASE 2 TEST RUNS (H=0)",
                 ["Split", "Feat.", "Model", "AP", "Prec.", "Rec.", "TSS"],
                 phase2_rows, [39, 34, 40, 31, 35, 34, layout.COL - 7 - 213])
    for split, number in (("region_split", "X"), ("time_split", "XI")):
        matrix = []
        for h in (6, 12, 24, 48):
            for feat, short in (("baseline", "B"), ("baseline_temporal", "B+T")):
                r = metric(rows, "phase3", h, split, feat)
                matrix.append([str(h), short, str(r["positive"]), f"{float(r['ap']):.3f}",
                               f"{float(r['precision']):.3f}", f"{float(r['recall']):.3f}",
                               f"{float(r['tss']):.3f}"])
        result_table(flow, f"TABLE {number}. PHASE 3 {split.replace('_', ' ').upper()} TEST RUNS",
                     ["H", "Feat.", "FL", "AP", "Prec.", "Rec.", "TSS"], matrix,
                     [26, 36, 34, 36, 39, 38, layout.COL - 7 - 209])
    add(flow,
        "The complete row and model artifact contract is stored under data/rebuild/runs. Every run has config.json, metrics.json, status.json, a family-specific model file, and validation/test row and probability arrays. The Phase 1 normalization JSONs and feature_columns.json are necessary for faithful inference. Source-data and model artifacts are gitignored locally, so a collaborator must receive a verified copy; CSV summaries alone are insufficient to reload a model."
    )

    heading(flow, "APPENDIX B", "Reproducibility and Disclosure",
        "From the repository root, forecast_labels.py builds the lead targets; run_rebuild_matrix.py performs the bounded Phase 2-3 training; phase2_sensitivity.py writes validation-only sensitivity results; rebuild_evaluate.py independently verifies all saved runs; and publish_rebuild_reports.py refreshes the report CSV and JSON files. The generator for this manuscript reads those verified artifacts and does not retrain or alter the model set. A changed Phase 1 array, normalization file, or label file invalidates the saved hash checks and requires a new verified model build.",
        "This manuscript was drafted with AI assistance from the team's recorded code and results. The team must review every citation, result interpretation, and individual contribution before submission, and disclose assistance according to instructor policy. The rubric requires separate similarity/plagiarism and AI-generated-content reports. No similarity percentage or AI-detection score is claimed here because those reports were not supplied. Their submission is an administrative step separate from the scientific results in this PDF.")

    heading(flow, "APPENDIX C", "Detailed Audit and Sensitivity Tables",
        "The following tables expose additional denominators and validation settings behind the compact result discussion. They are included because a model score is interpretable only with its sampling protocol and positive count. Table XII uses the original supplied binary labels before lead-specific censoring. It documents the sizes and positive counts of training, validation, and test partitions for each split. These are not the same as the final model test counts after filtering for future target coverage. The distinction prevents the leakage audit denominator of 49,815 random-test windows from being confused with the eligible H=0 diagnostic model test size of 44,980.")
    split_rows = []
    for split in ("random_split", "region_split", "time_split"):
        entry = leakage["splits"][split]
        for part in ("train", "val", "test"):
            split_rows.append([split.replace("_split", "").title(), part.title(),
                               f"{entry['counts'][part]:,}",
                               f"{entry['labels_by_split'][part]['FL']:,}"])
    result_table(flow, "TABLE XII. ORIGINAL WINDOW SPLIT COUNTS",
                 ["Split", "Subset", "Windows", "FL"], split_rows,
                 [68, 52, 68, layout.COL - 7 - 188])
    add(flow,
        "The random split's original test set has 946 FL labels in 49,815 windows. The region test has 939 FL labels in 50,610 windows, while the chronological test has 990 in 75,365. Once H=0 future-coverage eligibility is applied, the region and chronological test sets shrink to 45,757 and 67,279 windows respectively, though their positive counts remain 939 and 990. This is a property of the saved target construction and split row IDs. A report that mixes original and filtered denominators would give inconsistent prevalence and could mislead readers about AP baselines.",
        "Table XIII lists all 12 validation-only sensitivity runs. The family-specific setting is tree depth for XGBoost and hidden-layer width for ANN. Because these results were computed on validation rows, they must not be compared as if they were independent held-out performance. Their main role is to show how much the ranking of selected settings can vary across the two independent split definitions. The production result matrix retained its predefined architecture and should be interpreted separately from this exploratory check.")
    sensitivity_rows = []
    for r in sensitivity["results"]:
        setting = f"depth {r['depth']}" if r["model"] == "xgboost" else f"width {r['units'][0]}/{r['units'][1]}"
        sensitivity_rows.append(["Reg" if r["split"] == "region_split" else "Time",
                                 "XGB" if r["model"] == "xgboost" else "ANN",
                                 setting, f"{r['validation_ap']:.3f}"])
    result_table(flow, "TABLE XIII. VALIDATION-ONLY PARAMETER SENSITIVITY",
                 ["Split", "Family", "Setting", "Val. AP"], sensitivity_rows,
                 [45, 50, 87, layout.COL - 7 - 182])
    add(flow,
        "The sensitivity check also retains the number of training and validation rows and, for XGBoost, the best iteration. These counts differ from the final result table because the exercise is a separate validation experiment. The report does not transfer the best validation setting into a new model and then reuse the same test score. Such reuse would confound an exploratory choice with the locked final analysis. A future nested validation study could make that selection formally and then evaluate once on an untouched outer test set.",
        "For inference and artifact auditing, each run directory contains a family-specific model file together with config.json, metrics.json, status.json, and NumPy arrays of validation/test row IDs and probabilities. The six Phase 1 normalization files contain training-fitted medians and z-score statistics for baseline and temporal matrices under random, region, and chronological splits. The feature order is recorded in both the Phase 1 feature-columns file and each run config. Loading a model without applying the matching preprocessing and feature order would not reproduce any result in this paper. The XGBoost models are stored as native JSON, the ANN as Keras, and the two scikit-learn families as joblib; a copied CSV report is not an inference package.",
        "The saved model validation threshold belongs to its specific model, split, lead, and feature set. It should not be shared across configurations simply because their output is a probability. In particular, a web application must identify which of the several targets it serves: H=0, 6, 12, 24, or 48 hours all refer to peaks in different future intervals. A front end must communicate that interval in plain language and avoid promising onset warning. CPU inference may require configuring a CPU device when loading an XGBoost model trained with CUDA; this does not change the underlying learned trees but must be checked against saved probabilities before deployment.")

    heading(flow, "APPENDIX D", "Confusion Counts and Metric Interpretation",
        "The thresholded measures in the main text can be reconstructed from each run's four confusion counts. Table XIV exposes these counts for the H=24 task under both independent splits and both feature sets. The entries are particularly informative because all four rows use the same nominal minimum lead and the same Phase 3 eligibility rule, while region and chronological rows evaluate distinct populations. Every row's TN + FP + FN + TP equals its listed test count. The selected threshold was obtained on validation data, not by optimizing the test confusion matrix.")
    confusion_rows = []
    for split_name, split_short in (("region_split", "Reg"), ("time_split", "Time")):
        for feat, feat_short in (("baseline", "B"), ("baseline_temporal", "B+T")):
            r = metric(rows, "phase3", 24, split_name, feat)
            confusion_rows.append([split_short, feat_short, r["tp"], r["fp"], r["fn"],
                                   r["tn"], f"{float(r['threshold']):.3f}"])
    result_table(flow, "TABLE XIV. H=24 XGBOOST CONFUSION COUNTS",
                 ["Split", "Feat.", "TP", "FP", "FN", "TN", "Thr."],
                 confusion_rows, [39, 35, 30, 30, 30, 41, layout.COL - 7 - 205])
    add(flow,
        "Precision divides true positive windows by all positive predictions; recall divides them by all genuinely positive windows. A change in threshold can raise recall while lowering precision, which is why a model with a higher AP can still have a less attractive operating point. For the H=24 chronological temporal row, 427/(427 + 838) = 0.338 precision and 427/(427 + 454) = 0.485 recall at its selected threshold. These computations use window counts; they do not tell us how many unique flare events triggered at least one alarm.",
        "TSS subtracts false-positive rate from recall. For the same row, the false-positive rate is 838/(838 + 44,676), approximately 0.0184, giving TSS approximately 0.466. This skill measure handles imbalance differently from raw accuracy, yet it still corresponds to a particular threshold. HSS is another chance-adjusted summary of a two-by-two contingency table. Neither TSS nor HSS encodes the monetary or operational cost of an alert. Their role in this paper is to provide a second view of a model's decisions after the validation rule is fixed.",
        "ROC-AUC is a ranking probability across positive and negative pairs, while AP summarizes the precision-recall curve. With tens of thousands of negative windows, ROC-AUC may remain high even when the number of false alerts at a useful recall level is substantial. AP is more responsive to the rare positive class, but its baseline is the local prevalence rather than a universal zero. Therefore Tables V-VII specify split and lead, and the full CSV retains prevalence with each AP. A report that compares AP without its cohort or event definition can give a false impression of a general improvement.",
        "The repeated-window structure also affects inferential precision. Counting 32,144 region-test windows at H=24 does not mean 32,144 independent test objects: many windows arise from the same active region. The region-bootstrap comparison samples active regions, keeping their overlapping windows together, and thus targets a more appropriate uncertainty unit for the temporal-feature question. Its 500 replicates provide a finite Monte Carlo approximation of an interval. A different region sample, target definition, or bootstrap design could yield different limits. The eight current intervals all include zero, which is a statement about this defined comparison rather than a universal theorem about temporal features.",
        "Finally, the saved forecasts are retrospective probabilities for a 24-hour peak interval after a specified minimum lead. They are not calibrated odds of a flare beginning at an exact time. If an application presents the raw probability, it should name the target window and disclose that probability calibration has not been validated here. If it presents a binary alert, it should name the model-specific validation threshold and avoid using the random diagnostic model as evidence of independent-region performance. These constraints follow directly from the data and evaluation contract documented in this report.")

    flow.extend([layout.NextPageTemplate("Body"), layout.PageBreak(),
                 layout.p("REFERENCES", "SectionI")])
    refs = [
        "[1] R. A. Angryk et al., 'Multivariate time series dataset for space weather data analytics,' Scientific Data, vol. 7, art. 227, 2020, doi: 10.1038/s41597-020-0548-x. Dataset: doi: 10.7910/DVN/EBCFKM.",
        "[2] G. T. Goodwin, V. M. Sadykov, and P. C. Martens, 'Investigating performance trends of simulated real-time solar flare predictions: The impacts of training windows, data volumes, and the solar cycle,' Astrophysical Journal, vol. 964, art. 163, 2024, doi: 10.3847/1538-4357/ad276c.",
        "[3] A. Ji and B. Aydin, 'Active region-based flare forecasting with sliding window multivariate time series forest classifiers,' IEEE CogMI, pp. 196-203, 2023, doi: 10.1109/CogMI58952.2023.00036.",
        "[4] K. Pelkum Donahue and F. Inceoglu, 'Forecasting solar flares with a transformer network,' Frontiers in Astronomy and Space Sciences, vol. 10, art. 1298609, 2024, doi: 10.3389/fspas.2023.1298609.",
        "[5] J. Wen and R. A. Angryk, 'Class-based time series data augmentation to mitigate extreme class imbalance for solar flare prediction,' ICAISC, pp. 362-375, 2024, doi: 10.1007/978-3-031-84353-2_31.",
        "[6] M. EskandariNasab, S. M. Hamdi, and S. Filali Boubrahimi, 'Enhancing multivariate time series-based solar flare prediction with multifaceted preprocessing and contrastive learning,' arXiv:2409.14016, 2024.",
        "[7] Y. Abduallah and J. T. L. Wang, 'A deep learning approach to operational flare forecasting,' arXiv:2405.16080, 2024.",
        "[8] S. Zhang et al., 'A novel solar flare forecast model with deep convolution neural network and one-against-rest approach,' Advances in Space Research, vol. 74, pp. 3467-3480, 2024, doi: 10.1016/j.asr.2024.06.035.",
        "[9] J. Wen et al., 'Outlier detection and removal in multivariate time series for a more robust machine learning-based solar flare prediction,' Astrophysical Journal Supplement Series, vol. 277, art. 60, 2025, doi: 10.3847/1538-4365/adb9e3.",
        "[10] E. Camporeale and T. E. Berger, 'Verification of the NOAA Space Weather Prediction Center solar flare forecast (1998-2024),' Space Weather, vol. 23, e2025SW004546, 2025, doi: 10.1029/2025SW004546.",
        "[11] S. Riggi et al., 'Solar flare forecasting with foundational transformer models across image, video, and time-series modalities,' Astronomy and Computing, vol. 55, 101042, 2026, doi: 10.1016/j.ascom.2025.101042.",
        "[12] A. Azizian Foumani, S. Farokhi, and X. Qi, 'Predicting major solar flares using convolutional neural networks and multivariate magnetic field time-series data,' Solar Physics, vol. 301, art. 23, 2026, doi: 10.1007/s11207-026-02612-6.",
        "[13] D. Alatoom and N. Nikolaou, 'Multi-wavelength transformer-based 24-hour solar flare forecasting at the active-region level,' Space Weather, vol. 24, e2026SW005010, 2026, doi: 10.1029/2026SW005010.",
    ]
    flow.extend(layout.p(escape(ref), "ReferenceI") for ref in refs[:7])
    flow.append(layout.FrameBreak())
    flow.extend(layout.p(escape(ref), "ReferenceI") for ref in refs[7:])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    layout.IEEEReport(OUT).build(flow)
    print(f"Created {OUT}")


if __name__ == "__main__":
    main()
