"""Create a two-column IEEE-style report using the verified SWAN-SF results."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from generate_report_visuals import build_visuals


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "final" / "Solar_Flare_Forecasting_IEEE_Report.pdf"
NAVY = colors.HexColor("#111111")
RULE = colors.HexColor("#333333")
L = 45
R = 45
GAP = 18
WIDTH = letter[0] - L - R
COL = (WIDTH - GAP) / 2


def fonts():
    src = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Times", str(src / "times.ttf")))
    pdfmetrics.registerFont(TTFont("Times-Bold", str(src / "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("Times-Italic", str(src / "timesi.ttf")))
    pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold",
                                  italic="Times-Italic", boldItalic="Times-Bold")


fonts()
SS = getSampleStyleSheet()
for kw in [
    dict(name="TitleI", fontName="Times-Bold", fontSize=20, leading=22,
         alignment=TA_CENTER, spaceAfter=7),
    dict(name="AuthorI", fontName="Times", fontSize=10, leading=12,
         alignment=TA_CENTER, spaceAfter=8),
    dict(name="AbstractI", fontName="Times", fontSize=9, leading=11,
         alignment=TA_JUSTIFY, spaceAfter=5),
    dict(name="KeywordI", fontName="Times", fontSize=8.8, leading=11,
         spaceAfter=3),
    dict(name="SectionI", fontName="Times-Bold", fontSize=10, leading=12,
         alignment=TA_CENTER, spaceBefore=9, spaceAfter=4, keepWithNext=True),
    dict(name="SubsectionI", fontName="Times-Italic", fontSize=9.5, leading=11.5,
         spaceBefore=7, spaceAfter=3, keepWithNext=True),
    dict(name="BodyI", fontName="Times", fontSize=10, leading=12,
         alignment=TA_JUSTIFY, spaceAfter=4.5),
    dict(name="SmallI", fontName="Times", fontSize=8.5, leading=10.1,
         alignment=TA_JUSTIFY, spaceAfter=4),
    dict(name="TableI", fontName="Times", fontSize=7.8, leading=9.3),
    dict(name="TableHeadI", fontName="Times-Bold", fontSize=7.8, leading=9.3),
    dict(name="CaptionI", fontName="Times", fontSize=8, leading=9.5,
         alignment=TA_CENTER, spaceBefore=2, spaceAfter=7),
    dict(name="ReferenceI", fontName="Times", fontSize=8, leading=9.7,
         leftIndent=14, firstLineIndent=-14, spaceAfter=3),
]:
    SS.add(ParagraphStyle(**kw))


class IEEEReport(BaseDocTemplate):
    def __init__(self, path: Path):
        super().__init__(str(path), pagesize=letter, leftMargin=L, rightMargin=R,
                         topMargin=45, bottomMargin=52,
                         title="Solar Flare Forecasting: Leakage-Controlled Lead-Time Evaluation",
                         author="Harshit Dharmesh Pandya; Prasiddhi Singhal; Mohammed Ahsan Bagsaria")
        # The extended abstract needs the full title frame.  Keep the first
        # body columns below it so the abstract cannot spill into only the
        # left column and make FrameBreak skip to the right column.
        title_frame = Frame(L, 511, WIDTH, 236, leftPadding=0, rightPadding=0,
                            topPadding=0, bottomPadding=0, id="title")
        first_left = Frame(L, 52, COL, 451, leftPadding=0, rightPadding=7,
                           topPadding=0, bottomPadding=0, id="first_left")
        first_right = Frame(L + COL + GAP, 52, COL, 451, leftPadding=7,
                            rightPadding=0, topPadding=0, bottomPadding=0,
                            id="first_right")
        body_left = Frame(L, 52, COL, 695, leftPadding=0, rightPadding=7,
                          topPadding=0, bottomPadding=0, id="left")
        body_right = Frame(L + COL + GAP, 52, COL, 695, leftPadding=7,
                           rightPadding=0, topPadding=0, bottomPadding=0,
                           id="right")
        fig_top = Frame(L, 367, WIDTH, 380, leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0, id="figure_top")
        fig_left = Frame(L, 52, COL, 307, leftPadding=0, rightPadding=7,
                         topPadding=0, bottomPadding=0, id="figure_left")
        fig_right = Frame(L + COL + GAP, 52, COL, 307, leftPadding=7,
                          rightPadding=0, topPadding=0, bottomPadding=0,
                          id="figure_right")
        self.addPageTemplates([
            PageTemplate(id="First", frames=[title_frame, first_left, first_right],
                         onPage=self.footer, autoNextPageTemplate="Body"),
            PageTemplate(id="Body", frames=[body_left, body_right], onPage=self.footer),
            PageTemplate(id="Figure", frames=[fig_top, fig_left, fig_right],
                         onPage=self.footer, autoNextPageTemplate="Body"),
        ])

    @staticmethod
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(.4)
        canvas.line(L, 40, letter[0] - R, 40)
        canvas.setFont("Times", 7.5)
        canvas.drawCentredString(letter[0] / 2, 29, str(doc.page))
        canvas.restoreState()


def p(text, style="BodyI"):
    for original, replacement in {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-", "\u0394": "Delta ",
    }.items():
        text = text.replace(original, replacement)
    return Paragraph(text, SS[style])


def sec(number, title):
    return p(f"{number}. {escape(title.upper())}", "SectionI")


def sub(title):
    return p(escape(title), "SubsectionI")


def table(headers, data, widths):
    if abs(sum(widths) - (COL - 7)) > .5:
        raise ValueError("IEEE table must fit one text column")
    cells = [[p(escape(str(x)), "TableHeadI") for x in headers]]
    cells += [[p(escape(str(x)), "TableI") for x in row] for row in data]
    out = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    out.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), .6, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), .5, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), .6, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return out


def fig(path, caption, width=COL - 12, max_height=190):
    im = Image(str(ROOT / path))
    scale = min(width / im.imageWidth, max_height / im.imageHeight)
    im.drawWidth = im.imageWidth * scale
    im.drawHeight = im.imageHeight * scale
    im.hAlign = "CENTER"
    return KeepTogether([im, p(caption, "CaptionI")])


def wide_visual(path, caption):
    im = Image(str(ROOT / path))
    scale = min((WIDTH - 2) / im.imageWidth, 350 / im.imageHeight)
    im.drawWidth = im.imageWidth * scale
    im.drawHeight = im.imageHeight * scale
    im.hAlign = "CENTER"
    return [NextPageTemplate("Figure"), PageBreak(),
            im, p(caption, "CaptionI"), FrameBreak()]


def metric(rows, phase, lead, split, features, model="xgboost"):
    return next(r for r in rows if r["phase"] == phase and int(r["lead_hours"]) == lead
                and r["split"] == split and r["features"] == features and r["model"] == model)


def main():
    build_visuals()
    results = list(csv.DictReader((ROOT / "reports/phase4/phase4_results.csv").open(newline="", encoding="utf-8")))
    audit = json.loads((ROOT / "reports/phase4/phase4_results.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "data/phase2/horizon_targets_report.json").read_text(encoding="utf-8"))
    leakage = json.loads((ROOT / "data/leakage_summary.json").read_text(encoding="utf-8"))
    if len(results) != 33 or audit["verified_run_count"] != 33 or audit["warnings"]:
        raise ValueError("Report requires 33 verified runs and zero warnings")
    if labels["supplied_24h_label_comparison_on_valid_rows"]["mismatched"] != 0:
        raise ValueError("H=0 label comparison failed")
    flow = [
        p("Investigating the Early-Warning Capability and Generalization of Machine Learning Models for Solar Flare Forecasting", "TitleI"),
        p("Harshit Dharmesh Pandya (240968444), Prasiddhi Singhal (240968448), Mohammed Ahsan Bagsaria (240968484)<br/>Department of Data Science and Engineering, Manipal Institute of Technology, Manipal", "AuthorI"),
        p("<b>Abstract—</b> We evaluate M/X solar-flare forecasting on SWAN-SF with explicit minimum leads to the recorded flare peak. A 12-hour magnetic observation window predicts a peak in a fixed 24-hour validity interval after H = 0, 6, 12, 24 or 48 hours. Phase 2 compares four model families; Phase 3 compares baseline and temporal features across the four positive leads. Region and chronological splits assess generalization, while a random split diagnoses overlapping-window leakage. On unseen regions, baseline XGBoost average precision (AP) is 0.413 at 6 hours and 0.259 at 48 hours. A paired region bootstrap finds no reliable AP gain from the temporal summaries. All 33 trained runs pass artifact and metric verification. These results measure lead to peak, not flare onset, and remain retrospective.", "AbstractI"),
        p("<b>Index Terms—</b> solar flares, SWAN-SF, early warning, temporal leakage, XGBoost, active regions.", "KeywordI"),
        FrameBreak(),
        sec("I", "Introduction"),
        p("Solar flares arise from magnetic-energy release and may affect space infrastructure. The SWAN-SF benchmark contains photospheric magnetic time series linked to flare reports [1]. Overlapping 12-hour samples create a risk that random train-test assignment evaluates nearly repeated observations. The project therefore asks how forecasting skill changes with earlier minimum warning and whether split strategy alters the credibility of measured performance."),
        p("Our task is defined by the MFLARE/XFLARE peak annotation: a positive H-hour target has a peak in (issue + H, issue + H + 24 h]. This is a minimum lead to peak with a fixed 24-hour validity window. It cannot be interpreted as H hours before onset."),
        sec("II", "Related Work and Research Gap"),
        p("Prior SWAN-SF studies address solar-cycle training, temporal subintervals, imbalance and outliers [2], [3], [5], [6], [9]. Transformer and CNN studies use differing sensors, target horizons or evaluation units [4], [7], [8], [11]–[13]. Operational verification motivates transparent baselines [10]. Table I compares how these studies relate to our narrower split-and-lead analysis. Their published scores are not directly ranked against ours because label definitions, prevalence and test protocols differ."),
        p("TABLE I. COMPARATIVE LITERATURE SUMMARY", "CaptionI"),
        table(["Study", "Method and comparison point"], [
            ["Goodwin et al. [2]", "SWAN-SF simulated real-time training; cycle and window effects."],
            ["Ji and Aydin [3]", "Interval forest; interpretable temporal subwindows."],
            ["Pelkum Donahue et al. [4]", "Transformer on SHARP series; 24-hour occurrence target."],
            ["Wen and Angryk [5]", "Sequence augmentation for rare SWAN-SF flares."],
            ["EskandariNasab et al. [6]", "Preprocessing and contrastive/GRU classification."],
            ["Abduallah and Wang [7]", "Transformer with multiple operational horizons."],
            ["Zhang et al. [8]", "Multiclass one-against-rest CNN; different target."],
            ["Wen et al. [9]", "Outlier filtering affects SWAN-SF forecasts."],
            ["Camporeale and Berger [10]", "Retrospective operational forecast verification."],
            ["Riggi et al. [11]", "Image/video/irradiance transformer comparison."],
            ["Azizian Foumani et al. [12]", "SWAN-SF CNN with a 12-hour prediction task."],
            ["Alatoom and Nikolaou [13]", "Multi-wavelength transformer; AR-level test."],
        ], [80, COL - 7 - 80]),
        p("<b>Gap and objectives:</b> Assess a common, explicit lead-time definition with group-aware and chronological evaluation; compare standard and temporal features on the same held-out windows; and report uncertainty rather than interpreting an apparent score change as a universal improvement."),
    ]

    flow += wide_visual("reports/final/pipeline.png",
                        "Fig. 1. Implemented data path. Flare annotations enter targets only; normalization uses training rows. Source: project code and saved artifacts.")
    flow += [
        sec("III", "Dataset and Methods"),
        p("The local seven-archive SWAN-SF extraction yielded 4,098 full region series, 331,185 pre-sliced 12-hour windows and 3,205 distinct region IDs. The original label is positive in 6,234 windows (1.88%). Phase 1 generated 24 baseline magnetic values and 120 temporal summaries per window. The latter include lags, rates of change and rolling mean/standard deviation. Predictor arrays exclude flare, X-ray and quality columns; missing values are imputed by training-only medians and scaled by training-only z-scores."),
        p("Peak-time targets are constructed from the full region series, requiring the source to cover the entire future interval and at least 114 of 120 expected 12-minute samples. All 299,175 eligible H=0 targets agreed with the supplied label. The four Phase 3 leads share one cohort of 210,341 windows. Near the end of a series, missing future coverage is excluded, not labeled negative."),
        p("The region split groups all windows of each active region. The chronological split trains on partitions 1–3, validates on 4, and tests on 5. A random split is included only as a leakage diagnostic. Phase 1 measured median 91.5% overlap between adjacent same-region windows; 49,758 of 49,815 original random-test windows overlap training observation intervals. The region and time splits have no such train-test observation overlap in that audit."),
    ]
    flow += [
        p("For each run, a decision threshold maximizes validation F1. AP is the primary rare-event ranking measure; ROC-AUC, precision, recall, F1, TSS and HSS give complementary views. Temporal-minus-baseline AP is paired on identical test windows and bootstrapped 500 times by active region, keeping overlapping windows together."),
        sec("IV", "Implementation"),
        p("Phase 2 trains Logistic Regression, an 80-tree Random Forest, XGBoost (depth 3, at most 180 trees, 64 histogram bins and validation PR-AUC early stopping), and a 32/16-unit ANN. Each family uses both feature sets and both independent splits; an additional random-split XGBoost run is diagnostic. Phase 3 trains XGBoost at H = 6, 12, 24 and 48 hours across two splits and two feature sets. The 33 training runs execute sequentially. XGBoost ran on an RTX 4050; the installed native Windows TensorFlow ran the ANN on CPU."),
        p("A separate 12-configuration validation-only sensitivity check varied XGBoost depth and ANN width. It did not replace the saved test matrix. The evaluator reloaded each model, checked current data hashes and held-out row identities, recomputed all metrics, and checked 64 held-out probabilities per run. All 33 passed, with zero rejected."),
    ]
    flow += wide_visual("reports/final/architecture.png",
                        "Fig. 2. Actual model families and experiment counts. XGBoost used bounded CUDA training; the ANN trained on CPU. Source: project training plan and saved runs.")
    flow += [
        sec("V", "Experimental Results"),
        p("TABLE II. ORIGINAL H=0 BASELINE TEST PERFORMANCE", "CaptionI"),
    ]
    models = [("Logistic", "logistic"), ("Forest", "forest"),
              ("XGBoost", "xgboost"), ("ANN", "ann")]
    mdata = []
    for display, name in models:
        region = metric(results, "phase2", 0, "region_split", "baseline", name)
        time = metric(results, "phase2", 0, "time_split", "baseline", name)
        mdata.append([display, f"{float(region['ap']):.3f}", f"{float(time['ap']):.3f}",
                      f"{float(time['tss']):.3f}"])
    flow += [
        table(["Model", "Region AP", "Time AP", "Time TSS"], mdata,
              [70, 54, 54, COL - 7 - 178]),
        p("Baseline XGBoost AP is 0.421 for unseen regions and 0.455 for later partitions. Its random-split diagnostic AP is 0.551, consistent with measured overlap. The split test sets differ, so this is not a paired causal estimate of leakage inflation. Region-test ANN AP is 0.444; time-test logistic AP is 0.458. Model ranking varies by metric and split."),
        p("TABLE III. XGBOOST BASELINE RESULTS BY MINIMUM LEAD", "CaptionI"),
    ]
    hdata = []
    for h in (6, 12, 24, 48):
        region = metric(results, "phase3", h, "region_split", "baseline")
        time = metric(results, "phase3", h, "time_split", "baseline")
        hdata.append([str(h), f"{float(region['ap']):.3f}", f"{float(time['ap']):.3f}",
                      f"{float(region['tss']):.3f}", f"{float(time['tss']):.3f}"])
    flow += [
        table(["H (h)", "Reg. AP", "Time AP", "Reg. TSS", "Time TSS"], hdata,
              [36, 51, 52, 53, COL - 7 - 192]),
        p("Unseen-region baseline AP declines from 0.413 at 6 hours to 0.259 at 48 hours. Time-test AP is not monotone (0.409, 0.367, 0.347, 0.359). Although the Phase 3 eligible rows are common, each lead changes the label and prevalence. The curves describe separate forecast tasks, not one controlled event-level warning experiment."),
    ]
    flow += [
        fig("reports/rebuild/phase3/ap_by_lead.png", "Fig. 3. Held-out AP by lead. Source: verified project results.", max_height=145),
        sub("A. Temporal features and uncertainty"),
        p("TABLE IV. PAIRED TEMPORAL-MINUS-BASELINE AP", "CaptionI"),
    ]
    comps = sorted(audit["paired_comparisons"], key=lambda x: (x["lead_hours"], x["split"]))
    flow += [
        table(["H", "Split", "ΔAP", "95% region CI"], [
            [str(x["lead_hours"]), "Region" if x["split"] == "region_split" else "Time",
             f"{x['difference_ap']:+.3f}",
             f"[{x['ci95_region_bootstrap'][0]:+.3f}, {x['ci95_region_bootstrap'][1]:+.3f}]"]
            for x in comps
        ], [24, 48, 47, COL - 7 - 119]),
        p("All eight 95% intervals include zero. These experiments therefore do not establish a reliable AP gain from adding the 120 temporal features. For the H=24 chronological temporal model, AP is 0.369, TSS is 0.466, precision is 0.338, with 427 true positives and 838 false positives. A high ranking score does not remove the false-alarm burden."),
        fig("reports/rebuild/phase3/tss_by_lead.png", "Fig. 4. Test TSS at validation-selected thresholds. Source: verified project results.", max_height=145),
        sub("B. Model interpretation"),
        p("For the H=24 chronological temporal XGBoost model, TreeSHAP emphasizes rolling means and lags of R_VALUE and TOTUSJH and the rolling variation of ABSNJZH. One observation window per region was sampled for the trajectory plot. The 881 positive H=24 time-test windows arise from only 20 positive regions. The plots show model associations, not causal flare precursors."),
        fig("reports/rebuild/phase4/feature_importance.png", "Fig. 5. XGBoost feature associations. Source: verified project analysis.", max_height=170),
        fig("reports/rebuild/phase4/preflare_patterns.png", "Fig. 6. Sampled observation-window trajectories. Source: verified project analysis.", max_height=170),
        sec("VI", "Discussion and Limitations"),
        p("The 0.551 random-test AP should not be cited as independent-region performance: nearly every random-test observation interval overlaps training. Grouped and chronological tests answer different generalization questions. At longer leads, censoring is substantial: 120,844 original windows lack a valid H=48 target. The retained cohort may differ from excluded observations. AP also changes with prevalence, so cross-lead comparisons require caution."),
        p("Our label measures lead to the recorded peak, not onset. Potential annotation errors, repeated windows, class imbalance, fixed model settings and only 20 positive H=24 chronological test regions further constrain interpretation. The study is retrospective; prospective external and event-level validation are needed before operational use [10]."),
        sec("VII", "Conclusion and Future Work"),
        p("A common SWAN-SF pipeline with explicit peak-time labels produced 33 verified model runs. Baseline region-test XGBoost AP declines from 0.413 at 6 hours to 0.259 at 48 hours; temporal summaries show no statistically clear AP advantage in the paired regional bootstrap. Future work should evaluate onset-based event alerts, calibration and alert costs, newer cycles, region-level aggregation and sequence models under identical labels and splits."),
        sec("VIII", "Reproducibility and Disclosure"),
        p("Targets, Phase 1 arrays, row IDs, saved probabilities and model binaries remain in the local shared data directory. CSV/JSON results and this paper are portable, but another computer requires a verified artifact copy to reload the models. This manuscript was drafted with AI assistance from recorded project results; the team must review it and disclose assistance as required. The rubric requests separate similarity and AI-content reports; no such score is claimed here.", "SmallI"),
        sec("IX", "References"),
    ]
    references = [
        "[1] R. A. Angryk et al., “Multivariate time series dataset for space weather data analytics,” Scientific Data, vol. 7, art. 227, 2020, doi: 10.1038/s41597-020-0548-x. Dataset: doi: 10.7910/DVN/EBCFKM.",
        "[2] G. T. Goodwin, V. M. Sadykov, and P. C. Martens, “Investigating performance trends of simulated real-time solar flare predictions: The impacts of training windows, data volumes, and the solar cycle,” Astrophysical Journal, 2024, doi: 10.3847/1538-4357/ad276c.",
        "[3] A. Ji and B. Aydin, “Active region-based flare forecasting with sliding window multivariate time series forest classifiers,” IEEE CogMI, pp. 196–203, 2023, doi: 10.1109/CogMI58952.2023.00036.",
        "[4] K. Pelkum Donahue and F. Inceoglu, “Forecasting solar flares with a transformer network,” Frontiers in Astronomy and Space Sciences, vol. 10, 1298609, 2024, doi: 10.3389/fspas.2023.1298609.",
        "[5] J. Wen and R. A. Angryk, “Class-based time series data augmentation to mitigate extreme class imbalance for solar flare prediction,” ICAISC, pp. 362–375, 2024, doi: 10.1007/978-3-031-84353-2_31.",
        "[6] M. EskandariNasab, S. M. Hamdi, and S. Filali Boubrahimi, “Enhancing multivariate time series-based solar flare prediction with multifaceted preprocessing and contrastive learning,” arXiv:2409.14016, 2024.",
        "[7] Y. Abduallah and J. T. L. Wang, “A deep learning approach to operational flare forecasting,” arXiv:2405.16080, 2024.",
        "[8] S. Zhang et al., “A novel solar flare forecast model with deep convolution neural network and one-against-rest approach,” Advances in Space Research, vol. 74, pp. 3467–3480, 2024, doi: 10.1016/j.asr.2024.06.035.",
        "[9] J. Wen et al., “Outlier detection and removal in multivariate time series for a more robust machine learning-based solar flare prediction,” Astrophysical Journal Supplement Series, vol. 277, art. 60, 2025, doi: 10.3847/1538-4365/adb9e3.",
        "[10] E. Camporeale and T. E. Berger, “Verification of the NOAA Space Weather Prediction Center solar flare forecast (1998–2024),” Space Weather, vol. 23, e2025SW004546, 2025, doi: 10.1029/2025SW004546.",
        "[11] S. Riggi et al., “Solar flare forecasting with foundational transformer models across image, video, and time-series modalities,” Astronomy and Computing, vol. 55, 101042, 2026, doi: 10.1016/j.ascom.2025.101042.",
        "[12] A. Azizian Foumani, S. Farokhi, and X. Qi, “Predicting major solar flares using convolutional neural networks and multivariate magnetic field time-series data,” Solar Physics, vol. 301, art. 23, 2026, doi: 10.1007/s11207-026-02612-6.",
        "[13] D. Alatoom and N. Nikolaou, “Multi-wavelength transformer-based 24-hour solar flare forecasting at the active-region level,” Space Weather, vol. 24, e2026SW005010, 2026, doi: 10.1029/2026SW005010.",
    ]
    flow += [p(escape(item), "ReferenceI") for item in references]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    IEEEReport(OUT).build(flow)
    print(f"Created {OUT}")


if __name__ == "__main__":
    main()
