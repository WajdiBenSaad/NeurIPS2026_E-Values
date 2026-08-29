# E-Value Routing for Multilingual Classification

repository for the code and pinned configurations needed to reproduce
the experiments for our paper **"Discovering Translation-Worthy Languages with E-Values"** for the NeurIPS 2026 workshop **"E-Values: From Statistics to ML"**. 

datasets, model weights, translations, and
run outputs are generated locally and are not distributed.

## 1. Install

requirements: Python 3.11, internet access, and enough disk space for Hugging Face datasets/models.
an available accelerator is used
automatically; full CPU inference is substantially slower.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. Verify the installation

```bash
python scripts/validate_run.py --self-check
python scripts/run_simulations.py \
  --config configs/statistical_protocol.yaml \
  --output reproduced/statistical_selfcheck.json
```

for a short end-to-end check, use `configs/smoke.yaml` with the same discovery
steps shown below. the full configurations impose no per-language row limit.

## 3. SIB-200 discovery

prepare the pinned data and initialize a run:

```bash
python scripts/prepare_data.py --config configs/sib200.yaml
python scripts/init_discovery.py --config configs/sib200.yaml
```

the initializer prints the new run path. copy it exactly:

```bash
SIB_DISCOVERY_RUN="outputs/sib200/discovery/<printed-run-directory>"
```

run languages sequentially. each command uses a fresh process, saves its
checkpoint, and releases all process memory when it exits:

```bash
for language_code in \
  eng_Latn fra_Latn deu_Latn spa_Latn zho_Hans arb_Arab tur_Latn jpn_Jpan \
  pol_Latn nld_Latn swh_Latn ben_Beng tel_Telu amh_Ethi afr_Latn
do
  python scripts/run_discovery_language.py \
    --run-dir "$SIB_DISCOVERY_RUN" \
    --language-code "$language_code" || exit 1
done
```

finalize, validate, and freeze the discovery policy:

```bash
python scripts/finalize_discovery.py --run-dir "$SIB_DISCOVERY_RUN"
python scripts/validate_run.py --run-dir "$SIB_DISCOVERY_RUN"
python scripts/freeze_router.py \
  --run-dir "$SIB_DISCOVERY_RUN" \
  --output reproduced/sib200_frozen_router.json
```

the primary familywise-controlled e-value threshold is 280. threshold 20 is
retained only as the per-group sensitivity analysis.

## 4. SIB-200 held-out evaluation

```bash
python scripts/init_evaluation.py \
  --config configs/sib200.yaml \
  --router reproduced/sib200_frozen_router.json
SIB_EVALUATION_RUN="outputs/sib200/evaluation/<printed-run-directory>"

for language_code in \
  eng_Latn fra_Latn deu_Latn spa_Latn zho_Hans arb_Arab tur_Latn jpn_Jpan \
  pol_Latn nld_Latn swh_Latn ben_Beng tel_Telu amh_Ethi afr_Latn
do
  python scripts/run_evaluation_language.py \
    --run-dir "$SIB_EVALUATION_RUN" \
    --language-code "$language_code" || exit 1
done

python scripts/finalize_evaluation.py --run-dir "$SIB_EVALUATION_RUN"
python scripts/validate_run.py --run-dir "$SIB_EVALUATION_RUN"
```

## 5. MASSIVE discovery

```bash
python scripts/prepare_data.py --config configs/massive.yaml
python scripts/init_discovery.py --config configs/massive.yaml
MASSIVE_DISCOVERY_RUN="outputs/massive/discovery/<printed-run-directory>"

for locale_code in \
  en-US fr-FR de-DE es-ES pt-PT ar-SA tr-TR pl-PL nl-NL ro-RO \
  sw-KE te-IN bn-BD am-ET af-ZA
do
  python scripts/run_discovery_language.py \
    --run-dir "$MASSIVE_DISCOVERY_RUN" \
    --language-code "$locale_code" || exit 1
done

python scripts/finalize_discovery.py --run-dir "$MASSIVE_DISCOVERY_RUN"
python scripts/validate_run.py --run-dir "$MASSIVE_DISCOVERY_RUN"
python scripts/freeze_router.py \
  --run-dir "$MASSIVE_DISCOVERY_RUN" \
  --output reproduced/massive_frozen_router.json
```

## 6. MASSIVE held-out evaluation

```bash
python scripts/init_evaluation.py \
  --config configs/massive.yaml \
  --router reproduced/massive_frozen_router.json
MASSIVE_EVALUATION_RUN="outputs/massive/evaluation/<printed-run-directory>"

for locale_code in \
  en-US fr-FR de-DE es-ES pt-PT ar-SA tr-TR pl-PL nl-NL ro-RO \
  sw-KE te-IN bn-BD am-ET af-ZA
do
  python scripts/run_evaluation_language.py \
    --run-dir "$MASSIVE_EVALUATION_RUN" \
    --language-code "$locale_code" || exit 1
done

python scripts/finalize_evaluation.py --run-dir "$MASSIVE_EVALUATION_RUN"
python scripts/validate_run.py --run-dir "$MASSIVE_EVALUATION_RUN"
```

## 7. Generate the reported analyses

once all four runs are complete:

```bash
python scripts/build_paper_analysis.py \
  --sib-discovery-run "$SIB_DISCOVERY_RUN" \
  --sib-evaluation-run "$SIB_EVALUATION_RUN" \
  --massive-discovery-run "$MASSIVE_DISCOVERY_RUN" \
  --massive-evaluation-run "$MASSIVE_EVALUATION_RUN" \
  --reports-root reproduced/results \
  --paper-root reproduced/paper \
  --bootstrap-replicates 2000 \
  --order-repetitions 50

python scripts/build_eprocess_metrics.py \
  --config configs/eprocess_metrics.yaml \
  --sib-discovery-run "$SIB_DISCOVERY_RUN" \
  --massive-discovery-run "$MASSIVE_DISCOVERY_RUN" \
  --output-dir reproduced/eprocess_metrics
```

this generates the reported tables and figures, paired bootstrap intervals,
50 total outcome-independent orderings, threshold sensitivity, Type-I-error,
power, and stopping-time results under `reproduced/`.

## 8. Outputs, resumption, and checks

runs are written to
`outputs/<dataset>/<stage>/run_<timestamp>_seed-<seed>_<id>/`. per-language
progress is saved in `language_checkpoints/`. after interruption, rerun only
unfinished language commands; finalization requires every configured checkpoint.

```bash
python scripts/validate_run.py --self-check
python scripts/validate_run.py --run-dir <RUN_DIR>
python scripts/run_latency.py --run-dir <EVALUATION_RUN>
```

- rerun the same language command after an interrupted model download.
- do not change discovery thresholds after inspecting evaluation data.
- exact dataset/model revisions and all protocol parameters are pinned in
  `configs/`.
