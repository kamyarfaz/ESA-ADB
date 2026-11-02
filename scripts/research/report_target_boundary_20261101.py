"""Publish every predeclared arm and outcome after independent verification."""
import json
import shutil
import pandas as pd
from scripts.research.target_boundary_common import ROOT, OUTPUT, TABLE

NAMES = {'expanded_target':'Expanded inputs, target40–47 score',
         'masked_all':'Masked pretraining + fine-tuning',
         'expanded_all':'Expanded inputs, all11-output score',
         'reference_short_no_roll':'Original short-no-roll reference',
         'reference_pretrain_finetune':'Masked-pretrained unchanged reference',
         'reference_expanded_short_no_roll':'Expanded unchanged reference'}


def main():
    verification = json.loads((TABLE / 'verification.json').read_text())
    assert verification['verified']
    df = pd.read_csv(TABLE / 'verified_results.csv', dtype={'period':str})
    events = pd.read_csv(TABLE / 'event_ids.csv', dtype={'period':str})
    selection = json.loads((OUTPUT / 'selection_complete.json').read_text())
    lines = ['# Target-output scoring, alarm boundaries and post-processing — 1 November 2026', '',
             'This study changes score pooling and alarm processing using two fixed high-recall checkpoints. '
             'Original short-no-roll remains an unchanged reference. The three candidate arms are not three new trained models. '
             'All model training used2000–2005, calibration uses2006, and the reused assessment is2007–2013. '
             'No new training, new seeds or assessment-selected rules.', '',
             'Expanded all-output boundary trimming produced only a tiny F0.5 gain, with all detections retained. '
             'Masked-model trimming lost one target event and reduced F0.5. Target-only scoring substantially '
             'reduced F0.5 because it fragmented alarms. None of the four additional processing families selected '
             'a gain beyond its boundary fallback. All outcomes and unchanged controls are reported below.', '',
             '## Mechanisms and selection', '',
             'Expanded target-only scoring keeps all11 inputs (40–47,14,21,29) but permits only reconstruction errors from40–47 '
             'to enter the alarm score. The other expanded arm keeps all11 output errors, isolating boundary effects. '
             'Masked pretraining/fine-tuning uses eight inputs40–47 and ordinary unmasked reconstruction at inference.', '',
             'The unchanged detector proposes components using its128-minute window-mean score. Finer evidence averages '
             'squared residuals at their actual timestamps. Boundary localization crops each component to the first and last '
             'fine-score crossing, preserving the entire span between them. If there is no crossing, the component is retained. '
             'No true event boundaries enter this transformation. Fine evidence remains retrospective.', '',
             'All thresholds and decisions were selected on2006 and jointly frozen before new assessment inference. '
             'The objective was corrected local40–47 eventF0.5, retaining all original calibration-detected local IDs and '
             'at most1% total alarm fraction. Each of four post-processing families was separately applied before the '
             'already selected boundary rule: temporal merging/minimum duration, strength-duration, hysteresis and duration. '
             'Every family has an unchanged-boundary fallback. No assessment winner is substituted for the frozen choices.', '',
             '## Calibration choices', '',
             '| Arm | Base F0.5 | Boundary F0.5 | Nominal alarm seconds before / after | Boundary cutoff |',
             '|---|---:|---:|---:|---:|']
    summary = []
    for arm, spec in selection.items():
        b, c = spec['base']['local'], spec['boundary']['local']
        lines.append(f'| {NAMES[arm]} | {b["EW_F_0.50"]:.9f} | {c["EW_F_0.50"]:.9f} | '
                     f'{b["false_positive_seconds"]:.3f} / {c["false_positive_seconds"]:.3f} | '
                     f'{spec["boundary"]["rule"].get("cutoff", "unchanged")} |')
        for family, choice in spec['choices'].items():
            grid = json.loads((OUTPUT / arm / (family + '_candidates.json')).read_text())
            summary.append(dict(arm=arm, family=family, candidates=len(grid),
                                unique_masks=spec['unique_masks'][family], feasible=sum(r['feasible'] for r in grid),
                                selected_original_boundary=choice['rule']['original'],
                                selected_local_f05=choice['local']['EW_F_0.50'], rule=json.dumps(choice['rule'])))
    lines += ['', f'Independent verification replayed {verification["candidate_rows"]} calibration candidate rows '
              '(387 boundary/family entries plus the target-threshold grid). These are settings, not independent experiments. '
              'Duplicate prediction masks and all rejected candidates are retained in the compact evidence.', '',
              '| Arm | Family | Settings | Distinct masks | Feasible | Retained boundary fallback |',
              '|---|---|---:|---:|---:|---|']
    for r in summary:
        lines.append(f'| {NAMES[r["arm"]]} | {r["family"]} | {r["candidates"]} | {r["unique_masks"]} | '
                     f'{r["feasible"]} | {r["selected_original_boundary"]} |')
    lines += ['', '## Pooled comparison and measured changes', '',
              'TPR is event-ID recall; timeFPR is nominal false-alarm duration divided by nominal duration. '
              'Corrected precision includes the ESA nominal-duration penalty. Scores below are measurements on the '
              'reused benchmark, not a robust generalization or deployment claim.', '',
              '| Arm | Variant | Scope | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    pooled = df[df.period == '2007-2013']
    for _, r in pooled.iterrows():
        lines.append(f'| {NAMES[r.arm]} | {r.variant} | {r.scope} | {int(r.TPe)}/{int(r.num_true_events)} | '
                     f'{r.EW_precision:.6f} | {r.EW_recall:.6f} | {100*r.nominal_false_alarm_rate:.6f} | {r["EW_F_0.50"]:.9f} |')
    for arm, spec in selection.items():
        lines += ['', f'### {NAMES[arm]}', '']
        for scope in ['system_all_channels', 'input_channels_40_47']:
            base = pooled[(pooled.arm == 'reference_' + spec['method']) & (pooled.scope == scope)].iloc[0]
            for variant in ['base', 'boundary']:
                row = pooled[(pooled.arm == arm) & (pooled.variant == variant) & (pooled.scope == scope)].iloc[0]
                lines.append(f'- {scope}, {variant}: F0.5 {base["EW_F_0.50"]:.9f} → {row["EW_F_0.50"]:.9f}; '
                             f'detected IDs {int(base.TPe)} → {int(row.TPe)}; timeFPR '
                             f'{100*base.nominal_false_alarm_rate:.6f}% → {100*row.nominal_false_alarm_rate:.6f}%.')
        for family, choice in spec['choices'].items():
            lines.append(f'- {family}: ' + ('retained the selected boundary prediction.' if choice['rule']['original'] else
                         'selected a distinct calibration rule; see its full outcome above.'))
    lines += ['', '## Interpretation of the measured outcomes', '',
              'Expanded all-output boundary refinement increased system F0.5 from0.844573267 to0.844609096 '
              '(+0.000035829) and target40–47 F0.5 from0.830899674 to0.830935471 (+0.000035797). '
              'It retained80/91 system and62/67 target detections with no gained or lost IDs. Nominal alarm duration '
              'fell by9704.730 seconds, about2.70 hours across seven years, or0.0836% relative to its original '
              'false-alarm duration. This is a tiny measured gain, not evidence of a practically substantial or robust improvement.', '',
              'Masked-model boundary refinement reduced nominal alarm duration but lost id_40 in both scopes: '
              '61→60 system detections and60→59 target detections. False event count increased12→14 even though '
              'the transform preserves the number of predicted components: trimming can remove the overlap that '
              'previously connected a component to a real annotation. Its F0.5 decreased in both scopes. '
              'The original result remains visible; this unsuccessful frozen outcome is not replaced using test scores.', '',
              'Target-only scoring preserved62/67 target detections but lost system IDs id_188 and id_189. '
              'The selected numeric threshold was unchanged. Its mask was a strict subset of the original expanded '
              'mask, yet predicted components increased95→407 and false events10→241. A retrospective mask '
              'comparison shows one original component split into313 target components; no original component '
              'was entirely removed. Removing auxiliary-output evidence therefore reduced duration while fragmenting '
              'an extended alarm into many false events. This explains the large event-precision/F0.5 loss despite '
              'a lower timeFPR. Boundary trimming changes only outer edges and cannot repair this fragmentation. '
              'These diagnostics explain fixed results; they did not change selection or create new candidates.', '',
              'All four post-processing families retained the selected boundary fallback in every arm. They provided '
              'no additional selected gain within the declared calibration grids. The repeated family rows are '
              'identical predictions, not independent replications. Original short-no-roll still has the highest '
              'target-scope pooled F0.5 here (0.907139530), while detecting47/67 target events. There is no single '
              'method that simultaneously maximizes recall and precision in this comparison.', '',
              '[Fragmentation diagnostic](tables/target-boundary-2026-11-01/fragmentation_diagnostics.json).', '',
              '## Descriptive calibration stability', '',
              'The table reports the frozen base and boundary choices on2006 halves and quarters. '
              'These blocks were already used for model/rule selection and are not independent validation. '
              'Small denominators and uneven event coverage make near-ceiling full-year calibration scores weak '
              'evidence of generalization. All252 block/scope/variant rows are preserved in '
              '[the complete stability table](tables/target-boundary-2026-11-01/run-records/calibration_stability.csv).', '',
              '| Arm | Block | Base detected/total | Boundary detected/total | Base F0.5 | Boundary F0.5 |',
              '|---|---|---:|---:|---:|---:|']
    stability = pd.read_csv(OUTPUT / 'calibration_stability.csv')
    for arm in selection:
        for block in ['H1', 'H2', 'Q1', 'Q2', 'Q3', 'Q4']:
            part = stability[(stability.arm == arm) & (stability.block == block) &
                             (stability.scope == 'input_channels_40_47')]
            b = part[part.variant == 'base'].iloc[0]
            c = part[part.variant == 'boundary'].iloc[0]
            lines.append(f'| {NAMES[arm]} | {block} | {int(b.TPe)}/{int(b.num_true_events)} | '
                         f'{int(c.TPe)}/{int(c.num_true_events)} | {b["EW_F_0.50"]:.9f} | {c["EW_F_0.50"]:.9f} |')
    lines += ['', '## Annual and pooled metrics', '',
              'Predictions are continuous across calendar boundaries; annual slices are reporting only. '
              'Pooled metrics are recomputed over the complete period, not averaged from annual ratios.']
    for arm in df.arm.drop_duplicates():
        for scope in ['system_all_channels', 'input_channels_40_47']:
            lines += ['', f'### {NAMES[arm]} — {scope}', '',
                      '| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |',
                      '|---|---|---:|---:|---:|---:|---:|']
            part = df[(df.arm == arm) & (df.scope == scope)]
            for period in [str(y) for y in range(2007,2014)] + ['2007-2013']:
                for _, r in part[part.period == period].iterrows():
                    lines.append(f'| {period} | {r.variant} | {int(r.TPe)}/{int(r.num_true_events)} | '
                                 f'{r.EW_precision:.6f} | {r.EW_recall:.6f} | {100*r.nominal_false_alarm_rate:.6f} | {r["EW_F_0.50"]:.9f} |')
    lines += ['', '## Event changes', '']
    for (arm, variant, scope), group in events[(events.period == '2007-2013') & ~events.arm.str.startswith('reference_')].groupby(['arm','variant','scope']):
        lines.append(f'- {arm}/{variant}/{scope}: gained ' + (', '.join(group[group.gained].event_id) or 'none') +
                     '; lost ' + (', '.join(group[group.lost].event_id) or 'none') + '; missed ' +
                     (', '.join(group[~group.selected_detected].event_id) or 'none') + '.')
    lines += ['', '## Validation and limitations', '',
              'All161 preflight tests passed, including independent-transform properties and pinned corrected-metric parity. '
              'Independent verification replays every calibration selection, prediction, all336 metric rows and event IDs; '
              'unchanged references match the prior study. Full re-extracted all-output scores agree with saved scores and '
              'produce exactly the original thresholded predictions. CPU spot checks independently reconstruct residuals '
              'at predetermined calibration and assessment locations. This is sampled neural replication, not a second full inference run.', '',
              'All252 calibration half/quarter rows and detected IDs were independently replayed. This table is descriptive: '
              'the checkpoints and rules already used the entire2006 year. '
              'It cannot provide independent validation. A [chronological validation design](chronological-validation-design-2026-11-01.md) '
              'requires fold-specific training/scaling and separate calibration/validation before broader tuning; no such retraining '
              'was launched here. Only six target calibration IDs constrain selection.', '',
              'Single seed, reused benchmark, offline future context, expanded contextual inputs, unequal prior training and additional '
              'inference/selection compute limit interpretation. Inherited ESA preparation restores some annotated numerical values. '
              'This study measures that fixed prepared benchmark and does not quantify the preparation effect or claim label-independent '
              'operational performance. Failed outcomes and original references are preserved.', '',
              '[Protocol](target-boundary-protocol-2026-11-01.md), '
              '[verified metrics](tables/target-boundary-2026-11-01/verified_results.csv), '
              '[event IDs](tables/target-boundary-2026-11-01/event_ids.csv), '
              '[verification](tables/target-boundary-2026-11-01/verification.json), '
              '[W&B project](https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning), '
              '[Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/target-boundary-2026-11-01).']
    (ROOT / 'docs/research/target-boundary-results-2026-11-01.md').write_text('\n'.join(lines) + '\n')
    pd.DataFrame(summary).to_csv(TABLE / 'calibration_summary.csv', index=False)
    for p in OUTPUT.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.csv'):
            target = TABLE / 'run-records' / p.relative_to(OUTPUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    print(pooled[pooled.variant.isin(['original','base','boundary'])][['arm','variant','scope','EW_F_0.50','TPe','FNe','nominal_false_alarm_rate']].to_string(index=False))


if __name__ == '__main__':
    main()
