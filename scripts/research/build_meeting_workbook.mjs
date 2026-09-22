import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const data=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const out=process.argv[3];
if (!out) throw new Error('Usage: node build_meeting_workbook.mjs payload.json output_directory');
const wb=Workbook.create();
const names=['Overview','Results','Calibration','Channels','Correlations','Plan','Notes'];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
function table(name,title,subtitle,headers,rows,widths){
 const s=sheets[name];s.showGridLines=false;
 s.getCell(0,0).values=[[title]];s.getCell(1,0).values=[[subtitle]];
 s.getRangeByIndexes(3,0,1,headers.length).values=[headers];
 if(rows.length)s.getRangeByIndexes(4,0,rows.length,headers.length).values=rows;
 const used=s.getRangeByIndexes(0,0,rows.length+4,headers.length);
 used.format.font={name:'Liberation Sans',size:11,color:'#243447'};
 used.format.rowHeight=23;
 s.getRange('A1').format.font={name:'Liberation Sans',size:15,bold:true,color:'#172c43'};
 s.getRange('A2').format.font={italic:true,size:10,color:'#536477'};
 s.getRangeByIndexes(3,0,1,headers.length).format={fill:'#294661',font:{bold:true,color:'#ffffff'},wrapText:true,rowHeight:38};
 widths.forEach((w,i)=>s.getRangeByIndexes(0,i,rows.length+4,1).format.columnWidth=w);
 for(let i=0;i<rows.length;i++)if(i%2===1)s.getRangeByIndexes(i+4,0,1,headers.length).format.fill='#f0f4f7';
 if(rows.length){s.tables.add(s.getRangeByIndexes(3,0,rows.length+1,headers.length),true,name+'Table');}
 if(rows.length>12)s.freezePanes.freezeRows(4);
 return s;
}
const rows=data.rows;
const r=table('Results','Experiment results','Snapshot: 22 September 2026. Assessment = April–December. Blank means pending.',
 ['Year','Model','Channel subset','Inputs','Calibration F0.5','Assessment F0.5','Precision','Recall','TP events','FP events','FN events','FP seconds','Selected epoch','Seed','Completion','Source run'],
 rows.map(x=>[x.year,x.model,x.channels,x.n_channels,x.calibration_f05,x.assessment_f05,x.precision,x.recall,x.TPe,x.FPe,x.FNe,x.false_positive_seconds,x.epoch,x.seed,x.status,x.source_id]),
 [10,32,25,10,18,18,15,15,13,13,13,18,16,10,25,65]);
r.getRange(`E5:H${rows.length+4}`).setNumberFormat('0.0000');
r.getRange(`L5:L${rows.length+4}`).setNumberFormat('#,##0.000');
r.getRange(`A5:A${rows.length+4}`).setNumberFormat('0');
const cal=table('Calibration','Calibration decisions','January–March only. Parameters frozen before assessment; no assessment-based selection.',
 ['Year','Model','Channels','F0.5','Precision','Recall','TP events','FP events','FN events','Threshold','Merge gap (samples)','Min duration (samples)'],
 rows.map(x=>[x.year,x.model,x.channels,x.calibration_f05,x.calibration_precision,x.calibration_recall,x.calibration_TPe,x.calibration_FPe,x.calibration_FNe,x.threshold,x.merge_gap,x.min_dur]),
 [10,32,25,15,15,15,13,13,13,20,20,21]);
cal.getRange(`D5:F${rows.length+4}`).setNumberFormat('0.0000');cal.getRange(`J5:J${rows.length+4}`).setNumberFormat('0.000000');
const ov=table('Overview','ESA Mission 1 research results','Completed AE campaigns, seed 42. Scores are development assessments, not full-year or final-test scores.',
 ['Year','8 channels + pseudo','11 channels + pseudo','8 channels reconstruction','Target'],[[2003,null,null,null,.85],[2004,null,null,null,.85],[2005,null,null,null,.85]],[12,25,26,30,15,16,16,16,16]);
for(let y=0;y<3;y++)for(let m=0;m<3;m++){
 const model=['Baseline AE + pseudo','Expanded AE + pseudo','Reconstruction-only AE'][m];
 const i=rows.findIndex(x=>x.year===2003+y&&x.model===model);
 ov.getCell(4+y,1+m).formulas=[[`='Results'!F${i+5}`]];
}
ov.getRange('B5:E7').setNumberFormat('0.0000');
ov.getRange('A9').values=[['Retain pseudo-anomaly baseline; no consistent ≥0.85 result across all folds.']];
ov.getRange('A10').values=[['Specialist campaign is ongoing. Completed folds appear in Results; pending folds stay blank.']];
const chart=ov.charts.add('line',ov.getRange('A4:E7'));chart.title='Assessment F0.5 by development year';chart.setPosition('A12','I29');
chart.titleTextStyle.fontSize=14;chart.titleTextStyle.typeface='Liberation Sans';
chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Liberation Sans',fontSize:11}};
chart.yAxis={numberFormatCode:'0.0',numberFormatSourceLinked:false,textStyle:{typeface:'Liberation Sans',fontSize:11}};
chart.legend={position:'bottom',textStyle:{typeface:'Liberation Sans',fontSize:10}};
const profiles=data.channels;
const ch=table('Channels','Training-only input inventory','2000–2002: all nominal rows for min/max/flatness; hourly nominal sample for uniqueness and modal share.',
 ['Input','Kind','Subsystem','Unit code','Group','Target flag','Training min','Training max','Training std','Hourly unique','Hourly mode share','Unchanged 30s share','Review flag'],
 profiles.map(x=>[x.feature,x.kind,x.subsystem,x.unit_code,x.group,x.target,x.min,x.max,x.std,x.sample_unique,x.sample_mode_fraction,x.unchanged_30s_fraction,x.review_flag]),
 [23,17,19,22,10,15,17,17,18,17,20,22,37]);
ch.getRange(`G5:I${profiles.length+4}`).setNumberFormat('0.000000');ch.getRange(`K5:L${profiles.length+4}`).setNumberFormat('0.00%');
const pairs=data.pairs;
const cor=table('Correlations','Potentially redundant pairs','Hourly nominal sample only. Listed if |Pearson| or |Spearman| ≥0.95. Correlation is not a removal decision.',
 ['Input A','Input B','Pearson 2000–2002','Spearman 2000–2002','Pearson 2000','Pearson 2001','Pearson 2002'],
 pairs.map(x=>[x.feature_a,x.feature_b,x.pearson,x.spearman,x.pearson_2000,x.pearson_2001,x.pearson_2002]),[24,24,24,26,22,22,22]);
cor.getRange(`C5:G${pairs.length+4}`).setNumberFormat('0.0000');
const phases=[
 ['1. Inventory and raw quality','Audit raw cadence, missingness, duplicates, counters and telecommands. Preserve source files.','Dictionary and source-to-prepared report; physical meanings may remain anonymized.'],
 ['2. Plot temporal behavior','Review full-resolution training examples, distributions, flat stretches and monthly ranges.','Explain cleaning and transformations; never remove real anomalies as outliers.'],
 ['3. Review redundant groups','Check Pearson, Spearman, changes and lag relationships by year and subsystem.','Keep/review/drop rationale per channel; retain rare and correlation-break signals where useful.'],
 ['4. Small controlled pilot','Proposed 10k shared nominal windows, 5 epochs, training-internal chronological validation.','Freeze exact channel lists first. Compare original 8, broad telemetry and representative sets.'],
 ['5. Full comparison','Compare selected sets and window budgets separately with identical assessment scope.','Report every fold, false alarms, recall, memory and runtime; no test-driven selection.'],
 ['6. Tuning and robustness','Tune only after data policy is fixed; repeat selected configurations across seeds.','Bounded search and documented uncertainty; review sparse calibration with supervisor.']];
const plan=table('Plan','Data-first research plan','Finish the running experiment. Do not launch additional training until channel selection is justified.',
 ['Stage','Work','Deliverable / decision'],phases,[32,83,90]);
plan.getRange('A5:C10').format.wrapText=true;plan.getRange('A5:C10').format.rowHeight=66;
const notes=[
 ['Metric','Corrected ESA event-wise F0.5, version esa-ew-id-duration-v1. Precision includes nominal false-alarm duration penalty.'],
 ['2003 split','Train 2000–2002; calibrate January–March 2003; assess April–December 2003.'],
 ['2004 split','Train 2000–2003; calibrate January–March 2004; assess April–December 2004.'],
 ['2005 split','Train 2000–2004; calibrate January–March 2005; assess April–December 2005.'],
 ['Selection','Checkpoint and alarm settings use calibration only. These assessment folds have already informed development.'],
 ['EDA scope',`${data.protocol.rows.toLocaleString('en-US')} rows, ${data.protocol.nominal_rows.toLocaleString('en-US')} nominal; ${data.protocol.hourly_nominal_sample_rows.toLocaleString('en-US')} hourly nominal samples, 2000–2002 only.`],
 ['Input types','76 telemetry channels + 11 telecommands. Unit/subsystem names are anonymized codes, not known engineering units.'],
 ['Constant inputs','10 exactly constant in full nominal training. Another 9 are sample-modal ≥99.9% but nonconstant at full resolution. Review; do not auto-delete.'],
 ['Limits','Prepared filling can hide raw gaps. Hourly sampling can miss rare behavior. High correlation does not prove a signal is useless.'],
 ['Comparability','AE is retrospective; forecasting is causal with different scoring/alarm rules. No architecture-only conclusion from comparing them.'],
 ['Source records','Experiment results.json and frozen_rule.json, channel metadata and training-only pilot. Source run identifiers are in Results.'],
 ['Research backup','https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-update-2026-09-22'],
 ['Benchmark code','https://github.com/kplabs-pl/ESA-ADB'],
 ['Benchmark paper','https://arxiv.org/abs/2406.17826'],
 ['Latest specialist','Ongoing campaign. This is a fixed snapshot; no partial epoch metrics are reported as final results.']];
const notesSheet=table('Notes','Definitions and sources','Read before comparing models or treating flagged channels as removable.', ['Topic','Explanation'],notes,[28,125]);
notesSheet.getRange('A5:B19').format.wrapText=true;notesSheet.getRange('A5:B19').format.rowHeight=45;
console.log((await wb.inspect({kind:'table',range:'Overview!A4:E7',include:'values,formulas',tableMaxRows:5,tableMaxCols:5})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:30},summary:'Formula error scan'})).ndjson);
const ranges={Overview:'A1:I29',Results:'A1:H12',Calibration:'A1:F12',Channels:'A1:I15',Correlations:'A1:G15',Plan:'A1:C10',Notes:'A1:B19'};
for(const [sheetName,range] of Object.entries(ranges)){
 const preview=await wb.render({sheetName,range,scale:1,format:'png'});
 await fs.writeFile(`${out}/preview_${sheetName}.png`,new Uint8Array(await preview.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(`${out}/ESA_research_review.xlsx`);
console.log('Exported workbook');
