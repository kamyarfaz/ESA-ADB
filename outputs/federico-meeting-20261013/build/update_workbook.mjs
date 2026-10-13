import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const base='/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20261013';
const original='/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260930/deliverables/ESA_all_methods_results_2026-09-30.xlsx';
const w=await SpreadsheetFile.importXlsx(await FileBlob.load(original));
const d=JSON.parse(await fs.readFile(base+'/build/october_data.json','utf8'));
const originalCells=JSON.parse(await fs.readFile(base+'/build/original-cells.json','utf8'));
const fields=['id','campaign','method','run','seed','channels','period','scope','calibration_f05','EW_F_0.50','EW_precision','EW_recall','TPe','FPe','FNe','num_true_events','num_pred_events','epoch','epochs','window_cap','train_start','train_end','cal_start','cal_end','assessment_start','assessment_end','boundary','false_positive_seconds','nominal_seconds','nominal_false_alarm_rate','false_alarms_per_day','alarming_precision','metric_version','data','notes','source'];
const col=i=>{let a='';for(i++;i;i=Math.floor((i-1)/26))a=String.fromCharCode(65+(i-1)%26)+a;return a;};
const val=v=>typeof v==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(v)?new Date(v+'T00:00:00Z'):v??null;
const ref=(r,k='EW_F_0.50')=>`'Assessment results'!${col(fields.indexOf(k))}${d.records.indexOf(r)+2}`;
const fmt=(s,range)=>{const g=s.getRange(range);g.format.font={name:'Arial',size:11};g.format.verticalAlignment='center';g.format.rowHeight=27;};
function header(s,range){const g=s.getRange(range);g.format.fill='#233E50';g.format.font={name:'Arial',size:11,bold:true,color:'#FFFFFF'};g.format.wrapText=true;g.format.rowHeight=48;}
function append(name,values){const s=w.worksheets.getItem(name);const start=s.getUsedRange().values.length;const n=values[0].length;s.tables.items[0].rows.add(null,values.map(r=>r.map(val)));const g=s.getRangeByIndexes(start,0,values.length,n);g.format.font={name:'Arial',size:11};g.format.rowHeight=43;g.format.wrapText=true;g.format.verticalAlignment='center';return s;}
function newSheet(name,heads,rows,widths){const s=w.worksheets.add(name);s.showGridLines=false;s.getRangeByIndexes(0,0,rows.length+1,heads.length).values=[heads,...rows].map(r=>r.map(val));fmt(s,`A1:${col(heads.length-1)}${rows.length+1}`);header(s,`A1:${col(heads.length-1)}1`);for(let i=0;i<heads.length;i++)s.getRangeByIndexes(0,i,rows.length+1,1).format.columnWidth=widths?.[i]??21;s.tables.add(`A1:${col(heads.length-1)}${rows.length+1}`,true,'October'+w.worksheets.items.length);s.freezePanes.freezeRows(1);return s;}
function chart(s,range,title,start,end,format){const c=s.charts.add('bar',Array.isArray(range)?range:s.getRange(range));c.title=title;c.setPosition(start,end);c.titleTextStyle.fontSize=16;c.titleTextStyle.typeface='Arial';c.legend={position:'bottom',textStyle:{typeface:'Arial',fontSize:12}};c.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:12}};c.yAxis={numberFormatCode:format,numberFormatSourceLinked:false,crosses:'autoZero',textStyle:{typeface:'Arial',fontSize:12}};c.series.items.forEach((v,i)=>v.fill=['#3569A2','#DF8739'][i%2]);return c;}
let s=append('Assessment results',d.records.slice(d.old_counts.records).map(r=>fields.map(k=>r[k])));
const last=d.records.length+1;
s.getRange(`I146:L${last}`).setNumberFormat('0.000000');s.getRange(`U146:Z${last}`).setNumberFormat('yyyy-mm-dd');s.getRange(`AB146:AC${last}`).setNumberFormat('#,##0.000');s.getRange(`AD146:AF${last}`).setNumberFormat('0.000000');
for(let i=0;i<d.old_counts.records;i++)if(d.records[i].campaign==='No short rolls paired')s.getCell(i+1,34).values=[['Matched 5 epochs / 25000 windows. Seeds 42/43/44 completed; see No short rolls for all-seed means. Rolls remain in legacy branch.']];
append('Sources',d.sources.slice(d.old_counts.sources));
const pk=['run','campaign','epochs','seed','batch_size','window_cap','channels','selection','scoring','limits','source'];
append('Run protocols',d.protocols.slice(d.old_counts.protocols).map(r=>pk.map(k=>r[k])));
const hk=['run','arm','epoch','loss','normal_mse','pseudo_penalty','calibration_f05','seconds','source'];
s=append('Training history',d.histories.slice(d.old_counts.histories).map(r=>hk.map(k=>r[k])));s.getRange(`D803:H${d.histories.length+1}`).setNumberFormat('0.000000');
s=w.worksheets.getItem('Read me');
s.getRange('B2').values=[['Updated 13 October 2026 with completed research through 8 October. Preserves earlier results; 256 assessment records, including internal primary and secondary policies. No new training.']];
s.getRange('B12').values=[['Augmentation and no-short-roll summaries use all three seeds. Internal comparisons has both scopes and labels primary versus secondary policies. Means across seeds are not pooled event metrics.']];
s.getRange('B18').values=[['All recent candidates are complete; none justified promotion. Retain the full-period baseline. Consolidate findings and agree calibration/generalization coverage before another model search.']];
s.getRange('A19:B22').values=[['October results','Internal comparisons summarizes checkpoint policy, loss weight, channel isolation, score changes and direct telemetry. Complete per-seed rows are in Assessment results.'],['Internal protocol','Train before each fold year; calibrate January–June; validate July–December 2001 or 2002. Same real data per comparison. Not directly comparable with 2003–2005 or full-period scores.'],['Diagnostics','Calibration audit holds 60 selected/counterfactual grid rows, Event audit holds 132 event records, and Channel interventions holds 180 model-sensitivity records. These are diagnostics, not new benchmark scores.'],['Data integrity','Historical results are preserved. Destination migration verification has unresolved mismatches in other artifacts; new research environment checks remain pending. This workbook adds no new model evaluation.']];
fmt(s,'A19:B22');s.getRange('A19:B22').format.wrapText=true;s.getRange('A19:B22').format.rowHeight=62;
s=w.worksheets.getItem('Method guide');s.getRange('D10:E10').values=[['Three-seed mean improved all three years, but 2004 remained unstable. No promotion.','2003–2005 development, seeds 42/43/44']];
append('Method guide',[
 ['Fixed epoch five','Use final epoch instead of calibration-selected checkpoint','Test checkpoint-selection failure','Zero F0.5 in both folds; 2002 mean false events 53 to 100','Internal 2001/2002, three seeds'],
 ['Loss weight 0.1','Reduce synthetic-anomaly penalty weight from 1 to 0.1','Improve nominal reconstruction and separation','Better training MSE; 2002 mean false events 53 to 159','Internal 2001/2002, three seeds; both policies saved'],
 ['Channel isolation','Diagonal channel attention with shared training','Test cross-channel interference','Primary F0.5 zero; 2002 mean false events 128.67','Internal 2001/2002, three seeds; both policies saved'],
 ['Score change','Deviation of log score from previous one-day EWM','Detect upward and downward changes','Only seed 44 detects one 2002 event; no 2001 detections','Internal 2001/2002, three frozen checkpoints per fold'],
 ['Direct telemetry','Scaled deviation from previous one-day EWM','Compare without neural reconstruction','No validation alarms; every event missed','Internal 2001/2002, one deterministic run per fold'],
 ['Calibration and input audits','Inspect fixed rules, event scores and channel levels','Explain transfer failure','Narrow event coverage; cap relaxation is not a general fix','Diagnostics only; no revised benchmark predictions']]);
s.getRange('A19:E24').format.rowHeight=76;
// Update the existing no-roll view and chart to all three seeds, preserving its position.
s=w.worksheets.getItem('No short rolls');
for(let i=0;i<3;i++)for(let j=0;j<2;j++){
 const rs=d.records.filter(r=>r.campaign==='No short rolls paired'&&r.period===`${2003+i} Apr–Dec`&&r.scope==='system_all_channels'&&r.method===(j===0?'Short coordinated generator':'Short generator without rolls'));
 if(rs.length!==3)throw new Error('No-roll seed coverage');
 s.getCell(i+1,j+1).formulas=[['=AVERAGE('+rs.map(r=>ref(r)).join(',')+')']];
 s.getCell(i+1,j+4).formulas=[['=STDEV('+rs.map(r=>ref(r)).join(',')+')']];
}
s.getRange('D1:G1').values=[['Mean difference','Control sample SD','No-roll sample SD','Seeds']];header(s,'D1:G1');s.getRange('D1:G4').format.columnWidth=22;
for(let i=2;i<=4;i++){s.getRange(`D${i}`).formulas=[[`=C${i}-B${i}`]];s.getRange(`G${i}`).values=[[3]];}
s.getRange('B2:F4').setNumberFormat('0.0000');s.getRange('A7').values=[['Three-seed system means (42/43/44). Same real April–December periods; five epochs and 25000 windows. 2004 remains unstable. No full-period promotion.']];
s.charts.items[0].title='No short rolls: three-seed mean system F0.5';s.charts.items[0].setData(s.getRange('A1:C4'));
s.getRange('A33:G33').values=[['Period','Scope','Seed','With rolls F0.5','Without rolls F0.5','Difference','False events: no rolls']];header(s,'A33:G33');
let row=34;for(const year of [2003,2004,2005])for(const scope of ['system_all_channels','input_channels_40_47'])for(const seed of [42,43,44]){
 const pair=d.records.filter(r=>r.campaign==='No short rolls paired'&&r.period===`${year} Apr–Dec`&&r.scope===scope&&r.seed===seed);
 const a=pair.find(r=>r.method==='Short coordinated generator'),b=pair.find(r=>r.method==='Short generator without rolls');
 s.getRange(`A${row}:C${row}`).values=[[`${year} Apr–Dec`,scope==='system_all_channels'?'System':'Input channels',seed]];
 s.getRange(`D${row}:G${row}`).formulas=[['='+ref(a),'='+ref(b),`=E${row}-D${row}`,'='+ref(b,'FPe')]];row++;
}
fmt(s,'A34:G51');s.getRange('D34:F51').setNumberFormat('0.0000');
// Clear protocol boundaries in coverage without changing historical availability.
s=w.worksheets.getItem('Assessment coverage');
append('Assessment coverage',[
 ['Checkpoint policy','Not run','Not run','Not run','Not run','Internal July–December 2001 and 2002'],
 ['Loss weight 0.1','Not run','Not run','Not run','Not run','Internal July–December 2001 and 2002'],
 ['Channel isolation','Not run','Not run','Not run','Not run','Internal July–December 2001 and 2002'],
 ['Score change','Not run','Not run','Not run','Not run','Internal July–December 2001 and 2002'],
 ['Direct telemetry','Not run','Not run','Not run','Not run','Internal July–December 2001 and 2002']]);
const definitions=[
 ['Joint AE','Joint AE / calibration_selected','Primary'],['Joint epoch 5','Joint AE / fixed_epoch5','Secondary'],
 ['Loss weight 0.1','Loss weight 0.1 / calibration_selected','Primary'],['Loss 0.1 epoch 5','Loss weight 0.1 / fixed_epoch5','Secondary'],
 ['Isolated AE','Isolated AE / calibration_selected','Primary'],['Isolated epoch 5','Isolated AE / fixed_epoch5','Secondary'],
 ['Score change','Score change','Primary'],['Direct telemetry','Direct telemetry','Deterministic']];
const summary=[];const groups=[];
for(const year of [2001,2002])for(const scope of ['system_all_channels','input_channels_40_47'])for(const [label,method,policy] of definitions){
 const rs=d.records.filter(r=>r.method===method&&r.period===`${year} Jul–Dec`&&r.scope===scope);
 if(rs.length!==(method==='Direct telemetry'?1:3))throw new Error(`Coverage ${method}: ${rs.length}`);
 summary.push([label,`${year} Jul–Dec`,scope==='system_all_channels'?'System':'Input channels',policy,null,null,null,null,null,null]);groups.push(rs);
}
s=newSheet('Internal comparisons',['Method','Real validation','Event scope','Policy','Runs','Mean F0.5','Sample SD F0.5','Mean false events','Mean recall','Mean nominal alarm fraction'],summary,[25,22,23,20,10,18,21,22,18,28]);
groups.forEach((rs,i)=>{const r=i+2;s.getRange(`E${r}`).formulas=[['=COUNT('+rs.map(x=>ref(x)).join(',')+')']];for(const [c,key] of [['F','EW_F_0.50'],['H','FPe'],['I','EW_recall'],['J','nominal_false_alarm_rate']])s.getRange(`${c}${r}`).formulas=[['=AVERAGE('+rs.map(x=>ref(x,key)).join(',')+')']];if(rs.length>1)s.getRange(`G${r}`).formulas=[['=STDEV('+rs.map(x=>ref(x)).join(',')+')']];});
s.getRange('F2:G33').setNumberFormat('0.000000');s.getRange('H2:H33').setNumberFormat('0.00');s.getRange('I2:J33').setNumberFormat('0.00%');
s.getRange('A35:J37').merge();s.getRange('A35').values=[['Same internal real validation periods, channels 40–47. Means across seeds, not pooled event metrics. The direct reference is one deterministic run per fold; sample SD is not applicable. Primary checkpoint selection is calibration_selected; epoch five remains secondary. All advancement gates failed. Controls are stored once and reused across comparisons.']];fmt(s,'A35:J37');s.getRange('A35:J37').format.wrapText=true;
s.getRange('A90:C95').values=[['Primary method','Mean system F0.5','Mean false events'],...['Joint AE','Loss weight 0.1','Isolated AE','Score change','Direct telemetry'].map(x=>[x,null,null])];
for(let i=0;i<5;i++){const idx=summary.findIndex(r=>r[0]===['Joint AE','Loss weight 0.1','Isolated AE','Score change','Direct telemetry'][i]&&r[1]==='2002 Jul–Dec'&&r[2]==='System');s.getRange(`B${91+i}:C${91+i}`).formulas=[[`=F${idx+2}`,`=H${idx+2}`]];}
chart(s,'A90:B95','July–December 2002: mean system F0.5','A40','F59','0.00');chart(s,['A90:A95','C90:C95'].map(x=>s.getRange(x)),'July–December 2002: mean false events','A63','F82','0');
// Diagnostic evidence stays separate from assessment outcomes.
for(const [name,rows] of Object.entries(d.diagnostics)){
 const keys=Object.keys(rows[0]);const sh=newSheet(name,keys,rows.map(r=>keys.map(k=>r[k])),keys.map(k=>/source|seed|fold/.test(k)?13:/method|diagnostic|arm|period|channel/.test(k)?25:23));
 for(let c=0;c<keys.length;c++)if(rows.some(r=>typeof r[keys[c]]==='number'&&!Number.isInteger(r[keys[c]])))sh.getRangeByIndexes(1,c,rows.length,1).setNumberFormat('0.000000');
}
// Recalculate, check summary values and confirm unrelated sheets retain cells/formulas.
w.recalculate();
const nr=w.worksheets.getItem('No short rolls').getRange('B2:C4').values;
const expected=[[.836292,.918525],[.241707,.458731],[.931322,.953285]];
for(let i=0;i<3;i++)for(let j=0;j<2;j++)if(Math.abs(nr[i][j]-expected[i][j])>1e-6)throw new Error('No-roll summary mismatch');
const changed=new Set(['Read me','Method guide','Assessment results','Sources','Run protocols','Training history','No short rolls','Assessment coverage']);
for(const old of originalCells){if(changed.has(old.name))continue;const sh=w.worksheets.getItem(old.name);if(JSON.stringify(sh.getUsedRange().formulas)!==JSON.stringify(old.formulas)||JSON.stringify(sh.getUsedRange().values)!==JSON.stringify(old.values)||sh.charts.items.length!==old.charts)throw new Error('Unintended change '+old.name);}
const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Formula errors'});
await fs.writeFile(base+'/build/workbook_formula_scan.ndjson',errors.ndjson);console.log(errors.ndjson);
// Verify formula response to an input edit, then restore.
const rr=d.records.find(r=>r.method==='Score change'&&r.period==='2002 Jul–Dec'&&r.scope==='system_all_channels'&&r.seed===42);
const cell=w.worksheets.getItem('Assessment results').getRange(ref(rr).split('!')[1]);const old=cell.values[0][0];
const summaryRow=summary.findIndex(r=>r[0]==='Score change'&&r[1]==='2002 Jul–Dec'&&r[2]==='System')+2;
const meanCell=w.worksheets.getItem('Internal comparisons').getRange(`F${summaryRow}`);const before=meanCell.values[0][0];cell.values=[[.3]];w.recalculate();if(Math.abs(meanCell.values[0][0]-before-.1)>1e-9)throw new Error('Formula response failed');cell.values=[[old]];w.recalculate();
const out=base+'/deliverables/ESA_all_methods_results_2026-10-13.xlsx';await (await SpreadsheetFile.exportXlsx(w)).save(out);
const previews=[['Read me','A18:B22'],['Method guide','A19:E24'],['Assessment results','A146:J150'],['No short rolls','A1:G29'],['No short rolls','A33:G42'],['Internal comparisons','A1:J17'],['Internal comparisons','A18:J33'],['Internal comparisons','A40:F59'],['Internal comparisons','A63:F82'],['Calibration audit','A1:H8'],['Event audit','A1:K8'],['Channel interventions','A1:H8'],['Assessment coverage','A14:F18'],['Run protocols','A23:K28'],['Training history','A803:I809'],['Sources','A96:D102']];
for(let i=0;i<previews.length;i++){const [sheetName,range]=previews[i];const b=await w.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(base+`/build/excel-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
await fs.writeFile(base+'/build/workbook-validation.json',JSON.stringify({output:out,...d.validation,sheets:w.worksheets.items.length,charts:w.worksheets.items.reduce((a,s)=>a+s.charts.items.length,0),unrelatedSheetsPreserved:true,formulaResponseVerified:true,previews},null,2));
console.log(out);
