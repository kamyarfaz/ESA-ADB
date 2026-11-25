import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB';
const out=root+'/outputs/cross-mission-study-20261125';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/workbook-before.xlsx'));
console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:2500,tableMaxRows:2,tableMaxCols:5})).ndjson);
const source=wb.worksheets.getItem('Mission 2');
const before=await wb.render({sheetName:source.name,range:'A2:N12',scale:1});
await fs.writeFile(out+'/before.png',new Uint8Array(await before.arrayBuffer()));
if(!process.argv.includes('--edit'))process.exit(0);
const data=JSON.parse(await fs.readFile(out+'/workbook-data.json','utf8'));
const headers=['Mission','Technique','Seed','Condition','Event definition','Assessment','Base F0.5','Base TPR','Base time FPR','Base detected','Base total','System F0.5','System TPR','System time FPR','System detected','System total'];
for(const [name,rows] of [['Cross-mission fresh',data.rows.filter(r=>r[3]==='fresh_training')],['M1 to M2 transfer',data.rows.filter(r=>r[3]!=='fresh_training')]]){
  const s=wb.worksheets.add(name);s.showGridLines=false;
  s.getRange('A2').values=[[name==='Cross-mission fresh'?'Whole-system models: both missions, every seed':'Existing Mission 1 models assessed on Mission 2']];
  s.getRange('A3').values=[['M1 assessment: 2007–2013. M2 assessment: Oct 2001–Jun 2003. Pooled first, then annual periods.']];
  s.getRange('A4').values=[[name==='Cross-mission fresh'?'Base scope: fixed 13 mission-specific representatives. System scope: all mission channels.':'Base scope: mapped original eight Mission 1 inputs. Source weights and thresholds unchanged in both conditions.']];
  const end=5+rows.length;s.getRange('A5:P5').values=[headers];s.getRange(`A6:P${end}`).values=rows;
  s.getRange(`A2:P${end}`).format.font={name:'Arial',size:10,color:'#203D50'};
  s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#203D50'};
  s.getRange('A3:A4').format.rowHeight=25;
  s.getRange('A5:P5').format={fill:'#203D50',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:43};
  for(const [c,w] of [['A',9],['B',36],['C',8],['D',34],['E',23],['F',14],['G',14],['H',14],['I',15],['J',14],['K',12],['L',14],['M',14],['N',15],['O',14],['P',12]])s.getRange(`${c}5:${c}${end}`).format.columnWidth=w;
  s.getRange(`A6:P${end}`).format.rowHeight=26;s.getRange(`A6:P${end}`).format.verticalAlignment='center';
  for(const c of ['G','L'])s.getRange(`${c}6:${c}${end}`).setNumberFormat('0.000000');
  for(const c of ['H','I','M','N'])s.getRange(`${c}6:${c}${end}`).setNumberFormat('0.0000%');
  for(const c of ['A','C','J','K','O','P'])s.getRange(`${c}6:${c}${end}`).setNumberFormat('0');
  s.tables.add(`A5:P${end}`,true,name==='Cross-mission fresh'?'CrossMissionFreshResults':'CrossMissionTransferResults');
  s.freezePanes.freezeRows(5);s.freezePanes.freezeColumns(2);
  const notes=[
    'TPR counts detected event IDs. Time FPR measures false-alarm duration divided by nominal duration. F0.5 uses the corrected event metric.',
    'Anomaly-only uses identical predictions. Rare-event and out-of-scope annotations remain neutral for false alarms.',
    'Seeds 42,43,44 are independent training repeats. Every seed is retained. Pooled scores are recomputed, not averages of annual scores.',
    'Fresh models: 13 base inputs; expanded 25 (M1) / 26 (M2). Training/calibration splits remain separate from assessment.',
    'Only M1 channels 62/63 permit 900-second-old readings; all others 600s. V1 preparation failed before training; v2 completed 18/18.',
    'Transfer mapping is experimental and unstable. Physical channel identities are anonymized. Target-training normalization is adaptation.',
    'Inputs, score sources, raw preparation, whole-system calibration and batch 32 differ from historical studies. Benchmarks were previously viewed.',
    'M2 fresh models: eight of nine detect 0/6 anomalies; masked seed 43 detects 1/6. Primary gains mainly reflect rare-event detection.',
    'Transfer with source scaling alarms continuously. Target normalization still produces excessive false alarms.',
    'Three seeds and added compute do not establish robust target attainment or independent generalization.',
    'Source: https://github.com/kamyarfaz/ESA-ADB/blob/main/docs/research/cross-mission-results-2026-11-25.md'
  ];
  for(let i=0;i<notes.length;i++){const r=end+3+i;s.getRange(`A${r}`).values=[[notes[i]]];s.getRange(`A${r}:P${r}`).format={font:{name:'Arial',size:10,color:'#475569'},rowHeight:25};}
}
wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!',options:{useRegex:true,maxResults:20},maxChars:1500})).ndjson);
for(const [name,label] of [['Cross-mission fresh','fresh'],['M1 to M2 transfer','transfer']]){
  const end=5+data.rows.filter(r=>name==='Cross-mission fresh'?r[3]==='fresh_training':r[3]!=='fresh_training').length;
  for(const [part,range] of [['pooled','A2:P14'],['annual','A45:P50'],['notes',`A${end+3}:P${end+13}`]]){
    const img=await wb.render({sheetName:name,range,scale:1});await fs.writeFile(out+`/${label}-${part}.png`,new Uint8Array(await img.arrayBuffer()));
  }
}
await fs.mkdir(out+'/deliverables',{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_full_and_historical.xlsx');
