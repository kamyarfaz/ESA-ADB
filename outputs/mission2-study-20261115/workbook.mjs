import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB';
const out=root+'/outputs/mission2-study-20261115';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/workbook-before.xlsx'));
console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:3500,tableMaxRows:3,tableMaxCols:8})).ndjson);
const sheet=wb.worksheets.getItemAt(0);
await fs.writeFile(out+'/source-values.json',JSON.stringify(sheet.getUsedRange().values));
console.log((await wb.inspect({kind:'table',range:sheet.name+'!A1:O9',tableMaxCols:15,tableMaxRows:9,maxChars:5000})).ndjson);
const img=await wb.render({sheetName:sheet.name,range:'A1:M8',scale:1});
await fs.writeFile(out+'/before.png',new Uint8Array(await img.arrayBuffer()));
if(!process.argv.includes('--edit'))process.exit(0);
const data=JSON.parse(await fs.readFile(out+'/workbook-data.json','utf8'));
const study=wb.worksheets.add('Mission 2');
study.showGridLines=false;
study.getRange('A2').values=[['Mission 2: three techniques, all three seeds']];
study.getRange('A3').values=[['Train 2000; calibrate Jan–Sep 2001; assess Oct 2001–Jun 2003. Pooled first, then partial/full annual periods.']];
study.getRange('A4').values=[['Selected scope: 58,59,75,76,77,83,84,85. Expanded inputs add 90,86,89. Seeds are independent training repeats.']];
const headers=['Technique','Seed','Event definition','Assessment','Selected F0.5','Selected TPR','Selected time FPR','Selected detected','Selected total','System F0.5','System TPR','System time FPR','System detected','System total'];
const end=5+data.rows.length;
study.getRange('A5:N5').values=[headers];
study.getRange(`A6:N${end}`).values=data.rows;
study.getRange(`A2:N${end}`).format.font={name:'Arial',size:10,color:'#203D50'};
study.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#203D50'};
study.getRange('A3:A4').format.rowHeight=25;
study.getRange('A5:N5').format={fill:'#203D50',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:43};
for(const [col,width] of [['A',36],['B',8],['C',23],['D',14],['E',14],['F',14],['G',15],['H',14],['I',14],['J',14],['K',14],['L',15],['M',14],['N',14]])study.getRange(`${col}5:${col}${end}`).format.columnWidth=width;
study.getRange(`A6:N${end}`).format.rowHeight=26;
study.getRange(`A6:N${end}`).format.verticalAlignment='center';
for(const col of ['E','J'])study.getRange(`${col}6:${col}${end}`).setNumberFormat('0.000000');
for(const col of ['F','G','K','L'])study.getRange(`${col}6:${col}${end}`).setNumberFormat('0.0000%');
for(const col of ['H','I','M','N'])study.getRange(`${col}6:${col}${end}`).setNumberFormat('0');
study.tables.add(`A5:N${end}`,true,'Mission2Results');
study.freezePanes.freezeRows(5);study.freezePanes.freezeColumns(1);
const notes=[
 'Primary: anomaly_and_rare. Secondary: anomaly_only, with rare-event intervals neutral for false alarms. Predictions are identical.',
 'Selected scope always uses the same eight base channels. System scope includes all channels. TPR counts event IDs; FPR measures nominal-time duration.',
 'Pooled results are recomputed continuously, not means of annual metrics. 2001 covers Oct–Dec and 2003 covers Jan–Jun.',
 'Expanded inputs also expand score sources. Correlation does not prove useful extra information. Results vary substantially across seeds.',
 'All models use fresh Mission 2 weights and 40 epochs. Masked training allocates 20 masked plus 20 short-no-roll epochs.',
 'Detection recall is low. Added inputs improve selected-scope F0.5 in two seeds but worsen seed44. No consistent improvement is established.',
 'Only six selected calibration IDs and three seeds; reused benchmark and different Mission 1 preparation limit comparisons and generalization.',
 'Source: https://github.com/kamyarfaz/ESA-ADB/blob/main/docs/research/mission2-three-technique-results-2026-11-15.md'
];
for(let i=0;i<notes.length;i++){const r=end+3+i;study.getRange(`A${r}`).values=[[notes[i]]];study.getRange(`A${r}:N${r}`).format={font:{name:'Arial',size:10,color:'#475569'},rowHeight:25};}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Mission 2!A6:N8',tableMaxRows:3,tableMaxCols:14,maxChars:2500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!',options:{useRegex:true,maxResults:20}})).ndjson);
for(const [name,range] of [['pooled','A2:N14'],['annual','A24:N29'],['notes',`A${end+3}:N${end+10}`]]){const preview=await wb.render({sheetName:'Mission 2',range,scale:1});await fs.writeFile(out+'/'+name+'.png',new Uint8Array(await preview.arrayBuffer()));}
await fs.mkdir(out+'/deliverables',{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_full_and_historical.xlsx');
