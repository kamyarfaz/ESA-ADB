import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB/outputs/thesis-summary-20261013';
const d=JSON.parse(await fs.readFile(`${root}/build/summary-data.json`,'utf8'));
const wb=Workbook.create();const sh=wb.worksheets.add('Thesis results');sh.showGridLines=false;
sh.getRange('A2').values=[['Satellite anomaly detection — thesis results']];
sh.getRange('A3').values=[['All-channel event evaluation · corrected event-wise metrics · input channels listed per method']];
sh.getRange('A4').values=[['Means across runs; SD = F0.5 sample standard deviation. Compare methods within the same period and protocol; test periods were previously inspected.']];
const hdr=['Technique','Comparison group','Input channels','Train years','Calibration','Assessment','Runs','Precision','Recall','F0.5','F0.5 SD','False events'];
sh.getRange('A5:L5').values=[hdr];sh.getRange(`A6:L${d.rows.length+5}`).values=d.rows;
const used=sh.getRange(`A1:L${d.rows.length+5}`);used.format.font={name:'Arial',size:11,color:'#20313D'};used.format.rowHeight=32;used.format.verticalAlignment='center';
sh.getRange('A2').format.font={name:'Arial',size:16,bold:true};sh.getRange('A3:A4').format.font={name:'Arial',size:10,color:'#536574'};
const widths=[235,200,160,108,130,190,52,80,80,80,80,90];
for(let c=0;c<12;c++)sh.getRangeByIndexes(4,c,d.rows.length+1,1).format.columnWidthPx=widths[c];
sh.getRange(`A5:F${d.rows.length+5}`).format.wrapText=true;
sh.getRange(`G6:G${d.rows.length+5}`).setNumberFormat('0');sh.getRange(`H6:K${d.rows.length+5}`).setNumberFormat('0.0000');sh.getRange(`L6:L${d.rows.length+5}`).setNumberFormat('0.00');
const t=sh.tables.add(`A5:L${d.rows.length+5}`,true,'ThesisResults');t.showBandedRows=true;
sh.getRange('A5:L5').format={fill:'#243D50',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true};
sh.getRange(`J6:J${d.rows.length+5}`).format.font={name:'Arial',size:11,bold:true};
sh.freezePanes.freezeRows(5);
for(const p of d.provenance)wb.notes.add({id:`source-${p.excel_row}`,target:{cell:{sheetName:sh.name,sheetId:sh.sheetId,address:`A${p.excel_row}`}},authorId:'',createdAt:'',body:{plainText:`Sources (repository-relative): ${p.sources.join('; ')}. Epoch budget: ${p.epochs??'not recorded'}; training-window cap: ${p.window_cap??'not recorded'} (0 = uncapped). Includes every recorded seed for this row; no best-seed selection. Primary checkpoint policy except the explicit fixed-fifth-epoch comparison.`}});
wb.recalculate();
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:30},maxChars:1500});await fs.writeFile(`${root}/build/workbook-check.ndjson`,scan.ndjson);
for(const [name,range] of [['top','A1:L18'],['middle','A25:L39'],['bottom',`A54:L${d.rows.length+5}`]]){
 const img=await wb.render({sheetName:sh.name,range,scale:1,format:'png'});await fs.writeFile(`${root}/build/${name}.png`,new Uint8Array(await img.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(`${root}/deliverables/ESA_thesis_results.xlsx`);
console.log(JSON.stringify({rows:d.rows.length,sheets:1,scan:scan.ndjson}));
