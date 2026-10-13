import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {FileBlob,SpreadsheetFile,PresentationFile} from '@oai/artifact-tool';
import {finalizePresentation} from '/home/k_faz/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations/container_tools/artifact_tool_utils.mjs';
process.env.RUNTIME_NODE_MODULES='/home/k_faz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const base='/home/k_faz/projects/ESA-ADB/outputs/research-update-20261013';
const old='/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20261013/deliverables';
const skill='/home/k_faz/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const map=new Map([
 ['Federico’s assessment requirement','Assessment protocol'],
 ['What we changed','Modification'],['What happened','Findings'],['Why','Rationale'],
 ['Progress since 30 September  /  Federico Butchellato  /  13 October 2026','Research progress: 30 September–13 October 2026'],
 ['Where we left off','Previous experimental status'],
 ['Removing short temporal rolls looked promising in one seed. We planned to repeat the comparison.','The initial single-seed comparison suggested a benefit from removing short temporal rolls. Replication was required.'],
 ['What we completed','Completed investigations'],
 ['We kept the saved rules and investigated calibration transfer.','The saved decision rules were retained for the calibration-transfer analysis.'],
 ['What we achieved since 30 September','Research contributions since 30 September'],
 ['A stronger experimental record and a clearer account of the limitations','Replicated comparisons, diagnostic evidence and generalization limitations'],
 ['Completed comparisons and checked the evidence','Experimental replication and verification'],
 ['Repeated the promising result across seeds, tested specific explanations and verified saved predictions and metrics.','Completed replication across seeds, evaluated specific hypotheses, and verified saved predictions and metrics.'],
 ['Identified the unresolved problem','Unresolved generalization limitation'],
 ['Next step: agree the evaluation protocol','Proposed evaluation protocol'],
 ['A proposed direction for discussion before another model search','Calibration representativeness and operational evaluation criteria'],
 ['Make calibration coverage explicit','Calibration coverage'],
 ['Agree the operational trade-offs','Operational trade-offs'],
 ['Previously inspected periods remain development data. No new protocol is yet approved.','Previously inspected periods remain development data. The proposed protocol has not been implemented.']
]);
const w=await SpreadsheetFile.importXlsx(await FileBlob.load(old+'/ESA_all_methods_results_2026-10-13.xlsx'));
let edits=0;
for(const s of w.worksheets.items){const range=s.getUsedRange();const vs=range.values,fs=range.formulas;for(let i=0;i<vs.length;i++)for(let j=0;j<vs[i].length;j++)if(!fs[i]?.[j]&&map.has(vs[i][j])){s.getCell(i,j).values=[[map.get(vs[i][j])]];edits++;}}
w.recalculate();
await (await SpreadsheetFile.exportXlsx(w)).save(base+'/deliverables/ESA_all_methods_results_2026-10-13.xlsx');
for(const [name,range,file] of [['Read me','A1:B5','workbook-guide'],['Method guide','A1:E5','workbook-methods']]){
 const b=await w.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile(base+'/build/'+file+'.png',new Uint8Array(await b.arrayBuffer()));
}
const source=old+'/ESA_research_update_2026-10-13.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const records=(await p.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','elements.style','notes.text']})).records;
for(const rec of records){
 for(const e of rec.elements)if(e.kind==='textbox'&&map.has(e.text)){
  const t=p.resolve(e.id);t.text=map.get(e.text);t.text.style={...e.style,autoFit:'none'};
 }
 const s=p.slides.items[rec.index];
 let notes=rec.notes.text.replace('Opening: This update continues the September 30 presentation.','This report extends the research status documented on 30 September 2026.').replace('We retained channels 40–47','The experiments retained channels 40–47').replace('We completed the promised seeds 43 and 44.','Replication using seeds 43 and 44 was completed.');
 if(rec.index===10)notes=notes.replace(/The migration check onOctober13[\s\S]*?(?=\n\nSources)/,'');
 if(rec.index===11)notes='The completed comparisons indicate that several tested modifications do not reliably transfer across the evaluated periods. Further investigation should explicitly address sparse calibration events and variation in nominal operating conditions. The proposed evaluation protocol should specify the primary system-level metric and secondary local metric; calibration-event coverage; acceptable false-alarm burden and detection delay; and the treatment of previously inspected data. This proposal has not been implemented as a new experiment. The full-period reference and original experimental provenance are retained.\n\nSources (repository-relative):\ndocs/research/calibration-selection-audit-2026-10-08.md\nAI_HANDOFF.md';
 s.speakerNotes.text=notes;
}
const candidate=base+'/build/candidate.pptx';await (await PresentationFile.exportPptx(p)).save(candidate);
const final=base+'/deliverables/ESA_research_update_2026-10-13.pptx';
await finalizePresentation({workspaceDir:base,candidatePath:candidate,finalPath:final,pythonExecutable:'/home/k_faz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[2,3,5,10].flatMap(x=>['--require-native-table-slide',String(x)])],requiredNativeTableOwnerSlides:[2,3,5,10],fontPolicy:{basis:'reference',families:['Abyssinica SIL'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:base+'/build/presentation-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<q.slides.items.length;i++){const b=await q.export({slide:q.slides.items[i],format:'png',scale:1});await fs.writeFile(base+`/build/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log('Workbook text edits:',edits);console.log('Completed academic revisions.');
