import argparse, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path
from urllib.request import urlretrieve

p=argparse.ArgumentParser(); p.add_argument('--bundle'); p.add_argument('--output-dim',type=int); p.add_argument('--beam',type=int); p.add_argument('--depth',type=int); p.add_argument('--output'); a=p.parse_args()
bundle=Path(a.bundle); out=Path(a.output); runtime=bundle/'pilgrim_runtime'; work=Path('/tmp/paper_pilgrim_runtime')
if work.exists(): shutil.rmtree(work)
shutil.copytree(runtime, work)
release_base='https://github.com/TryDotAtwo/MultiGPUBeamSearchPaper/releases/download/v1.0.0'
checkpoint_specs={1:('weights_megaminx2048_512_8_e4000.pth','7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759'),24:('p900-t000-q-sym_1777988767_best.pth','e7bda332b53acc9363edd8ec682a211c1a8b8a315ec26ff8e8928efa5d2ca670')}
checkpoint_name,expected_sha=checkpoint_specs[a.output_dim]
matches=[x for x in Path('/kaggle/input').rglob(checkpoint_name)]
if len(matches)>1: raise RuntimeError(f'expected at most one attached native output{a.output_dim} model: {matches}')
if matches:
 weights=matches[0]
else:
 model_cache=Path('/tmp/paper_checkpoint_assets'); model_cache.mkdir(parents=True,exist_ok=True)
 weights=model_cache/checkpoint_name
 if not weights.is_file(): urlretrieve(f'{release_base}/{checkpoint_name}',weights)
sha=hashlib.sha256(weights.read_bytes()).hexdigest()
if sha!=expected_sha: raise RuntimeError(f'checkpoint SHA256 mismatch: {sha} != {expected_sha}')
info_path=work/('model_output1.json' if a.output_dim==1 else 'model_output24.json')
cmd=[sys.executable,'test.py','--group_id','900','--target_id','0','--model_info_path',str(info_path),'--weights_path',str(weights),'--best','--states_path',str(work/'paper_state.pt'),'--B',str(a.beam),'--num_steps',str(a.depth),'--num_attempts','1','--tests_num','1','--gpu_ids','0','--eval_batch_size','384','--search_state_dtype','uint8','--cuda_memory_stats']
samples=[]; stop=threading.Event()
def monitor():
 while not stop.is_set():
  try:
   values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
   samples.extend(float(x.strip()) for x in values[:1])
  except Exception: pass
  stop.wait(0.2)
t=threading.Thread(target=monitor,daemon=True); t.start(); started=time.perf_counter()
proc=subprocess.run(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
wall=time.perf_counter()-started; stop.set(); t.join()
print(proc.stdout,flush=True)
status='completed' if proc.returncode==0 else ('cuda_oom' if 'out of memory' in proc.stdout.lower() else 'process_exit')
row={'status':status,'checkpoint_sha256':sha,'checkpoint_file':checkpoint_name,'observed_output_dim':a.output_dim,'move_count':24,'returncode':proc.returncode,'implementation':'Pilgrim native Searcher' if a.output_dim==1 else 'Pilgrim native QSearcher','search_wall_s':wall,'seconds_per_depth':wall/a.depth,'peak_device_used_mib':max(samples) if samples else None,'peak_device_used_bytes':int(max(samples)*1024**2) if samples else None}
out.write_text(json.dumps(row,indent=2),encoding='utf-8')
raise SystemExit(proc.returncode)
