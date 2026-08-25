import argparse, hashlib, json, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path
from urllib.request import urlretrieve

p=argparse.ArgumentParser(); p.add_argument('--bundle'); p.add_argument('--output-dim',type=int); p.add_argument('--beam',type=int); p.add_argument('--depth',type=int); p.add_argument('--output'); a=p.parse_args()
bundle=Path(a.bundle); out=Path(a.output); repo=Path('/tmp/paper_multigpu_repo'); run_out=out.parent/'native_output'; run_out.mkdir(parents=True,exist_ok=True)
commit='a1db0e6d9bb5458c8a842b37dfa99572d3025667'
if not repo.exists():
 subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/TryDotAtwo/MultiGPUBeamSearch.git',str(repo)],check=True)
 subprocess.run(['git','fetch','--depth','1','origin',commit],cwd=repo,check=True); subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=repo,check=True)
release_base='https://github.com/TryDotAtwo/MultiGPUBeamSearchPaper/releases/download/v1.0.0'
checkpoint_specs={1:('weights_megaminx2048_512_8_e4000.pth','7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759'),24:('p900-t000-q-sym_1777988767_best.pth','e7bda332b53acc9363edd8ec682a211c1a8b8a315ec26ff8e8928efa5d2ca670')}
checkpoint_name,expected_sha=checkpoint_specs[a.output_dim]; matches=list(Path('/kaggle/input').rglob(checkpoint_name))
if len(matches)>1: raise RuntimeError(f'expected at most one attached native output{a.output_dim} model: {matches}')
if matches: weights=matches[0]
else:
 model_cache=Path('/tmp/paper_checkpoint_assets'); model_cache.mkdir(parents=True,exist_ok=True); weights=model_cache/checkpoint_name
 if not weights.is_file(): urlretrieve(f'{release_base}/{checkpoint_name}',weights)
sha=hashlib.sha256(weights.read_bytes()).hexdigest()
if sha!=expected_sha: raise RuntimeError(f'checkpoint SHA256 mismatch: {sha} != {expected_sha}')
inputs=Path('/kaggle/input'); puzzle_info=next(inputs.rglob('puzzle_info.json')); test_csv=next(inputs.rglob('test.csv'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()[0]
arch='75' if 'T4' in gpu else '60' if 'P100' in gpu else None
if arch is None: raise RuntimeError(f'unsupported GPU {gpu}')
cutlass=Path('/tmp/cutlass'); cutlass_commit="afa1772203677c5118fcd82537a9c8fefbcc7008"; build=Path(f'/tmp/paper_build_sm{arch}'); export=Path(f'/tmp/paper_stream1_weights_out{a.output_dim}')
if not cutlass.exists(): subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/NVIDIA/cutlass.git',str(cutlass)],check=True)
subprocess.run(['git','fetch','--depth','1','origin',cutlass_commit],cwd=cutlass,check=True); subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=cutlass,check=True)
actual_cutlass_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cutlass,text=True).strip()
if actual_cutlass_commit != cutlass_commit: raise RuntimeError(f'CUTLASS commit mismatch: {actual_cutlass_commit}')
manifest_path=export/'manifest.json'
if not manifest_path.is_file():
 if export.exists(): shutil.rmtree(export)
 subprocess.run([sys.executable,str(repo/'tools/export_stream1.py'),'--weights',str(weights),'--out',str(export)],cwd=repo,check=True)
exported=[x for x in export.rglob('*') if x.is_file()]
if not manifest_path.is_file() or any(x.stat().st_size == 0 for x in exported):
 raise RuntimeError(f'incomplete stream1 export for output_dim={a.output_dim}: {[(str(x), x.stat().st_size) for x in exported]}')
runner=build/'production_runner'
if not runner.exists():
 # The master debug switch is required for BEAM_ENABLE_DEPTH_LOGS to take
 # effect. All heavier trace/timing switches remain OFF, so this adds only one
 # host wall-time record per completed depth to the production algorithm.
 subprocess.run(['cmake','-S',str(repo),'-B',str(build),'-GNinja','-DCMAKE_BUILD_TYPE=Release',f'-DBEAM_CUDA_ARCHITECTURES={arch}',f'-DCUTLASS_DIR={cutlass}','-DBEAM_ENABLE_DEBUG=ON','-DBEAM_ENABLE_DEPTH_LOGS=ON'],check=True)
 subprocess.run(['cmake','--build',str(build),'--target','production_runner','-j','2'],check=True)
# Keep the measured high-throughput profiles, but also cover the small-beam
# end of the public sweep.  BEAM_B_MICRO is an inference-row budget: the
# scalar head consumes 24 rows per parent while the Q head consumes one.  A
# Stream3 ring slot then holds parent_batch * 24 candidates.  Consequently the
# old minimums (4096/256) could not fit a beam-1024 logical shard even though
# the GPU was almost empty.  These descending profiles change batching only;
# the requested beam, model head, state, generators, and search algorithm stay
# unchanged.
row_budgets=(
 [49152,32768,24576,16384,8192,4096,2048,1024,512,256,128,64,48,24]
 if a.output_dim==1 else
 [2048,1024,512,256,128,64,42,32,16,8,4,2,1]
)
base_env=os.environ.copy(); base_env.update({'BEAM_WEIGHT_DIR':str(export),'BEAM_PUZZLE_INFO_JSON':str(puzzle_info),'BEAM_GENERATOR_PATH':str(puzzle_info),'BEAM_TEST_CSV':str(test_csv),'BEAM_RUNTIME_CONFIG_MODE':'auto','BEAM_GPU_HEADROOM_BYTES':str(256*1024*1024),'WORLD_SIZE':'1','LOCAL_RANK':'0','RANK':'0'})
attempts=[]; proc=None; selected_row_budget=None
for row_budget in row_budgets:
 env=base_env.copy(); env['BEAM_B_MICRO']=str(row_budget)
 samples=[]; stop=threading.Event()
 def monitor():
  while not stop.is_set():
   try:
    values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    samples.extend(float(x.strip()) for x in values[:1])
   except Exception: pass
   stop.wait(0.2)
 t=threading.Thread(target=monitor,daemon=True); t.start(); started=time.perf_counter()
 current=subprocess.run([str(runner),'0',str(a.depth),str(a.beam),'1','0'],cwd=repo,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 wall=time.perf_counter()-started; stop.set(); t.join()
 low=current.stdout.lower(); retryable=('no runtime config fits gpu/final-layout budget' in low or 'out of memory' in low)
 attempts.append({'stream1_row_budget':row_budget,'returncode':current.returncode,'retryable_profile_failure':retryable,'search_wall_s':wall,'peak_device_used_mib':max(samples) if samples else None})
 print(f'PROFILE_ATTEMPT row_budget={row_budget} returncode={current.returncode}',flush=True); print(current.stdout,flush=True)
 proc=current
 if current.returncode==0:
  selected_row_budget=row_budget; break
 if not retryable: break
status='completed' if proc.returncode==0 else ('cuda_oom' if 'out of memory' in proc.stdout.lower() else 'process_exit')
metric=attempts[-1]
depth_times={int(m.group(1)):float(m.group(2)) for m in re.finditer(r'depth_done=(\d+)\s+depth_sec=([0-9.eE+-]+)',proc.stdout)}
last_depth_index=a.depth-1
row={'status':status,'checkpoint_sha256':sha,'checkpoint_file':checkpoint_name,'observed_output_dim':a.output_dim,'move_count':24,'returncode':proc.returncode,'implementation':'MultiGPUBeamSearch WORLD_SIZE=1','commit':commit,'cutlass_commit':cutlass_commit,'stream1_row_budget':selected_row_budget,'stream1_row_budget_attempts':attempts,'stream1_concurrency':'auto','stream3_ring_slots':'auto','gpu_headroom_bytes':256*1024*1024,'search_wall_s':metric['search_wall_s'],'seconds_per_depth':metric['search_wall_s']/a.depth,'completed_depth_times_s':depth_times,'last_requested_depth_index':last_depth_index,'last_depth_seconds':depth_times.get(last_depth_index),'peak_device_used_mib':metric['peak_device_used_mib'],'peak_device_used_bytes':int(metric['peak_device_used_mib']*1024**2) if metric['peak_device_used_mib'] is not None else None}
out.write_text(json.dumps(row,indent=2),encoding='utf-8'); raise SystemExit(proc.returncode)
