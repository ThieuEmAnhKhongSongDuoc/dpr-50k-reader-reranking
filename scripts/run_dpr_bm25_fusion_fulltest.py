from __future__ import annotations
import argparse, base64, json, pickle, subprocess, sys, time, unicodedata
from pathlib import Path
import hydra, numpy as np, torch
from hydra import compose, initialize_config_dir
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpr.utils.tokenizers import SimpleTokenizer
from dpr.indexer.faiss_indexers import DenseFlatIndexer
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu
from dpr.utils.model_utils import get_model_obj, load_states_from_checkpoint, setup_for_distributed_mode
from evaluation_utils import answer_match_type, normalize_wiki_passage_id, passage_has_answer, write_jsonl
JAVA=Path(r'D:\DPR-data\temurin8\jdk8u504-b01\bin')
JARS=Path(r'D:\DPR-data\lucene-8.11.2')
LUCENE_INDEX=Path(r'D:\DPR-data\lucene_bm25_fulltest\lucene_index')
LUCENE_WORK=Path(r'D:\DPR-data\lucene_bm25_fulltest')
JAVA_SOURCE=ROOT/'scripts'/'lucene_bm25'/'LuceneBm25Scores.java'
CLASS_DIR=LUCENE_WORK/'score_classes'
KEYS=['nq_test','trivia_test','webq_test','curatedtrec_test','squad1_test']
def b64(s): return base64.b64encode(s.encode('utf-8')).decode('ascii')
def norm(s,tok):
    return ' '+' '.join(tok.tokenize(unicodedata.normalize('NFD',s)).words(uncased=True))+' '
def encode(encoder,tensorizer,questions,batch_size,device):
    out=[]; encoder.eval()
    with torch.no_grad():
      for i in range(0,len(questions),batch_size):
        ids=[tensorizer.text_to_tensor(q) for q in questions[i:i+batch_size]]
        batch=torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=tensorizer.get_pad_id()).to(device)
        seg=torch.zeros_like(batch); att=tensorizer.get_attn_mask(batch)
        _,v,_=encoder(batch,seg,att); out.append(v.detach().cpu().numpy())
        if (i//batch_size+1)%100==0: print(f'encoded queries {min(i+batch_size,len(questions))}/{len(questions)}',flush=True)
    return np.concatenate(out)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--encoded',type=Path,required=True)
    ap.add_argument('--result',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--candidate-k',type=int,default=2000); ap.add_argument('--lambda-dpr',type=float,default=1.1)
    ap.add_argument('--datasets',nargs='+',choices=KEYS,default=KEYS,
                    help='Datasets to evaluate; supports a focused rerun such as curatedtrec_test')
    args=ap.parse_args()
    required_paths=[args.checkpoint,args.encoded,JAVA/'javac.exe',JAVA/'java.exe',JARS/'lucene-core-8.11.2.jar',JARS/'lucene-analyzers-common-8.11.2.jar']
    missing=[p for p in required_paths if not p.exists()]
    if missing: raise FileNotFoundError(missing[0])
    if not any(LUCENE_INDEX.glob('segments_*')):
      raise FileNotFoundError(f'No Lucene segment manifest under {LUCENE_INDEX}')
    CLASS_DIR.mkdir(parents=True,exist_ok=True)
    cp=';'.join([str(JARS/'lucene-core-8.11.2.jar'),str(JARS/'lucene-analyzers-common-8.11.2.jar')])
    subprocess.run([str(JAVA/'javac.exe'),'-encoding','UTF-8','-cp',cp,'-d',str(CLASS_DIR),str(JAVA_SOURCE)],check=True)
    runtime_cp=';'.join([str(CLASS_DIR),cp])
    with initialize_config_dir(config_dir=str(ROOT/'conf'),version_base='1.1'):
      cfg=compose(config_name='dense_retriever',overrides=['datasets=local_qa_tests','ctx_sources=local_wiki50k','hydra.job.chdir=False',f'model_file={args.checkpoint.as_posix()}'])
    cfg=setup_cfg_gpu(cfg); state=load_states_from_checkpoint(str(args.checkpoint)); set_cfg_params_from_state(state.encoder_params,cfg)
    tensorizer,biencoder,_=init_biencoder_components(cfg.encoder.encoder_model_type,cfg,inference_only=True); biencoder.load_state(state,strict=False)
    qenc,_=setup_for_distributed_mode(biencoder.question_model,None,cfg.device,cfg.n_gpu,cfg.local_rank,cfg.fp16); qenc.eval()
    ctx=hydra.utils.instantiate(cfg.ctx_sources.wiki50k); passages={}; ctx.load_data_to(passages)
    tok=SimpleTokenizer()
    index=DenseFlatIndexer(buffer_size=50000); index.init_index(get_model_obj(qenc).get_out_size())
    with args.encoded.open('rb') as f: encoded=pickle.load(f)
    index.index_data(encoded)
    results={}; per_query=[]
    class_file=CLASS_DIR/'LuceneBm25Scores.class'
    if not class_file.exists(): raise RuntimeError(f'Missing compiled class: {class_file}')
    for key in args.datasets:
      qa=hydra.utils.instantiate(cfg.datasets[key]); qa.load_data(); qs=[x.query for x in qa]; ans=[x.answers for x in qa]
      query_file=LUCENE_WORK/f'{key}_queries_b64.tsv'; hits_file=LUCENE_WORK/f'{key}_bm25_top{args.candidate_k}_scores.tsv'
      with query_file.open('w',encoding='utf-8',newline='\n') as f:
        for i,q in enumerate(qs): f.write(f'{i}\t{b64(q)}\n')
      java_out=subprocess.run([str(JAVA/'java.exe'),'-Xmx2g','-cp',runtime_cp,'LuceneBm25Scores','search',str(query_file),str(LUCENE_INDEX),str(hits_file),str(args.candidate_k)],check=True,capture_output=True,text=True)
      print(java_out.stdout.strip(),flush=True)
      qvec=encode(qenc,tensorizer,qs,args.batch_size,cfg.device)
      dpr_scores,dpr_idx=index.index.search(qvec,args.candidate_k)
      cutoff={20:0,100:0}; current=None; bm={}; row_iter=iter(hits_file.open('r',encoding='utf-8'))
      line=next(row_iter,None)
      for qi in range(len(qs)):
        bm={}
        while line is not None:
          fields=line.rstrip('\n').split('\t')
          rowid=int(fields[0])
          if rowid != qi: break
          bm[normalize_wiki_passage_id(fields[2])]=float(fields[3]); line=next(row_iter,None)
        dense={index.index_id_to_db_id[int(j)]:float(sc) for j,sc in zip(dpr_idx[qi],dpr_scores[qi]) if int(j)>=0}
        dense={normalize_wiki_passage_id(pid):score for pid,score in dense.items()}
        candidates=set(bm)|set(dense)
        # Break equal-score ties by passage ID so runs do not depend on set iteration order.
        ranked=sorted(candidates,key=lambda pid:(-(bm.get(pid,0.0)+args.lambda_dpr*dense.get(pid,0.0)),pid))[:100]
        match_type=answer_match_type(key)
        candidate_rows=[]; first=None
        for rank,pid in enumerate(ranked,start=1):
          answer_hit=passage_has_answer(ans[qi],passages[pid].text,tok,key)
          candidate_rows.append({'rank':rank,'passage_id':pid,'bm25_score':bm.get(pid,0.0),'dpr_score':dense.get(pid,0.0),'fused_score':bm.get(pid,0.0)+args.lambda_dpr*dense.get(pid,0.0),'answer_hit':answer_hit})
          if answer_hit and first is None: first=rank
        if first is not None:
          for k in cutoff:
            if first<=k: cutoff[k]+=1
        per_query.append({'query_id':f'{key}:{qi}','query_index':qi,'dataset':key,'question':qs[qi],'answers':[str(a) for a in ans[qi]],'match_type':match_type,'first_answer_rank':first,'candidates':candidate_rows})
        if qi and qi%1000==0: print(f'{key}: fused {qi}/{len(qs)}',flush=True)
      results[key]={'questions':len(qs),'candidate_k_each':args.candidate_k,'lambda_dpr':args.lambda_dpr,'hits':cutoff,'top_k_accuracy':{str(k):cutoff[k]/len(qs) for k in cutoff}}
      print(key,results[key],flush=True)
      args.result.parent.mkdir(parents=True,exist_ok=True)
      per_query_path=args.result.with_name(args.result.stem+'_per_query')/'all_datasets.jsonl.gz'
      write_jsonl(per_query_path,per_query)
      args.result.write_text(json.dumps({'retriever':'BM25+DPR score fusion','bm25':'Apache Lucene 8.11.2 BM25Similarity k1=.9 b=.4; EnglishAnalyzer','dpr_checkpoint':str(args.checkpoint),'corpus_size':index.index.ntotal,'bm25_candidate_k':args.candidate_k,'dpr_candidate_k':args.candidate_k,'fusion':'BM25(q,p) + lambda_dpr * DPR_dot_product(q,p)','lambda_dpr':args.lambda_dpr,'id_normalization':'Lucene numeric IDs are normalized to the DPR wiki:<id> format before union and lookup.','test_results':results,'per_query_results':str(per_query_path)},indent=2),encoding='utf-8')
if __name__=='__main__': main()
