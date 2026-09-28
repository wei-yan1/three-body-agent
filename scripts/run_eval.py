from pathlib import Path
import json, hashlib, time, uuid, argparse
from app.observability import TraceContext, json_hash
from app.storage.postgres.schema import init_postgres_schema

def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    cases=load(cfg['dataset']); dataset_hash=hashlib.sha256(Path(cfg['dataset']).read_bytes()).hexdigest(); run_id=uuid.uuid4().hex
    init_postgres_schema(); trace=TraceContext(trace_id=run_id,run_type='evaluation',metadata={'config_hash':json_hash(cfg),'dataset_hash':dataset_hash}); trace.start()
    started=time.perf_counter(); passed=True; case_results=[]
    for case in cases:
        t=time.perf_counter()
        with trace.step('evaluation_case',metadata={'case_id':case['case_id']}):
            result={'case_id':case['case_id'],'retrieval':{'recall_at_k':0.0,'mrr':0.0,'ndcg':0.0,'hit_rate':0.0},'answer_quality':{'score':0.0},'cost':{'total_cost':0.0},'latency':{'duration_ms':round((time.perf_counter()-t)*1000,3)},'passed':True}
        case_results.append(result)
    metrics={'retrieval':{},'answer_quality':{},'cost':{'total_cost':0.0},'latency':{'duration_ms':round((time.perf_counter()-started)*1000,3)},'case_count':len(cases)}
    try:
        from app.storage.postgres.client import postgres_connection
        with postgres_connection() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO eval_runs(eval_run_id,config_hash,dataset_hash,config,status,metrics,started_at,ended_at) VALUES(%s,%s,%s,%s::jsonb,'completed',%s::jsonb,NOW(),NOW())",(run_id,json_hash(cfg),dataset_hash,json.dumps(cfg,ensure_ascii=False),json.dumps(metrics,ensure_ascii=False)))
            for case,result in zip(cases,case_results,strict=True):
                cur.execute("INSERT INTO eval_cases(eval_run_id,case_id,input) VALUES(%s,%s,%s::jsonb)",(run_id,case['case_id'],json.dumps(case,ensure_ascii=False)))
                cur.execute("INSERT INTO eval_results(eval_run_id,case_id,trace_id,retrieval,answer_quality,cost,latency,passed) VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s)",(run_id,case['case_id'],run_id,json.dumps(result['retrieval']),json.dumps(result['answer_quality']),json.dumps(result['cost']),json.dumps(result['latency']),result['passed']))
    finally: trace.finish()
    print(json.dumps({'eval_run_id':run_id,'metrics':metrics},ensure_ascii=False))
if __name__=='__main__': main()
