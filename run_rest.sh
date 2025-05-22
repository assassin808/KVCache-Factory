bash scripts/scripts_longBench/eval.sh 512 0.3 samsum,lcc
bash scripts/scripts_longBench/eval.sh 1024 0.3 gov_report,qmsum,multi_news,samsum,lcc
bash scripts/scripts_longBench/eval.sh 2048 0.3 gov_report,qmsum,multi_news,samsum,lcc
bash scripts/scripts_longBench/eval.sh 256 0.5 qmsum,multi_news,samsum,lcc
bash scripts/scripts_longBench/eval.sh 256 0.7 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval.sh 1024 0.7 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,trec,triviaqa,passage_count,passage_retrieval_en,repobench-p
bash scripts/scripts_longBench/eval.sh 2048 0.7 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,trec,triviaqa,passage_count,passage_retrieval_en,repobench-p
bash scripts/scripts_longBench/eval_head.sh 2048 0.7 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval_head.sh 256 0.7 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval_head.sh 512 0.7 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval_head.sh 1024 0.7 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p

