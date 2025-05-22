
bash scripts/scripts_longBench/eval.sh adakv 1024 0.5 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval.sh adakv 2048 0.5 narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p

bash scripts/scripts_longBench/eval.sh PyramidKV 256 0.5 musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval.sh PyramidKV 512 0.5 musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval.sh PyramidKV 1024 0.5 musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
bash scripts/scripts_longBench/eval.sh PyramidKV 2048 0.5 musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p


bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/0e9e39f249a16976918f6564b8830bc894c89659_1024/
bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/0e9e39f249a16976918f6564b8830bc894c89659_2048/
bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/0e9e39f249a16976918f6564b8830bc894c89659_256/
bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/0e9e39f249a16976918f6564b8830bc894c89659_512/
bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/_1024/
bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/_2048/
bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/_256/
bash scripts/scripts_longBench/metrics.sh results_long_bench0.5/_512/

bash scripts/scripts_longBench/eval.sh PyramidKV 512 0.5 qmsum,multi_news,samsum;
bash scripts/scripts_longBench/eval.sh PyramidKV 1024 0.5 musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p;
bash scripts/scripts_longBench/eval.sh PyramidKV 2048 0.5 musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p;