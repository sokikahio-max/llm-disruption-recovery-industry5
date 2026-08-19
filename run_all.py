"""Reproduce the deterministic analyses; optionally rerun the 60-call LLM study."""
from __future__ import annotations
import argparse, os
import run_operational_experiments
import statistical_analysis
import prepare_explanation_study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-llm', action='store_true', help='Rerun the 60 OpenAI calls; requires OPENAI_API_KEY and incurs API cost.')
    args=parser.parse_args()
    run_operational_experiments.run()
    statistical_analysis.run()
    prepare_explanation_study.prepare()
    if args.with_llm:
        if not os.getenv('OPENAI_API_KEY'):
            raise RuntimeError('OPENAI_API_KEY is required with --with-llm')
        import run_llm_explanation_study
        run_llm_explanation_study.main()
    print('Reproducibility pipeline completed.')

if __name__=='__main__': main()
