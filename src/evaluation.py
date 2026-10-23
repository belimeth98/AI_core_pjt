"""동일 모델·질문의 RAG 전후 응답을 기록하는 선택 심화 F2015 평가."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from src.common import ROOT, digest, read_json, write_json, utc_now
from src.llm import chat_completion
from src.pipeline import run_pipeline
from src.retrieval import search_company
from src.answering import prepare_sources, generate_answer, check_citations


CASES = [
    ('branches', '01133217', '카카오뱅크는 고객이 방문하는 오프라인 영업점을 운영하나요? 고객센터와 구분해 설명해 주세요.'),
    ('business', '01133217', '카카오뱅크 사업보고서 기준으로 뱅킹 비즈니스와 플랫폼 비즈니스는 어떻게 다른가요?'),
    ('holding', '00382199', '신한금융지주와 신한은행은 같은 회사인가요? 지주회사의 역할과 예금·대출을 제공하는 은행의 역할을 구분해 주세요.'),
    ('divisions', '00126380', '삼성전자 사업보고서에서 DX 부문과 DS 부문은 각각 어떤 사업을 하나요?'),
    ('unknown', '01133217', '카카오뱅크의 2027년 신입 개발자에게 지급할 확정 초봉은 얼마인가요?'),
]

BASELINE_PROMPT = '''너는 재무 초보 취업 준비생에게 기업을 쉽게 설명하는 도우미다.
외부 문서를 제공받지 않은 상태에서 알고 있는 범위만 답한다.
모르는 사실·미래 수치·현재 조건은 추측하지 말고 확인할 수 없다고 말한다.
제공되지 않은 근거 번호나 확인하지 않은 출처를 만들지 않는다.
쉬운 설명, 필요한 용어 풀이, 확인 범위와 한계를 간단히 작성한다.'''


def compare_case(case, output_dir, refresh=False):
    case_id, code, question = case
    company = next(c for c in read_json(ROOT / 'data/companies.json') if c['corp_code'] == code)
    fingerprint = digest({'version': 'rag-comparison-v1', 'case': case,
        'receipt': company['canonical_receipt_no'],
        'chat_model': os.getenv('GMS_CHAT_MODEL'), 'embedding_model': os.getenv('GMS_EMBEDDING_MODEL'),
        'settings': read_json(ROOT / 'config/retrieval.json')})
    path = output_dir / f'{case_id}.json'
    if path.exists() and not refresh:
        saved = read_json(path)
        if saved.get('fingerprint') == fingerprint and saved.get('complete'):
            return {**saved, 'cache_reused': True}
    baseline_start = time.perf_counter()
    baseline = chat_completion([
        {'role': 'system', 'content': BASELINE_PROMPT},
        {'role': 'user', 'content': question},
    ])
    baseline_seconds = time.perf_counter() - baseline_start
    started = time.perf_counter()
    if case_id == 'branches':
        result = run_pipeline(question, code)
        pipeline_executed = True
    else:
        retrieved = search_company(question, code)
        sources = prepare_sources(retrieved)
        answer = generate_answer(question, sources)
        result = {'corp_code': code, 'receipt_no': company['canonical_receipt_no'],
            'question': question, 'answer': answer, 'sources': sources,
            'cited_ids': check_citations(answer, sources), 'questions': []}
        pipeline_executed = False
    rag_seconds = time.perf_counter() - started
    value = {'case_id': case_id, 'complete': True, 'fingerprint': fingerprint,
        'evaluated_at': utc_now(), 'chat_model': os.getenv('GMS_CHAT_MODEL'),
        'embedding_model': os.getenv('GMS_EMBEDDING_MODEL'), 'company': company['corp_name'],
        'question': question, 'baseline': baseline, 'baseline_seconds': baseline_seconds,
        'rag': result, 'rag_seconds': rag_seconds, 'full_pipeline_executed': pipeline_executed,
        'review_status': '사실 정확성은 원문을 대조한 수동 검토 필요',
        'cache_reused': False}
    write_json(path, value)
    print(f'COMPARE {case_id} baseline={baseline_seconds:.1f}s rag={rag_seconds:.1f}s citations={result["cited_ids"]}', flush=True)
    return value


def evaluate_rag(*, refresh=False):
    load_dotenv(ROOT / '.env')
    output = ROOT / 'data/processed/evaluation'
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda case: compare_case(case, output, refresh), CASES))
    value = {'evaluated_at': utc_now(), 'case_count': len(results),
        'comparison_policy': '동일 질문·동일 채팅 모델. baseline은 문서 미제공, RAG는 현행 검색 근거 제공. 온라인 지연은 동시 실행 영향을 받으며 벤치마크가 아님.',
        'results': results}
    write_json(output / 'comparison.json', value)
    return value
