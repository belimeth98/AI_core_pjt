import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.disclosure_parser import normalize_xml, normalized_text, parse_report_archive, render_table
from src.collection import choose_report, CollectionError


def test_bare_ampersands_and_literal_brackets_survive():
    root, fixes = normalize_xml(b'<DOCUMENT><P>R&D &amp; A <new> item</P></DOCUMENT>')
    assert normalized_text(root) == 'R&D & A <new> item'
    assert fixes['bare_ampersands_escaped'] == 1


def test_malformed_display_attribute_does_not_drop_text():
    root, _ = normalize_xml(b'<DOCUMENT><P ENG=""Snow Corporation">Snow Corporation</P></DOCUMENT>')
    assert normalized_text(root) == 'Snow Corporation'


def test_rowspan_colspan_and_negative_values_preserved():
    root, _ = normalize_xml(b'<TABLE><TR><TH ROWSPAN="2">item</TH><TH COLSPAN="2">2025</TH></TR><TR><TD>KRW millions</TD><TD>(1,234)</TD></TR></TABLE>')
    text, rows = render_table(root)
    assert rows == 2
    assert 'item | 2025 | 2025' in text
    assert 'item | KRW millions | (1,234)' in text


def fixture_xml(corp='00126380'):
    text = '2025년 사업의 개요 원문입니다. ' * 900
    return (f'<DOCUMENT><DOCUMENT-NAME ACODE="11011">사업보고서</DOCUMENT-NAME>'
            f'<COMPANY-NAME AREGCIK="{corp}">기업</COMPANY-NAME><BODY>'
            '<COVER><P>2025년 01월 01일부터 2025년 12월 31일까지</P></COVER>'
            f'<SECTION-1><TITLE>II. 사업의 내용</TITLE><SECTION-2><TITLE>1. 사업의 개요</TITLE><P>{text}</P>'
            '</SECTION-2></SECTION-1></BODY></DOCUMENT>').encode()


def test_exact_main_member_identified_and_all_text_retained():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / '20260310002820.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('20260310002820_00760.xml', b'<DOCUMENT><DOCUMENT-NAME ACODE="00760">audit</DOCUMENT-NAME></DOCUMENT>')
            z.writestr('20260310002820.xml', fixture_xml())
        report = {'corp_code':'00126380','receipt_no':'20260310002820','report_period':'2025.12'}
        parsed, text = parse_report_archive(path, report, Path(tmp))
        assert parsed['main_xml_member'] == '20260310002820.xml'
        assert parsed['section_count'] == 2
        assert text.count('2025년 사업의 개요 원문입니다.') == 900


def test_wrong_company_rejected():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / '20260310002820.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('20260310002820.xml', fixture_xml('99999999'))
        with pytest.raises(ValueError, match='main_document_not_uniquely_identified'):
            parse_report_archive(path, {'corp_code':'00126380','receipt_no':'20260310002820','report_period':'2025.12'}, Path(tmp))


@pytest.mark.parametrize("kind,code,period", [("분기보고서", "11013", "2025.03"), ("반기보고서", "11012", "2025.06"), ("분기보고서", "11014", "2025.09")])
def test_periodic_report_xml_document_code(kind, code, period, tmp_path):
    path = tmp_path / "20260310002820.zip"
    xml = fixture_xml().replace(b'ACODE="11011"', f'ACODE="{code}"'.encode()).replace(
        "2025년 12월 31일까지".encode(), f"2025년 {period[5:]}월 30일까지".encode())
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("20260310002820.xml", xml)
    report = {"corp_code": "00126380", "receipt_no": "20260310002820", "report_period": period, "report_name": kind}
    parsed, text = parse_report_archive(path, report, tmp_path)
    assert parsed["report_period"] == period and "사업의 개요" in text


def test_quarterly_report_with_wrong_cover_period_is_rejected(tmp_path):
    path = tmp_path / "20260310002820.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("20260310002820.xml", fixture_xml().replace(b'ACODE="11011"', b'ACODE="11013"'))
    report = {"corp_code": "00126380", "receipt_no": "20260310002820", "report_period": "2025.06", "report_name": "분기보고서"}
    with pytest.raises(ValueError, match="target_period_not_found"):
        parse_report_archive(path, report, tmp_path)


def test_report_selection_keeps_latest_revision_and_period():
    cached = {'collected_at':'test','reports':[
        {'corp_code':'00126380','rcept_no':'20260310000001','rcept_dt':'20260310','report_nm':'사업보고서 (2025.12)'},
        {'corp_code':'00126380','rcept_no':'20260619000001','rcept_dt':'20260619','report_nm':'[기재정정]사업보고서 (2025.12)'},
        {'corp_code':'00126380','rcept_no':'20260719000001','rcept_dt':'20260719','report_nm':'사업보고서 (2024.12)'},
    ]}
    selected = choose_report(cached, {'corp_code':'00126380','corp_name':'기업'}, '2025.12')
    assert selected['receipt_no'] == '20260619000001'
    assert '[기재정정]' in selected['report_name']


def test_missing_period_does_not_fall_back_to_another_year():
    with pytest.raises(CollectionError):
        choose_report({'collected_at':'test','reports':[]}, {'corp_code':'00126380','corp_name':'기업'}, '2025.12')
