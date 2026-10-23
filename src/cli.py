"""명령행 오류를 인증 정보 없이 표시하고 실패 종료 코드를 반환합니다."""
from zipfile import BadZipFile
from openai import APIError
from pydantic import ValidationError
from requests import RequestException
from src.collection import CollectionError


def run_cli(main):
    try:
        return main() or 0
    except APIError as error:
        print(f"GMS 요청 실패: {type(error).__name__}, HTTP {getattr(error, 'status_code', None)}")
    except CollectionError as error:
        print(f"DART 처리 실패: {error.stage}, 상태: {error.code}")
    except RequestException:
        print("DART 요청에 실패했습니다. 연결 상태와 설정을 확인하세요.")
    except ValidationError:
        print("모델 응답이 정해진 형식과 다릅니다.")
    except BadZipFile:
        print("ZIP 원문 파일이 손상되어 있습니다.")
    except (OSError, KeyError, TypeError):
        print("필요한 데이터 파일 또는 필드를 확인하세요.")
    except (ValueError, RuntimeError) as error:
        print(f"실행 중단: {error}")
    return 1
