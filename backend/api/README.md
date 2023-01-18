# API Server  
FastAPI를 이용한 API 서버입니다.

## Getting Started  
### Python requirements  
`python`: 3.7 이상, 3.10 미만이 필요합니다.  
`가상환경`: 사전에 Poetry 설치가 필요합니다. 동일한 환경설정을 위해 poetry 사용이 권장됩니다.  

### Installation  
1. 가상 환경 설정  
    - Poetry
        ```shell
        backend/api> poetry shell
        backend/api> poetry install  
        ```
    - Virtualenv  
        ```shell
        backend/api> python -m virtualenv $가상환경이름
        backend/api> source $가상환경이름/bin/activate
        backend/api> pip install -r requirements.txt
        ```
2. config.yaml 값 설정
3. 서버 실행  - `uvicorn main:app`