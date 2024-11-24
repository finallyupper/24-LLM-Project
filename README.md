# 24-LLM-Project
24-Fall-NLP LLM project

### Additional Notice
MMLUpro의 경우 Law, Psychology, Business, Philosophy, History 다섯 개의 domain만 사용

#### Configurations 
`config.yaml` 에서 data_root: 데이터 폴더의 경로, *_faiss_path는 FAISS 데이터베이스를 저장할 로컬 위치
top_k는 상위 몇개까지 retriever가 뽑을지를 나타내는 하이퍼파라미터. 그외 chunk_size, chunk_overlap
- data_root, ewha_faiss_path, arc_faiss_path는 본인 절대경로 기준으로 수정이 필요함.
- NOTE: 레포내에 .env 파일을 만들어 UPSTAGE_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT, LANGCHAIN_API_KEY, ROOT를 반드시 입력해두어야함.

#### How to play
커맨드에 아래를 실행하면 데이터베이스를 불러와 chain을 정의하고 questions를 테스트해보고 정확도까지 출력해볼 수 있음.
```
python main.py
```

##### main_multivec.py
- `config.yaml`에 아래 4줄 추가
    ```
    summ_chroma_path: 24-LLM-Project/db/summ_chroma
    summ_faiss_path: 24-LLM-Project/db/summ_faiss
    pc_chroma_path: 24-LLM-Project/db/pc_chroma
    pc_faiss_path: 24-LLM-Project/db/pc_faiss
    ```
    
- command examples
    - pc: parent-child
    - summ: summary
    ```
    python main_multivec.py -v faiss -l pc
    python main_multivec.py -v faiss -l summ
    python main_multivec.py -v chroma -l pc
    python main_multivec.py -v chroma -l summ
    ```

#### 호출 함수 관련
- langchain과 관련된 핵심 메소드들은 `langchain_engine.py`에 정의되어 있음.
- `utils.py` 에는 그 외에 필요한 함수들이 있음(ex. read data/yaml 등)
- `dataset.py`는 수진님 것과 동일.
- `prompts.py`는 일단 프롬프트 기록용으로 만들어준 의미없는 파일.

### API Key 관리
1. `.env`파일 생성후 아래 내용 작성 (gitignore에 의해 add되지 않을것)
```
API_KEY='YOUR API KEY'
```
2. `.gitignore` 파일 생성 및 아래 내용 저장
```
.env
```
3. 파일에서 API KEY 부를때
```
from dotenv import load_dotenv 
import os 
load_dotenv(".env") # env파일 경로 
UPSTAGE_API_KEY = os.getenv("API_KEY") # 할당완료
```


### References
- Prompt Engineering/ Langchain
    - https://platform.openai.com/docs/guides/prompt-engineering
    - https://docs.anthropic.com/claude/docs/intro-to-prompting
    - https://smith.langchain.com/hub
    - https://python.langchain.com/docs/integrations/chat/upstage/
- Upstage API
    - https://console.upstage.ai/docs/capabilities/embeddings
- wikidocs
    - https://wikidocs.net/book/14314  (upstage)
