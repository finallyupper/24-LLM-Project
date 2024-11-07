# 24-LLM-Project
24-Fall-NLP LLM project

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