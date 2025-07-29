import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError

# AWS Bedrock 클라이언트 초기화
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client('bedrock-runtime', region_name='us-east-1')
    except Exception as e:
        st.error(f"AWS Bedrock 클라이언트 초기화 실패: {e}")
        return None

def generate_introduction_with_nova(name, major, hobby, experiences, target_job):
    """Nova Lite 모델을 사용하여 자기소개서 생성"""
    bedrock_client = get_bedrock_client()
    
    if not bedrock_client:
        return "❌ AWS Bedrock 연결에 실패했습니다."
    
    prompt = f"""
당신은 전문적인 자기소개서 작성 도우미입니다. 다음 정보를 바탕으로 매력적이고 전문적인 자기소개서를 작성해주세요.

개인 정보:
- 이름: {name}
- 전공: {major}
- 취미: {hobby}
- 경험/활동: {experiences}
- 희망 직무: {target_job}

다음 구조로 자기소개서를 작성해주세요:
1. 인사말 및 자기소개
2. 전공 관련 역량
3. 경험 및 활동
4. 취미를 통한 개성 표현
5. 희망 직무에 대한 열정
6. 마무리

한국어로 작성하며, 진정성 있고 전문적인 톤으로 작성해주세요.
"""

    try:
        # Nova Lite 모델 호출 (올바른 형식)
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",  # Nova Lite 모델 ID
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 1000,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        )
        
        # 응답 파싱
        response_body = json.loads(response.get('body').read())
        
        # Nova 모델의 응답 구조에 맞게 텍스트 추출
        if 'output' in response_body and 'message' in response_body['output']:
            content = response_body['output']['message']['content']
            if content and len(content) > 0 and 'text' in content[0]:
                return content[0]['text']
        
        return '자기소개서 생성에 실패했습니다.'
        
    except ClientError as e:
        return f"❌ AWS API 호출 오류: {e}"
    except Exception as e:
        return f"❌ 자기소개서 생성 중 오류 발생: {e}"

# 앱 제목
st.title("🤖 AI 자기소개서 생성기")

# 입력 필드들
st.write("## 📝 정보 입력")
name = st.text_input("이름", "홍길동")
major = st.text_input("전공", "컴퓨터공학과")
hobby = st.text_input("취미", "독서, 영화감상")
experiences = st.text_area("경험/활동", "프로그래밍 동아리 활동, 웹 개발 프로젝트 참여")
target_job = st.text_input("희망 직무", "소프트웨어 개발자")

# 생성 버튼
if st.button("🤖 자기소개서 생성하기", type="primary"):
    if name and major and hobby and experiences and target_job:
        with st.spinner("생성 중..."):
            generated_intro = generate_introduction_with_nova(
                name, major, hobby, experiences, target_job
            )
        
        st.write("## 📋 생성된 자기소개서")
        st.write(generated_intro)
    else:
        st.error("모든 필드를 입력해 주세요!")

st.write("---")
st.info("💡 AWS Bedrock Nova Lite 모델을 사용한 AI 자기소개서 생성기입니다.")
