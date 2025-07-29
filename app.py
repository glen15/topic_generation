import streamlit as st
import boto3
import json
import os
from datetime import datetime
from botocore.exceptions import ClientError

# AWS Bedrock 클라이언트 초기화
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client('bedrock-runtime', region_name='us-east-1')
    except Exception as e:
        st.error(f"AWS Bedrock 클라이언트 초기화 실패: {e}")
        return None

def generate_hackathon_idea_with_nova(problem_area, target_problem, ai_technology, target_users, expected_impact, idea_length="보통", debug_mode=False):
    """Nova Lite 모델을 사용하여 리빙랩 해커톤 아이디어 생성"""
    bedrock_client = get_bedrock_client()
    
    if not bedrock_client:
        return "❌ AWS Bedrock 연결에 실패했습니다."
    
    # 길이 옵션에 따른 설정 (한국어 특성 고려하여 토큰 수 증가)
    length_settings = {
        "간단": {
            "char_limit": 800,
            "max_tokens": 1000,  # 800자 × 1.25 (한국어 여유분)
            "sections": {
                "title": "10자 이내",
                "overview": "80자 이내",
                "problem": "60자 이내", 
                "ai_tech": "80자 이내",
                "users": "40자 이내",
                "features": "120자 이내, 3개 기능",
                "impact": "80자 이내",
                "tech_stack": "60자 이내",
                "test_plan": "60자 이내",
                "expansion": "60자 이내"
            }
        },
        "보통": {
            "char_limit": 1500,
            "max_tokens": 2000,  # 1500자 × 1.33 (한국어 여유분)
            "sections": {
                "title": "20자 이내",
                "overview": "150자 이내",
                "problem": "100자 이내",
                "ai_tech": "150자 이내",
                "users": "80자 이내", 
                "features": "200자 이내, 3-4개 기능",
                "impact": "150자 이내",
                "tech_stack": "100자 이내",
                "test_plan": "120자 이내",
                "expansion": "100자 이내"
            }
        },
        "상세": {
            "char_limit": 2500,
            "max_tokens": 3200,  # 2500자 × 1.28 (한국어 여유분)
            "sections": {
                "title": "30자 이내",
                "overview": "250자 이내",
                "problem": "200자 이내",
                "ai_tech": "300자 이내",
                "users": "150자 이내",
                "features": "400자 이내, 4-5개 기능",
                "impact": "250자 이내",
                "tech_stack": "200자 이내",
                "test_plan": "200자 이내",
                "expansion": "200자 이내"
            }
        }
    }
    
    settings = length_settings[idea_length]
    sections = settings["sections"]
    
    prompt = f"""
당신은 지속가능한 세상을 위한 리빙랩 해커톤의 전문 멘토입니다. 다음 정보를 바탕으로 창의적이고 실현 가능한 해커톤 아이디어를 체계적으로 정리해주세요.

입력 정보:
- 문제 영역: {problem_area}
- 해결하고자 하는 문제: {target_problem}
- 활용할 AI 기술: {ai_technology}
- 타겟 사용자: {target_users}
- 기대 효과: {expected_impact}

**중요**: 각 섹션은 간결하고 핵심적인 내용으로 작성해주세요. 전체 응답은 {settings["char_limit"]}자 이내로 제한합니다.

다음 구조로 해커톤 아이디어를 정리해주세요:

## 🎯 프로젝트 제목
({sections["title"]}의 창의적이고 임팩트 있는 프로젝트명)

## 📋 프로젝트 개요 ({sections["overview"]})
프로젝트의 핵심 내용과 목적을 간단명료하게 설명

## 🌍 해결 문제 ({sections["problem"]})
구체적인 문제 정의와 현재 상황을 설명

## 🤖 AI 기술 활용 ({sections["ai_tech"]})
어떤 AI 기술을 어떻게 활용할지 구체적으로 설명

## 👥 타겟 사용자 ({sections["users"]})
주요 사용자와 이해관계자를 나열

## 💡 핵심 기능 ({sections["features"]})
주요 기능을 간단한 문장으로 나열
- 기능 1: (한 줄 설명)
- 기능 2: (한 줄 설명)
- 기능 3: (한 줄 설명)

## 🎊 기대 효과 ({sections["impact"]})
지속가능성 측면에서의 기대효과를 구체적 수치나 결과로 설명

## 🛠️ 기술 스택 ({sections["tech_stack"]})
개발에 필요한 핵심 기술들을 나열

## 📊 실증 계획 ({sections["test_plan"]})
실제 환경에서의 테스트 방법을 설명

## 🚀 확장 가능성 ({sections["expansion"]})
향후 발전 방향을 설명

한국어로 작성하며, 각 섹션은 지정된 글자 수를 엄격히 준수해주세요. 실현 가능하면서도 혁신적인 아이디어로 구성해주세요.
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
                    "maxTokens": settings["max_tokens"],  # 길이 옵션에 따라 동적 설정
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        )
        
        # 응답 파싱
        response_body = json.loads(response.get('body').read())
        
        # 디버깅: 전체 응답 구조 확인 (디버깅 모드일 때만)
        if debug_mode:
            st.write("### 🔍 디버깅 정보")
            st.write("**응답 구조:**", response_body.keys())
        
        # Nova 모델의 응답 구조에 맞게 텍스트 추출
        if 'output' in response_body and 'message' in response_body['output']:
            content = response_body['output']['message']['content']
            if content and len(content) > 0 and 'text' in content[0]:
                generated_text = content[0]['text']
                
                if debug_mode:
                    # 디버깅 정보
                    st.write(f"**생성된 텍스트 길이:** {len(generated_text)}자")
                    st.write(f"**요청한 최대 토큰:** {settings['max_tokens']}")
                    st.write(f"**목표 글자 수:** {settings['char_limit']}자")
                    
                    # 토큰 사용량 확인 (있는 경우)
                    if 'usage' in response_body:
                        usage = response_body['usage']
                        st.write(f"**실제 사용 토큰:** {usage}")
                
                # 응답이 완료되었는지 확인 (항상 체크하지만 메시지는 조건부)
                if 'stopReason' in response_body.get('output', {}):
                    stop_reason = response_body['output']['stopReason']
                    
                    if debug_mode:
                        st.write(f"**응답 종료 이유:** {stop_reason}")
                    
                    if stop_reason == 'max_tokens':
                        if debug_mode:
                            st.warning("⚠️ 토큰 제한으로 인해 응답이 잘렸습니다!")
                        
                        # 자동 재시도 (토큰 수 증가) - 디버깅 모드와 관계없이 실행
                        if settings["max_tokens"] < 4000:  # 최대 4000 토큰까지
                            if debug_mode:
                                st.info("🔄 더 긴 응답을 위해 자동 재시도합니다...")
                            
                            increased_tokens = min(settings["max_tokens"] + 1000, 4000)
                            
                            # 재시도 요청
                            retry_response = bedrock_client.invoke_model(
                                modelId="amazon.nova-lite-v1:0",
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
                                        "maxTokens": increased_tokens,
                                        "temperature": 0.7,
                                        "topP": 0.9
                                    }
                                })
                            )
                            
                            retry_body = json.loads(retry_response.get('body').read())
                            if 'output' in retry_body and 'message' in retry_body['output']:
                                retry_content = retry_body['output']['message']['content']
                                if retry_content and len(retry_content) > 0 and 'text' in retry_content[0]:
                                    retry_text = retry_content[0]['text']
                                    if debug_mode:
                                        st.write(f"**재시도 결과 길이:** {len(retry_text)}자")
                                    return retry_text
                    
                    elif stop_reason == 'end_turn' and debug_mode:
                        st.success("✅ 응답이 정상적으로 완료되었습니다.")
                
                return generated_text
        
        return '해커톤 아이디어 생성에 실패했습니다.'
        
    except ClientError as e:
        return f"❌ AWS API 호출 오류: {e}"
    except Exception as e:
        return f"❌ 해커톤 아이디어 생성 중 오류 발생: {e}"

def generate_streamlit_prd(idea_content):
    """Nova Lite 모델을 사용하여 간단한 Streamlit 앱 PRD 생성"""
    bedrock_client = get_bedrock_client()
    
    if not bedrock_client:
        return "❌ AWS Bedrock 연결에 실패했습니다."
    
    prompt = f"""
다음 해커톤 아이디어를 바탕으로 **초기 MVP(Minimum Viable Product)** 버전의 Streamlit 앱 구현을 위한 간단한 PRD를 Markdown 형식으로 작성해주세요.

해커톤 아이디어:
{idea_content}

**중요 제약사항:**
- 이것은 초기 MVP 버전이므로 핵심 기능만 포함
- AI 기능은 **텍스트 처리만으로 제한** (이미지, 음성, 영상 처리 제외)
- 복잡한 AI 모델보다는 간단한 텍스트 분석, 분류, 요약 등에 집중
- 실제 1-2일 내에 구현 가능한 범위로 한정

다음 구조로 간단하고 실용적인 PRD를 작성해주세요:

# 프로젝트명 (MVP 버전)

## 📋 프로젝트 개요
- **목적**: (한 줄 설명)
- **타겟 사용자**: (주요 사용자)
- **MVP 범위**: 텍스트 기반 AI 기능만 포함

## 🎯 주요 기능 (MVP 핵심)
### 필수 기능 (텍스트 처리 한정)
1. 기능 1 - 텍스트 입력 및 처리
2. 기능 2 - 텍스트 분석/분류/요약 등
3. 기능 3 - 결과 표시 및 피드백

### 제외 기능 (향후 버전)
- 이미지/음성/영상 처리
- 복잡한 머신러닝 모델
- 실시간 스트리밍

## 📱 Streamlit 앱 구성
### 화면 구성
- **메인 페이지**: 텍스트 입력 및 설정
- **결과 페이지**: 처리 결과 및 분석

### 사용할 Streamlit 컴포넌트
- 입력: `st.text_input()`, `st.text_area()`, `st.selectbox()`
- 출력: `st.write()`, `st.markdown()`, `st.dataframe()`
- 상호작용: `st.button()`, `st.tabs()`, `st.expander()`

## 💾 데이터 처리 (텍스트만)
- **입력 데이터**: 사용자 텍스트 입력
- **처리 과정**: 간단한 텍스트 분석/처리 알고리즘
- **출력 형태**: 텍스트 결과, 차트, 표 형태

## 🤖 AI 기능 (텍스트 한정)
- **사용 모델**: 간단한 텍스트 처리 라이브러리 또는 API
- **처리 범위**: 텍스트 분류, 키워드 추출, 감정 분석, 요약 등
- **제외 항목**: 이미지/음성/영상 AI 기능

## 📚 필요한 라이브러리
```python
streamlit
pandas
matplotlib (또는 plotly)
requests (API 사용시)
nltk 또는 spacy (텍스트 처리)
openai 또는 transformers (텍스트 AI, 선택적)
```

## ⚡ 구현 순서 (MVP 기준)
1. **1단계**: 기본 UI 및 텍스트 입력 구성
2. **2단계**: 핵심 텍스트 처리 기능 구현
3. **3단계**: 결과 표시 및 기본 개선

## 🚀 향후 확장 계획
- 2차 버전: 이미지 처리 기능 추가
- 3차 버전: 더 복잡한 AI 모델 적용

**MVP 버전으로 간단하고 실용적으로 작성하되, 텍스트 기반 AI 기능만 포함하여 실제 구현 가능한 내용으로 해주세요.**
"""

    try:
        # Nova Lite 모델 호출
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
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
                    "maxTokens": 1500,  # 간단한 PRD용으로 토큰 수 줄임
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
        
        return 'PRD 생성에 실패했습니다.'
        
    except ClientError as e:
        return f"❌ AWS API 호출 오류: {e}"
    except Exception as e:
        return f"❌ PRD 생성 중 오류 발생: {e}"

def save_prd_to_markdown(prd_content, filename=None):
    """PRD 내용을 Markdown 파일로 저장"""
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"streamlit_app_prd_{timestamp}.md"
        
        # 현재 디렉토리에 저장
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prd_content)
        
        return True, filepath
    except Exception as e:
        return False, str(e)

# 앱 제목
st.title("🌱 AI × 지속가능성 리빙랩 해커톤 아이디어 생성기")

# 탭 생성
tab1, tab2 = st.tabs(["💡 아이디어 생성", "📋 PRD 생성"])

with tab1:
    # 입력 필드들
    st.write("## 💡 아이디어 핵심 요소 입력")

    problem_area = st.selectbox(
        "🌍 문제 영역",
        ["환경 보호", "에너지 효율", "폐기물 관리", "지속가능한 농업", "스마트 시티", "기후 변화 대응", "순환 경제", "친환경 교통", "수자원 관리", "생물 다양성 보전", "기타"]
    )

    target_problem = st.text_area(
        "🎯 해결하고자 하는 구체적인 문제",
        "예: 음식물 쓰레기 증가로 인한 환경 오염과 자원 낭비 문제"
    )

    ai_technology = st.text_input(
        "🤖 활용할 AI 기술",
        "예: 컴퓨터 비전, 자연어 처리, 머신러닝 예측 모델"
    )

    target_users = st.text_input(
        "👥 타겟 사용자",
        "예: 일반 가정, 식당 운영자, 지자체"
    )

    expected_impact = st.text_area(
    "🎊 기대하는 지속가능성 효과",
    "예: 음식물 쓰레기 30% 감소, CO2 배출량 저감, 자원 순환 촉진"
    )

    st.write("---")

    # 옵션 설정
    col1, col2 = st.columns([3, 1])

    with col1:
        # 아이디어 길이 선택
        idea_length = st.selectbox(
        "📏 아이디어 생성 범위",
        ["간단", "보통", "상세"],
        index=1,  # 기본값: 보통
    )

    with col2:
        # 디버깅 모드 체크박스
        debug_mode = st.checkbox("🔍 디버깅 모드", help="AI 응답 분석 정보를 표시합니다")

        # 길이 정보 표시
        length_info = {
            "간단": {"chars": "800자", "time": "⚡ 빠름", "desc": "핵심만 간략히"},
            "보통": {"chars": "1,500자", "time": "⚖️ 균형", "desc": "적당한 상세도"}, 
            "상세": {"chars": "2,500자", "time": "🔍 상세", "desc": "충분한 설명"}
        }
        
        info = length_info[idea_length]
        st.write(f"**{info['time']}**")
        st.write(f"📝 {info['chars']}")
        st.write(f"💡 {info['desc']}")

    # 생성 버튼
    if st.button("🚀 해커톤 아이디어 생성하기", type="primary"):
        if problem_area and target_problem and ai_technology and target_users and expected_impact:
            with st.spinner("AI가 혁신적인 아이디어를 생성하고 있습니다..."):
                generated_idea = generate_hackathon_idea_with_nova(
                    problem_area, target_problem, ai_technology, target_users, expected_impact, idea_length, debug_mode
                )
            
            # 생성된 아이디어를 세션 상태에 저장
            st.session_state.current_idea = {
                "generated_content": generated_idea,
                "idea_length": idea_length
            }
            st.session_state.idea_generated = True
            
            st.success("✅ 아이디어가 생성되었습니다!")
        else:
            st.error("모든 필드를 입력해 주세요!")

    # 생성된 아이디어 표시 및 저장 기능 (세션 상태 기반)
    if hasattr(st.session_state, 'idea_generated') and st.session_state.idea_generated and hasattr(st.session_state, 'current_idea'):
        current_idea = st.session_state.current_idea
        
        # 아이디어 헤더 (길이 정보 포함)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("## 📋 생성된 해커톤 아이디어")
        with col2:
            idea_length_used = current_idea.get('idea_length', '보통')
            length_badge = {
                "간단": "⚡ 간단형 (800자)",
                "보통": "⚖️ 표준형 (1,500자)",
                "상세": "🔍 상세형 (2,500자)"
            }
            st.info(f"📏 {length_badge.get(idea_length_used, '표준형')}")
        
            # 실제 글자 수 계산 및 표시
        content_length = len(current_idea['generated_content'])
        st.caption(f"📊 실제 생성된 글자 수: {content_length:,}자")
        
        st.markdown(current_idea['generated_content'])
        
        st.write("---")
        
        # 새 아이디어 생성 버튼
        if st.button("🔄 새 아이디어 생성", key="new_idea_button", type="primary"):
            # 현재 세션 상태 초기화
            if 'current_idea' in st.session_state:
                del st.session_state.current_idea
            if 'idea_generated' in st.session_state:
                del st.session_state.idea_generated
            st.rerun()

    st.write("---")
    st.info("🌱 지속가능한 세상을 위한 혁신적인 AI 솔루션 아이디어를 만들어보세요! AWS Bedrock Nova Lite가 도와드립니다.")

    with tab2:
        st.write("## 📋 간단한 Streamlit 앱 PRD 생성")
        
        # 아이디어 입력 방식 선택
        input_method = st.radio(
            "📥 아이디어 입력 방식",
            ["직접 입력", "아이디어 생성 탭에서 가져오기"],
            horizontal=True
        )
        
        if input_method == "아이디어 생성 탭에서 가져오기":
            if hasattr(st.session_state, 'current_idea') and 'generated_content' in st.session_state.current_idea:
                st.write("### 📝 가져온 아이디어")
                with st.expander("생성된 아이디어 내용 보기"):
                    st.markdown(st.session_state.current_idea['generated_content'])
                idea_content = st.session_state.current_idea['generated_content']
            else:
                st.warning("⚠️ 아이디어 생성 탭에서 먼저 아이디어를 생성해주세요.")
                idea_content = ""
        else:
            idea_content = st.text_area(
                "💡 해커톤 아이디어 내용",
                height=200,
                placeholder="생성된 해커톤 아이디어를 여기에 붙여넣어 주세요..."
            )
        
        st.write("---")
        
        # PRD 생성 버튼
        if st.button("📋 간단한 PRD 생성하기", type="primary"):
            if idea_content.strip():
                with st.spinner("간단한 Streamlit 앱 PRD를 생성하고 있습니다..."):
                    prd_content = generate_streamlit_prd(idea_content)
                
                st.write("## 📋 생성된 PRD")
                st.markdown(prd_content)
                
                # PRD를 세션에 저장
                st.session_state.current_prd = prd_content
                
                st.write("---")
                
                # 파일 저장 옵션
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("💾 MD 파일로 저장", type="secondary"):
                        success, result = save_prd_to_markdown(prd_content)
                        if success:
                            st.success(f"✅ PRD가 저장되었습니다!")
                            st.info(f"📁 저장 위치: `{result}`")
                        else:
                            st.error(f"❌ 저장 실패: {result}")
                
                with col2:
                    st.download_button(
                        label="📥 MD 파일 다운로드",
                        data=prd_content,
                        file_name=f"streamlit_app_prd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            else:
                st.error("❌ 아이디어 내용을 입력해주세요!")
        
        # 기존 PRD 표시 (세션에 있는 경우)
        if hasattr(st.session_state, 'current_prd'):
            st.write("---")
            st.write("### 📋 현재 세션의 PRD")
            with st.expander("PRD 내용 보기"):
                st.markdown(st.session_state.current_prd)
                
                # 기존 PRD도 다운로드 가능
                st.download_button(
                    label="📥 현재 PRD 다운로드",
                    data=st.session_state.current_prd,
                    file_name=f"current_prd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    key="download_current_prd"
                ) 
