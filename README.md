# MLOps 마케팅 캠페인 상세 가이드

## 서론

이 상세 가이드는 프로젝트에 구현된 전체 MLOps 워크플로우를 안내합니다. 다음 단계를 따르면 Cloudera AI(CAI)를 사용하여 머신러닝 모델을 구축하고, 배포하고, 자동화하고, 모니터링 하는 방법을 배울 수 있습니다.

## 전제조건

시작하기 전에 다음 사항을 확인하세요:

- Cloudera AI Workbench에 대한 접근 권한
- Python, machine learning 개념, 및 SQL에 대한 기본적인 이해
- CAI에서 projects, sessions, models, 및 jobs 을 생성할 수 있는 사용자 권한
- Project Settings => Advanced => Environment Variables로 CONNECTION_NAME에 Spark Connection 이름 추가(Spark Connection이름은 Project Settings => Data Connection에서 확인 가능)

## 설정

### CAI Session 생성하기

1. CAI Workbench를 열기
2. Sessions에서 new session 생성:
   - **Editor**: JupyterLab
   - **Kernel**: Python 3.10
   - **Edition**: Standard
   - **Version**: 2025.01
   - **Add-on**: Spark 3.3
   - **Resource Profile**: 2 vCPU / 4 GiB Memory (or larger if needed)


## 단계별 워크플로우

### Step 1: 데이터 수집
*파일: 00_download.py*

**기능:**
이 스크립트는 은행 고객과 마케팅 캠페인 기간 동안 정기 예금에 가입했는지 여부에 대한 정보를 담고 있는 은행 마케팅 데이터셋을 다운로드 합니다.

**실행 방법:**
1. CAI Session에서 `00_download.py` 파일 열기
2. Terminal Access창을 열고, `python 00_download.py`을 입력하여 실행

**주요 구성 요소:**
- UCI 저장소에서 zip 파일을 다운로드
- CSV 데이터 추출
- CAI session의 로컬 파일 시스템에 저장

**MLOps - 데이터 수집 단계:**
이는 모델링에 필요한 원시 데이터를 수집하는 ML 파이프라인의 데이터 수집 단계에 해당합니다.

### Step 2: 데이터 레이크 통합
*파일: 01_write_to_dl.py*

**기능:**
이 스크립트는 Apache Iceberg 형식을 사용하여 다운로드한 데이터셋을 데이터 레이크로 이동시켜 ML 워크플로우에서 접근할 수 있도록 합니다.

**실행 방법:**
1. CAI Session에서 `01_write_to_dl.py` 파일 열기
2. Terminal Access창을 열고, `python 01_write_to_dl.py`을 입력하여 실행


**주요 구성 요소:**
- CAI 데이터 연결을 사용하여 데이터 레이크에 대한 연결 설정
- 은행 데이터로  Iceberg 테이블 생성
- 적절한 스키마 관리 및 데이터 유형 구현

**MLOps - 데이터 저장 및 구성 단계:**
이는 원시 데이터가 분석 및 모델 학습에 적합한 형식으로 적절하게 저장되고 버전 추적 기능이 포함되는 데이터 저장 및 구성 단계에 해당합니다.

### Step 3: 탐색적 데이터 분석 (EDA)
*파일: 02_EDA.ipynb*

**기능:**
이 노트북은 뱅킹 데이터셋을 분석하여 패턴, 분포 및 상관관계를 이해하는 과정을 안내합니다.

**실행 방법:**
1. CAI Session에서 `02_EDA.ipynb` 파일 열기
2. 셀을 순차적으로 실행

**주요 구성 요소:**
- 이전 단계에서 생성한 Iceberg 테이블에 연결
- 피처(feature)에 대한 통계 분석 수행
- 데이터를 더 잘 이해하기 위한 시각화 생성
- 잠재적인 특징 엔지니어링 기회 식별

**MLOps - 탐색적 데이터 분석(EDA) 단계:**
이는 데이터 과학자가 모델 설계 결정을 내리는 데 필요한 통찰력을 얻는 데이터 이해 단계에 해당합니다.

### Step 4: MLflow를 사용한 모델 학습
*파일: 03_train.py*

**기능:**
이 스크립트는 고객이 정기 예금에 가입할지 여부를 예측하기 위해 다양한 분류 모델을 학습하고, 모든 실험을 MLflow로 추적합니다.

**실행 방법:**
1. CAI Session에서 `03_train.py` 파일 열기
2. Terminal Access창을 열고, `python 03_train.py`을 입력하여 실행

**주요 구성 요소:**
- Iceberg 테이블에서 데이터 읽기
- 데이터 전처리 및 피처 엔지니어링 수행
- 여러 XGBoost 모델을 다양한 하이퍼파라미터로 학습
- MLflow에 메트릭, 매개변수 및 모델 로깅

**MLOps - 실험 및 모델 개발 단계:**
이는 여러 접근 방식을 시도하고 재현성 및 비교를 위해 체계적으로 추적하는 실험 및 모델 개발 단계에 해당합니다.

### Step 5: 모델 선택 및 배포
*파일: 04_api_deployment.py*

**기능:**
이 스크립트는 테스트 정확도를 기준으로 최적의 모델을 선택하고, 모델 레지스트리에 등록한 후 REST API 엔드포인트로 배포합니다.

**실행 방법:**
1. CAI Session에서 `04_api_deployment.py` 파일 열기
2. Terminal Access창을 열고, `python 04_api_deployment.py`을 입력하여 실행

**주요 구성 요소:**
- MLflow 실험을 쿼리하여 최상의 성능을 보이는 모델 찾기
- CAI - AI Registry에 모델 등록
- REST API 엔드포인트를 포함하는 모델 배포 생성
- 배포된 모델에 대한 모니터링 설정

**MLOps - 모델 배포 단계:**
이는 모델을 표준화된 인터페이스를 통해 다른 애플리케이션 및 시스템에서 접근할 수 있도록 하는 배포 단계에 해당합니다.

### Step 6: 자동화된 모델 수명 주기 설정 (6-8단계)

다음 세 가지 스크립트는 함께 작동하여 자동화된 모델 업데이트 워크플로우를 보여줍니다. 각 스크립트에 대해 CAI Job을 생성하고 종속성을 구성합니다.

#### Step 6.1: 새 배치 데이터 생성
*파일: 05_newbatch.py*

**기능:**
이 스크립트는 새로운 합성 뱅킹 데이터를 생성하여 데이터 레이크에 저장함으로써 새로운 고객 상호작용의 도착을 시뮬레이션합니다.

**Job으로 실행하는 방법:**
1. 프로젝트 왼쪽 메뉴에서 "Jobs" 으로 이동
2. "New Job" 클릭
3. 다음 매개변수 설정:
   ```
   Name: New Batch 
   Script: 05_newbatch.py
   Editor: Workbench
   Kernel: Python 3.10
   Spark Add On: Spark 3.3
   Edition: Standard
   Version: 2025.01
   Schedule: Manual
   Resource Profile: 2 vCPU / 4 GiB / 0 GPU
   ```
4. "Create Job" 클릭

**주요 구성 요소:**
- 합성 뱅킹 거래 데이터 생성
- 기존 Iceberg 테이블에 새 데이터 추가
- 시간이 지남에 따라 실제 데이터가 누적되는 상황 시뮬레이션

**MLOps - 모델 운영을 통한 신규 데이터 수집 단계:**
이는 모델이 배포된 후에도 새로운 데이터가 계속 도착하는 프로덕션 시스템에서의 지속적인 데이터 수집에 해당합니다.


#### Step 6.2: 모델 재학습
*파일: 06_retrain.py*

**기능:**
이 스크립트는 과거 데이터와 새 데이터를 모두 사용하여 모델을 재학습하고, 새로운 MLflow 실험을 생성합니다.

**Job으로 실행하는 방법**
1. 프로젝트 왼쪽 메뉴에서 "Jobs" 으로 이동
2. "New Job" 클릭
3. 다음 매개변수 설정:
   ```
   Name: Retrain Models
   Script: 06_retrain.py
   Editor: Workbench
   Kernel: Python 3.10
   Spark Add On: Spark 3.3
   Edition: Standard
   Version: 2025.01
   Schedule: Dependent on New Batch
   Resource Profile: 2 vCPU / 4 GiB / 0 GPU
   ```
4. "Create Job" 클릭

**주요 구성 요소:**
- 원본 및 새 데이터가 포함된 확장된 데이터셋 로드
- 업데이트된 파라메터로 모델 재학습
- MLflow에 새로운 실험 결과 로깅

**MLOps - 주기적인 모델 재학습 단계:**
이는 모델 성능을 유지하기 위해 새로운 데이터로 주기적으로 모델을 재학습하는 모델 유지 관리 주기에 해당합니다.

#### Step 6.3: 모델 재배포
*파일: 07_api_redeployment.py*

**기능:**
이 스크립트는 재학습 실험에서 최적의 모델을 선택하고, 이전 모델 버전을 대체하도록 재배포합니다.

**Job으로 실행하는 방법**
1. 프로젝트 왼쪽 메뉴에서 "Jobs" 으로 이동
2. "New Job" 클릭
3. 다음 매개변수 설정:
   ```
   Name: Redeploy Model
   Script: 07_api_redeployment.py
   Editor: Workbench
   Kernel: Python 3.10
   Spark Add On: Spark 3.3
   Edition: Standard
   Version: 2025.01
   Schedule: Dependent on Retrain Models
   Resource Pro파일: 2 vCPU / 4 GiB / 0 GPU
   ```
4. "Create Job" 클릭

5. 세 가지 작업을 모두 생성한 후, New Batch 작업을 수동으로 트리거합니다. Job History 탭에서 실행을 모니터링하고, 완료되면 MLOps 파이프라인의 다음 작업인 Retrain XGBoost가 트리거되고 마지막으로 API Redeployment 작업이 실행되는 것을 관찰합니다.



**주요 구성 요소:**
- 재학습에서 가장 우수한 성능을 보이는 모델 선택
- 레지스트리의 모델 업데이트
- 이전 버전을 대체하는 새 배포 생성
- 모델 API 사용자를 위한 원활한 전환 보장

**MLOps - 모델 버전 관리 및 운영 모델 업데이트 단계:**
이는 모델 수명 주기 관리의 중요한 부분인 프로덕션에서의 모델 버전 관리 및 업데이트를 나타냅니다.

### Step 7: 모델 모니터링 설정
*파일: 08_model_simulation.py*

**기능:**
이 스크립트는 클라이언트 애플리케이션이 배포된 모델 엔드포인트에 호출을 수행하고 예측을 로깅하는 것을 시뮬레이션합니다.

**실행 방법:**
1. CAI Session에서 `08_model_simulation.py` 파일 열기
2. Terminal Access창을 열고, `python 08_model_simulation.py`을 입력하여 실행

**주요 구성 요소:**
- 모델 API에 대한 합성 요청 생성
- 예측 및 시뮬레이션된 "실제 결과(ground truth)" 로깅
- 모델 성능 모니터링을 위한 기반 마련

**MLOps - 모델 서빙 및 로깅 단계:**
이는 프로덕션에서 모델 성능을 모니터링하는 데 필요한 모델 서빙 및 로깅 인프라를 나타냅니다.

### Step 8: Performance Tracking Dashboard
*파일: 09_model_monitoring_dashboard.ipynb*

**기능:**
이 노트북은 시간 경과에 따른 모델 성능 메트릭을 모니터링하는 대시보드를 생성합니다.

**실행 방법:**
1. CAI Session에서 `09_model_monitoring_dashboard.ipynb` 파일 열기
2. 셀을 순차적으로 실행합니다

**주요 구성 요소:**
- 예측 로그 및 실제 결과(ground truth) 검색
- 주요 성능 메트릭 계산
- 시간 경과에 따른 성능 추세 시각화
- 이해관계자를 위한 대시보드 생성

**MLOps - 모델 모니터링 및 알림:**
이는 모델이 프로덕션에서 지속적으로 잘 작동하도록 보장하는 데 필요한 모니터링 및 알림 기능을 나타냅니다.

## 완전한 MLOps 워크플로우

이 모든 단계를 완료함으로써 다음을 포함하는 포괄적인 MLOps 워크플로우를 구현했습니다::

1. **데이터 관리** - 수집, 저장 및 버전 관리
2. **실험** - 추적된 결과와 함께 여러 모델 반복
3. **배포** - 모델을 API로 운영화(프로덕션화)
4. **자동화** - 데이터 처리 및 모델 업데이트를 위한 예약된 작업
5. **모니터링** - 운영(프로덕션)에서 모델 성능 추적

이 워크플로우는 엔터프라이즈 데이터 제어, 간소화된 운영, 확장 가능한 인프라, 포괄적인 거버넌스 및 반복적인 모델 관리를 갖춘 통합 플랫폼을 사용하는 Cloudera의 MLOps 접근 방식을 구현한 것입니다.

## 추가 고려 사항

### 모델 선택 기준

이 랩에서는 단순화를 위해 테스트 정확도를 기반으로 모델을 선택합니다. 실제 시나리오에서는 모델 선택에 다음이 포함될 수 있습니다::

- 여러 메트릭 (정밀도, 재현율, F1-스코어)
- 비즈니스별 KPI
- 공정성 및 편향 고려 사항
- 모델 설명 가능성 요구 사항
- 추론 속도 및 리소스 제약 조건

### 향후 개선 사항

현재 워크플로우는 다음을 통해 개선될 수 있습니다::

- 개념 및 데이터 드리프트 감지
- 더 나은 특징 관리를 위한 피처 저장소
- 점진적인 모델 롤아웃을 위한 A/B 테스트
- 모델 설명 가능성 도구
- (예약 기반이 아닌) 트리거 기반 재학습
- 더 정교한 모니터링 알림

## Additional Resources

이 랩에서 사용된 도구 및 기술에 대한 자세한 내용은 다음을 참고하세요

- [Cloudera AI Documentation](https://docs.cloudera.com/machine-learning/cloud/index.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Apache Iceberg](https://iceberg.apache.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

[odog96/cml-mlops-banking-campain](https://github.com/odog96/cml-mlops-banking-campain.git)의 한글화 버전입니다.
