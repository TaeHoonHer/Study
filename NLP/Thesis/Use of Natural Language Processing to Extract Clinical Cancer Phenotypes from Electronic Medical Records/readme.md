## Thesis Study


전자 의료 기록을 Omics 데이터와 연관시키기 위한 모델은 암 환자의 표현형 정보(Phenotype information)(2)의 중요한 원천인 임상 텍스트 대부분을 무시한다. 임상 텍스트에서 암 표현형 데이터을 추출하기 위해 자연어처리(NLP) 방법을 사용한다. 여기서는 PubMed(4)에 수록된 출판물과 NLP 및 기계 학습 컨퍼런스 진행을 기반으로 종양학과 관련된 NLP 및 정보 추출 방법의 발전을 검토한다

---
종양학에서 임상 치료와 연구 과정에서 생성된 데이터는 나날이 증가하고 있다. 지난 10년동안 전자 의료 기록의 사용량을 보면 꾸준히 증가하고 있는 추세임을 확인할 수 있다.

NCI(6), 역학 및 최종 결과 프로그램, 국립 암 데이터베이스, 암 게놈 지도책(TCGA)(5) 및 인간 종양 지도책(7)과 같은 대규모 데이터베이스는 임상 및 번역 종양 연구에 점점 더 중요한 수단이 되고 있다. 그러나 중요한 뉘앙스 표현형 데이터(8)는 임상 자유 텍스트(9)에 고정되어 있으며, 이는 임상 프레젠테이션, 제공자 인상(10), 절차 세부 정보 및 관리 의사 결정을 문서화하고 전달하는 주요 형태로 남아 있다. 

EMR(11) 및 -omics 데이터의 확산에도 불구하고, 중요하고 정확한 표현형 정보는 임상 문서에만 자세히 설명된다. 언어를 계산 가능한 표현으로 변환하는 것으로 광범위하게 정의되는 자연어 처리는 임상 텍스트 내에서 미묘한 데이터를 대규모로 추출하는 데 핵심적이다

[즉, 임상 자유 텍스트를 표현가능하도록 변환하는 것이 NLP가 임상 문서에서 활용되는 이유이다]

---

임상 또는 건강관리 텍스트의 분석을 도와주는 Clinical NLP(12)는 수십 년 전부터 사용이 되고 있다. 하지만, 최근 몇 년 동안 계산 능력과 알고리즘이 충분히 발전하여 종양학적 조사를 확대하는 데 어마어마한 능력을 보여주었다.

CNLP의 기사중에 Clinical Natural Language Processing in the Age of Big Data and Deep Learning이라는 기사가 있다. CNLP에 대한 최신 연구 동향과 발전 가능성에 대한 리뷰를 작성한 기사인데, 해당 기사에서는 다양한 데이터 소스에서 정보 추출 및 분석을 위한 NLP 기술을 소개하고, Deep Learning 기술의 적용 가능성 또한 다루고 있다

해당 기사에서는 CLinical NLP 분야에서 가장 많이 사용되는 정보 추출 방법 중 Named Entity Recognition(NER)과 같은 주요 기술을 설명하고 있다. 또한, Electronic Health Record(EHR)과 같은 대규모 병원 데이터를 처리하는 방법, CLinical NLP 기술을 활용해 의료 연구 수행 방법 또한 설명을 하고 있다.

기사에서 다루는 내용 중, DL(Deep Learning)기술을 CNLP 분야에 적용하는 방법이 있다. DL을 사용하면 높은 정확도로 정보를 추출할 수 있고, 자연어 처리를 함에 있어 매우 효과적인 방법이라고 할 수 있다.

---

2010년대 중반은 인공지능과 NLP에 대한 폭발적인 관심과 함께 양질의, 수많은 디지털화가 된 텍스트 데이터가 있었고, 하드웨어 발전으로 어마어마한 기술적 발전이 이루어졌다. 그 후, 우리는  암에 대한 cNLP 방법의 주요 최근 개발을 검토하기 시작했다. 

---

### Major NLP Algorithmic Advances

3년간 NLP에 대한 다양한 방법론의 개발과 함께 딥러닝을 응용한 다양한 사례를 보여주었다. DL 기술 자체는 1980년대에 처음 고안이 되었지만, 
- 대규모 디지털 텍스트 코퍼스
- 데이터 집약 알고리즘
- 그래픽 처리 장치를 활용한 대규모 병렬 컴퓨팅 

이 3가지가 수렴이 될 때가지는 정상적으로 작동이 되지가 않았다.

다양한 산업군에서 DL은 만능, 최첨단 인공지능으로 인식이 되고는 한다. 

What is Omics medical data?
- Omics medical data refers to a large amount of data generated from various high-throughput techniques used in biomedical research, such as genomics, transcriptomics, proteomics, metabolomics, and epigenomics. These techniques allow researchers to study the molecular mechanisms underlying diseases and to identify potential biomarkers and therapeutic targets.

- Genomics, for example, involves the study of the entire genome, including the coding and non-coding regions of DNA, and can provide information about genetic variations that may be associated with disease susceptibility or treatment response. Transcriptomics studies the expression of all genes in a cell or tissue at a given time, while proteomics studies the protein content of a cell or tissue. Metabolomics analyzes the small molecules, such as metabolites, present in a biological sample, and epigenomics studies the modifications to DNA and chromatin structure that can affect gene expression.

- By integrating data from these different omics approaches, researchers can gain a more comprehensive understanding of the molecular mechanisms underlying diseases and develop personalized approaches to diagnosis and treatment. However, the analysis and interpretation of omics data require sophisticated computational tools and expertise in bioinformatics and statistics.

- Omics 데이터란 생물의학 연구에서 사용되느 다양한 고처리 기술을 의미합니다. 예를 들어 유천체, 전사체, 단백체, 대사체, 유행체학 등에서 생성된 대량의 데이터입니다. 이러한 기술은 질병의 분자 메커니즘을 연구하고 잠재적인 생체 마커(1) 및 치료 대상을 식별하는데 사용이 됩니다


용어
1) 생체 마커 : 생물학적 과정, 질병, 상태 등을 나타내는 분자, 유전자, 세포, 조직 또는 생리학적 측정치를 말한다. 예를 들어 혈액 내의 특정 단백질 농도, 유전자 돌연변이, 조직 내의 특정 세포 수 등이 생체 마커가 될 수 있다. 생체 마커는 진단, 예방, 치료, 치료 효과 평가, 약물 개발 등 다양한 의료분야에 이용이 될 수 있고, 개인 맞춤형 치료 및 의학 연구에 매우 중요한 역할을 한다.

2) 표현형 정보 : 개체가 나타내는 모든 외부적 형질과 행물학적 특성의 의미다. 즉, 개체의 생물학적 특성을 실제로 관찰할 수 있는 것들을 말하는데 그 종류로는
    - 신장
    - 체중
    - 피부색
    - 눈색깔
    - 혈액형
    
같은 정보들을 말하고, 이러한 정보들은 모두 개체의 페노타입 정보, 의학 연구에서 개인 맞춤형 치료를 개발하는데 중요한 역할을 한다. 유전자형과 페노타입(3) 정보를 결합, 질병 예측 및 진단, 약물 개발에 사용이 된다

3) Phenotype : 개체가 나타내는 모든 회부적 형질과 생물학적 특성을 의미. 개체의 유전자형과 상호작용하여 나타나는 개체의 생물학적 특성을 말한다.

4) PubMed : 생명의학, 생물학, 바이오메디컷, 보건 등 다양한 분야의 문헌을 검색할 수 있는 검색 엔진.

5) 암 게놈 지도책이란? : Cancer Genome Atlas(TCGA)는 암 발생에 영향을 미치는 유전자 변이 및 암 세포의 유전체 데이터를 수집, 분석하고 공유하는 의료 프로젝트이다

6) NCI : National Cancer Institute(국립암연구소)의 줄임말이다
7) 인간 종양 지도책(Human Cancer Atlas) : 전 세계 암 연구자들이 함께 참여해 모든 종양에 대한 세부적인 분석 결과를 수집, 연결 및 공유하는 대규모 프로젝트이다. 종양 내부의 세포 유형과 그들의 상호작용, 세포 내 분자 신호체계, 세포 외 기질 등 다양한 영역에서 유전자 변이, 엑솜, 유전자 발현, 단백질 발현, 대사물질 및 생체 마커 등을 분석하여 인간 종양의 정밀한 분석을 시도하고 있다.
8) nuance phenotype(뉘앙스 표현형) 데이터 in 종양학 : 종양의 특성을 서로 다른 그룹으로 분류, 특정 기능을 하는 유전자 변이를 연관시키는 등 다양한 연구에서 사용되는 데이터이다.
    - 종류
        - 종양의 크기
        - 모양
        - 색상
        
같은 기본적인 형태적 특징부터
    - 세부적 특징
        - 세포 간 상호작용
        - 세포 내 분자 상호작용
        - 대사 물질
        - 생체 마커
        
와 같은 세부적인 특징을 분석해 얻어진다

9) 임상 자유 텍스트(clinical free text) :  의료 기록, 의사소통 및 의료 문사 등에서 얻어진 비정형화된 자연어 데이터를 의미. 구조화된 데이터와 다르게 일반적으로 컴퓨터가 쉽게 이해할 수 없는 형태로 이루어져 있다. 즉, 문맥에 따라 여러 방향성으로 해석이 가능하다.
- 예시
    - 대표적인 예시로 [화학요법], [항암제 치료]는 같은 의미를 내포하고 있지만, 다른 용어로 사용이 가능한 것 처럼 말이다 

임상 자유 텍스트는 EHR(11), 의료 보고서, 의사소통 기록에서 수집이 가능하다. 이러한 비정형 데이터를 NLP 기술을 활용하여 구조화된 데이터로 변환할 수 있다.

10) 제공자 인상 : 보험금 청구 시 의료 서비스를 제공한 의료인 또는 병원에 대해 보험회사가 지급하는 금액을 의미

11) EMR data : Electronic Medical Record에서 추출한 정보를 의미한다. EMR은 환자의 건강 상태와 관련 정보를 기록하는 시스템으로 의료진이 진료 및 치료 과정에서 수집한 정보를 포함한다. 
- 종류
    - 환자의 개인정보
    - 진단 정보
    - 처방 정보
    - 검사 결과
    - 수술 기록

EMR 데이터는 Healthcare Big Data의 중요한 출처 중 하나이고, EMR 데이터를 분석하여 의료진이 환자에게 보다 적절하고 정확한 진료, 치료 서비스를 제공할 수 있도록 도와준다.

12) Clinical NLP : 자연어 처리 기술이 의료 분야에 적용이 된 것이다. 의료 기록, 진단 보고서, 치료 정보 등을 기계가 이해할 수 있는 형태로 변환하여 의료 연구, 진단, 치료 개발 등에 활용할 수 있다.

일반적인 NLP와 유사하게 텍스트 분석, 정보 추출, 정보 분류 및 자동 번역 등의 기술이 들어가있다. 하지만, 의료분야에 적용이 되는 만큼 전문 용어와 약어, 의료 용어 등이 사용되어 일반적인 NLP보다 처리가 어렵다.

의료 연구에서도 사용이 되고는 한다. 대량의 의료 정보를 수집, 분석해야하는 의료 연구의 특성상 Clinical NLP를 활용하여 이를 자동화하고 효율적으로 처리할 수 있다. 이를 통해 연구자는 질병 예측, 새로운 치료 개발 등 보다 정확한 결론을 도출할 수 있다.
