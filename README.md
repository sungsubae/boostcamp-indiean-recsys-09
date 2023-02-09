![sdfds2](https://user-images.githubusercontent.com/75313644/217699370-333fccd7-2a50-43bd-ae01-9517db260b0d.png)
# IndieAn - 인디게임 추천
### 🎮 유저의 Steam 히스토리를 사용해, 유저에게 맞춤형 **인디게임**을 추천해주는 서비스 **IndieAn**입니다.   
## 1. 기획

### **문제인식**

- 2021년 기준, 전체 스팀게임 중 인디게임의 수가 96%를 차지하고 있음에도 실제 판매량과 판매수익은 AAA급 게임이 60-70%를 차지
- 인디게임은 적은 개발 비용으로 인해 홍보가 크게 이루어지지 못해, 대중에게 노출되지 않음
- 인디게임은 각각의 게임마다 개성이 강하므로 단순한 리스트업 보단 개인화 추천시스템이 필요함

### **기대효과**

- 유저는 개인취향에 맞는 새로운 인디게임을 추천받음으로 다양한 게임경험을 할 수 있음
- 인디게임 제작자는 자신들의 게임이 노출되는 기회를 갖게 됨
- 인디게임 활성화로 인한 게임시장 활성화

### **프로젝트 구조도**

![플젝-페이지-1 drawio (2)](https://user-images.githubusercontent.com/75313644/217700589-19e33760-93b8-4a46-ac3f-47784cfb0755.png)

### **사용자 흐름도**

![플젝-페이지-2 drawio](https://user-images.githubusercontent.com/75313644/217700599-4ebabeb8-e118-4545-9172-e7bb2ff7893f.png)


## 2. 데이터 및 DB

### 데이터

| 데이터 분류 | 데이터 설명 | 활용 | 크기 |
| --- | --- | --- | --- |
| 유저-아이템 상호작용 데이터 | [Steam Video Game and Bundle Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data), USCD에서 수집한 데이터 중 유저-아이템 상호작용 데이터 | 추천시스템 모델 학습 | 5153209*5 |
| 게임 아이템 메타 데이터 | [2022 Steam Games Kaggle](https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022)에서 수집한 게임 메타 데이터 | 아이템 필터링, 큐레이션,콘텐츠 기반 추천시스템 모델 | 55691*23 |
| 유저 데이터 | 유저 ID를 받아 Steam API를 이용해 직접 받아오는 유저-아이템 상호작용 데이터 | 개인화 추천시스템 모델 Input | - |

### DataBase

![image](https://user-images.githubusercontent.com/75313644/217702168-a5530b02-7e5f-4ae2-86d1-0714e2d4fb07.png)


## 3. 모델

### 두 가지 고려사항

- **성능과 Inference 속도의 Trade-off**: 좋은 성능을 갖고 있더라도 실 서비스 예측이 느리면 문제
- **User Free & Item Cold-start 문제**: 실시간으로 추가되는 아이템들과 보유 데이터의 차이로 인한 cold-start문제, 개별 유저의 상호작용 데이터를 사용하기 위한 user free 모델 문제

### [Embarrassingly Shallow Autoencoders for Sparse Data : EASE](https://arxiv.org/abs/1905.03375)

![image](https://user-images.githubusercontent.com/75313644/217702651-16f3f24e-fe69-4003-a0a8-cf831bfa2790.png)



## 4. 백엔드
FastAPI를 통해 구현, 크게 **API Server**와 **Inference Server** 분류
### API Server
- 프론트엔드, 인퍼런스 서버 그리고 데이터베이스를 연결해주는 역할로 프론트엔드에 필요한 API를 제공하고 인퍼런스 결과를 데이터베이스에 저장해주는 서버를 구축

### Inference Server
- 백엔드로 부터 전송 받은 유저 히스토리 데이터를 모델에 인풋하여 인퍼런스를 통해 유저의 추천 결과를 백엔드로 다시 전송할 수 있는 서버를 구축



## 5. 프론트엔드 
![샘플2](https://user-images.githubusercontent.com/75313644/217703335-505873a8-1f65-49de-a9c5-50db6745128e.gif)



## 6. CICD

### Github action & Slack
- **Docker & Github Action** : 백엔드와 프론트엔드를 도커를 이용해 이미지를 빌드하여, GCP 환경 VM에 업로드 진행 및 Github Action을 활용하여 CI/CD 환경 구축
- **Slack 알림 기능 :** 협업을 위해 Github Action의 결과를 Slack에도 알림을 갈 수 있도록 하는 기능을 Github Action에서 구축
       
### Airflow
- DataLake에 적제된 신규데이터를 DataWarehouse인 BigQuery에 적재한 뒤, 모델 학습을 통해 성능을 체크하는 하나의 Flow 구축
- 실제 서비스 이후, Online data가 현재 모델 성능에 주는 영향을 파악해 추가 개선



## 7. 팀소개
![image](https://user-images.githubusercontent.com/75313644/217704990-176c03fc-988c-448e-b277-78af497d7baa.png)

| [<img src="https://avatars.githubusercontent.com/u/94108712?v=4" width="200px">](https://github.com/KChanho) | [<img src="https://avatars.githubusercontent.com/u/22442453?v=4" width="200px">](https://github.com/sungsubae) | [<img src="https://avatars.githubusercontent.com/u/28619804?v=4" width="200px">](https://github.com/JJI-Hoon) | [<img src="https://avatars.githubusercontent.com/u/71113430?v=4" width="200px">](https://github.com/sobin98) | [<img src="https://avatars.githubusercontent.com/u/75313644?v=4" width="200px">](https://github.com/dnjstka0307) |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------:
|                          [김찬호](https://github.com/KChanho)                            |                            [배성수](https://github.com/sungsubae)                             |                        [이지훈](https://github.com/JJI-Hoon)                           |                          [정소빈](https://github.com/sobin98)                           |                            [조원삼](https://github.com/dnjstka0307)                   
