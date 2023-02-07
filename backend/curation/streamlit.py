import streamlit as st
from pyparsing import empty

import requests


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
empty1,con1,empty2 = st.columns([0.2,0.6,0.2])
empty1,con2,empty2 = st.columns([0.2,0.6,0.2])
empyt1,con3,con4,con5,empty2 = st.columns([0.2,0.2,0.2,0.2,0.2])
empyt1,con6,empty2 = st.columns([0.2,0.6,0.2])


def main():

    with empty1:
        empty() # 여백부분1

    with con1:
        # 타이틀
        st.title("Curation")

    with con2:
        # 필터링 조건 입력받기 - 가격
        price = st.slider('가격대', 0, 4000, 10)

    with con3:
        # 필터링 조건 입력받기 - 장르
        genre = st.multiselect(
            '장르',
            ['Action', 'Adventure', 'Casual', 'Racing', 'RPG', 'Simulation', 'Sports', 'Strategy'])

    with con4:
        # 필터링 조건 입력받기 - 카테고리
        category = st.multiselect(
            '카테고리',
            ['Single-player', 'Multi-player', 'Co-op', 'PvP', 'Remote Play', 'Steam Cloud'])

    with con5:
        # 필터링 조건 입력받기 - 플랫폼
        platform = st.multiselect(
            '플랫폼',
            ['windows', 'mac', 'linux'])

    with con6:
        # 큐레이션 결과 출력 - 그냥 간단히 csv 형태로 출력!
        if st.button('결과 확인'):
            # 백엔드/프론트엔드 분리
            data = {
                'genre': genre, 
                'category': category, 
                'price': price, 
                'platform': platform
            }
            response = requests.post("http://localhost:8000/curation", json=data)
            curation = response.json()
            if curation:
                st.write(curation)
            else:
                st.write("조건에 맞는 인디게임이 없습니다!")
            
        else:
            st.write('버튼을 눌러 큐레이션 결과를 확인하세요!')
    
    with empty2:
        empty() # 여백부분2


main()