# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_korean_font():
    """한글 폰트를 설정하는 함수"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows에서 사용 가능한 한글 폰트들
        font_list = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Dotum']
    elif system == 'Darwin':  # macOS
        font_list = ['AppleGothic', 'NanumGothic', 'Arial Unicode MS']
    else:  # Linux
        font_list = ['NanumGothic', 'DejaVu Sans', 'Liberation Sans']
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in font_list:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            break
    else:
        # 기본 폰트로 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False
