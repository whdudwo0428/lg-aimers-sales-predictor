import platform
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 정확한 폰트 파일 경로
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


# FontProperties 객체 생성
font_prop = fm.FontProperties(fname=FONT_PATH)
font_name = font_prop.get_name()
print(f"[DEBUG] FontProperties resolved name: {font_name}")

# matplotlib에 새 폰트 등록 및 강제 설정
matplotlib.font_manager.fontManager.addfont(FONT_PATH)
matplotlib.rcParams["font.family"] = font_name
matplotlib.rcParams["axes.unicode_minus"] = False

# 확인용: 현재 설정된 폰트 패밀리
print("[DEBUG] matplotlib.rcParams['font.family']:", matplotlib.rcParams["font.family"])

plt.figure()
plt.title("한글 테스트: 메뉴별 매출", fontsize=14)
plt.xlabel("x축 레이블")
plt.ylabel("y축 레이블")
plt.text(0.5, 0.5, "테스트 글자: 나눔고딕 적용 여부", ha='center', va='center')
plt.savefig("font_test.png")
print("Saved font_test.png")
