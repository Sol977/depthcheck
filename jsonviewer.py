import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# 1. 파일 경로 설정
# 사용자분이 알려주신 WSL 경로를 Windows에서 접근 가능한 형식으로 지정했습니다.
# 만약 파일명이 planes.json이 아니라면 끝부분을 수정해주세요.
file_path = r"/home/sori/depthlearner/out/planes/planes.json"

# 경로 확인
if not os.path.exists(file_path):
    print(f"오류: 파일을 찾을 수 없습니다.\n경로를 확인해주세요: {file_path}")
    # 만약 WSL(Ubuntu) 내부에서 이 파이썬을 실행한다면 경로는 다음과 같아야 합니다:
    print("TIP: WSL 우분투 터미널 내부에서 실행 중이라면 경로는 '/home/sori/depthlearner/out/planes/planes.json' 이어야 합니다.")
else:
    print(f"파일을 읽어옵니다: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 시각화 설정
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    planes = data.get('planes', [])
    print(f"총 {len(planes)}개의 평면 데이터를 시각화합니다.")

    # 각 평면 그리기
    for plane in planes:
        hull_xyz = plane.get('hull_xyz')
        
        if hull_xyz:
            # 좌표 변환
            vertices = [np.array(hull_xyz)]
            
            # 다각형 생성 (투명도 0.6, 테두리 검은색)
            # 색상은 ID에 따라 다르게 설정
            poly = Poly3DCollection(vertices, alpha=0.6, edgecolor='k', linewidths=0.5)
            color = plt.cm.jet(plane['id'] / len(planes)) # ID 기반 색상 매핑
            poly.set_facecolor(color)
            
            ax.add_collection3d(poly)

    # 축 범위 자동 설정
    all_points = []
    for plane in planes:
        for p in plane['hull_xyz']:
            all_points.append(p)
    
    if all_points:
        all_points = np.array(all_points)
        ax.set_xlim(all_points[:,0].min(), all_points[:,0].max())
        ax.set_ylim(all_points[:,1].min(), all_points[:,1].max())
        ax.set_zlim(all_points[:,2].min(), all_points[:,2].max())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Planes Visualization ({len(planes)} planes)")
    
    # 마우스로 회전 가능하도록 설정 후 보여주기
    plt.show()