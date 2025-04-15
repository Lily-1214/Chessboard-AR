import cv2
import numpy as np

# 영상 경로 및 출력 파일명
video_file = 'recorded_video.mp4'
out_file = 'pose_output.mp4'

# 카메라 파라미터
K = np.array([[954.16431808, 0, 628.52580128],
              [0, 941.38014155, 349.91132073],
              [0, 0, 1]])
dist_coeff = np.array([-0.57718832, 0.00269906, 0.00406432, 0.00878291, -0.02297618])

# 체스보드 설정
board_pattern = (10, 7)
board_cellsize = 0.025  # 2.5cm

# 체스보드의 3D 좌표 생성 (슬라이드 참고)
obj_points = board_cellsize * np.array(
    [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])], dtype=np.float32
)

# 영상 열기
video = cv2.VideoCapture(video_file)
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

# 원기둥 AR 생성 함수
def create_cylinder(center, radius, height, segments=36):
    angle_step = 2 * np.pi / segments
    top_circle = [
        [center[0] + radius * np.cos(i * angle_step),
         center[1] + radius * np.sin(i * angle_step),
         center[2] - height] for i in range(segments)
    ]
    bottom_circle = [
        [center[0] + radius * np.cos(i * angle_step),
         center[1] + radius * np.sin(i * angle_step),
         center[2]] for i in range(segments)
    ]
    return np.array(top_circle + bottom_circle, dtype=np.float32).reshape(-1, 1, 3)

# 원기둥 중심 위치들 (체스보드 위)
cylinder_centers = board_cellsize * np.array([[3, 3, 0], [6, 3, 0], [3, 6, 0], [6, 6, 0] ], dtype=np.float32)

while True:
    valid, img = video.read()
    if not valid:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    board_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found, corners = cv2.findChessboardCorners(gray, board_pattern, board_criteria)

    if found:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), board_criteria)
        ret, rvec, tvec = cv2.solvePnP(obj_points, corners, K, dist_coeff)

        for center in cylinder_centers:
            cylinder_pts = create_cylinder(center, radius=0.01, height=0.05, segments=36)
            imgpts, _ = cv2.projectPoints(cylinder_pts, rvec, tvec, K, dist_coeff)
            imgpts = imgpts.reshape(-1, 2)

            # 그리기 (위, 아래 원 연결)
            n = len(imgpts) // 2
            for i in range(n):
                pt1 = tuple(imgpts[i].astype(int))
                pt2 = tuple(imgpts[(i+1)%n].astype(int))
                pt3 = tuple(imgpts[i+n].astype(int))
                pt4 = tuple(imgpts[(i+1)%n + n].astype(int))
                cv2.line(img, pt1, pt2, (255, 0, 255), 1)
                cv2.line(img, pt3, pt4, (255, 0, 255), 1)
                cv2.line(img, pt1, pt3, (255, 0, 255), 1)

        # 카메라 위치 출력
        R, _ = cv2.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f"XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]"
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('AR Cylinders', img)
    out.write(img)

    key = cv2.waitKey(int(1000 // fps)) & 0xFF
    if key == 27:
        break

video.release()
out.release()
cv2.destroyAllWindows()
