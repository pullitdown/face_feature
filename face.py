import cv2
import numpy as np
import dlib
import time
from scipy import signal

# 常量
WINDOW_TITLE = 'Pulse Observer'
BUFFER_MAX_SIZE = 500       # 存储的近期ROI平均值的数量
MAX_VALUES_TO_GRAPH = 50    # 在脉冲图中显示最近的ROI平均值
MIN_HZ = 0.83       # 50 BPM - 最小允许心率
MAX_HZ = 3.33       # 200 BPM - 最大允许心率
MIN_FRAMES = 100    # 在计算心率之前所需的最小帧数
DEBUG_MODE = False  # 是否更精确


# 创建指定的Butterworth过滤器并应用
def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


# 获取前额的区域
def get_forehead_roi(face_points):
    # 将这些点存储在Numpy数组中，可以很容易地通过切片得到x和y的最小值和最大值
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)
    # 前额两眉间区域
    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom)


# 获取鼻子的区域
def get_nose_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)
    # 鼻子和脸颊
    min_x = int(points[36, 0])
    min_y = int(points[28, 1])
    max_x = int(points[45, 0])
    max_y = int(points[33, 1])
    left = min_x
    right = max_x
    top = min_y + (min_y * 0.02)
    bottom = max_y + (max_y * 0.02)
    return int(left), int(right), int(top), int(bottom)


# 获取全部区域，包括前额、眼睛和鼻子
# Note:额头和鼻子的组合效果更好。这可能是因为这个ROI包括眼睛，而眨眼会增加噪音
def get_full_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)
    # 去掉勾勒下颚轮廓的点，只保留与脸部内部特征相对应的点(如嘴、鼻子、眼睛、眉毛)
    min_x = int(np.min(points[17:47, 0]))
    min_y = int(np.min(points[17:47, 1]))
    max_x = int(np.max(points[17:47, 0]))
    max_y = int(np.max(points[17:47, 1]))
    center_x = min_x + (max_x - min_x) / 2
    left = min_x + int((center_x - min_x) * 0.15)
    right = max_x - int((max_x - center_x) * 0.15)
    top = int(min_y * 0.88)
    bottom = max_y
    return int(left), int(right), int(top), int(bottom)


def sliding_window_demean(signal_values, num_windows):
    window_size = int(round(len(signal_values) / num_windows))
    demeaned = np.zeros(signal_values.shape)
    for i in range(0, len(signal_values), window_size):
        if i + window_size > len(signal_values):
            window_size = len(signal_values) - i
        curr_slice = signal_values[i: i + window_size]
        if DEBUG_MODE and curr_slice.size == 0:
            print('Empty Slice: size={0}, i={1}, window_size={2}'.format(signal_values.size, i, window_size))
            print(curr_slice)
        demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
    return demeaned


# 对两个像素数组的绿色值取平均值
def get_avg(roi1, roi2):
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    return avg


# 返回列表中的最大绝对值
def get_max_abs(lst):
    return max(max(lst), -min(lst))


# 在GUI窗口中绘制心率图
def draw_graph(signal_values, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / MAX_VALUES_TO_GRAPH
    # 基于绝对值最大的值自动垂直缩放
    max_abs = get_max_abs(signal_values)
    scale_factor_y = (float(graph_height) / 2.0) / max_abs

    midpoint_y = graph_height / 2
    for i in range(0, len(signal_values) - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
    return graph


# 在GUI窗口中绘制心率文本(BPM)
def draw_bpm(bpm_str, bpm_width, bpm_height):
    bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
    bpm_text_size, bpm_text_base = cv2.getTextSize(bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                   thickness=2)
    bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
    bpm_text_y = int(bpm_height / 2 + bpm_text_base)
    cv2.putText(bpm_display, bpm_str, (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2.7, color=(0, 255, 0), thickness=2)
    bpm_label_size, bpm_label_base = cv2.getTextSize('BPM', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                     thickness=1)
    bpm_label_x = int((bpm_width - bpm_label_size[0]) / 2)
    bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
    cv2.putText(bpm_display, 'BPM', (bpm_label_x, bpm_label_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 255, 0), thickness=1)
    return bpm_display


# 在GUI窗口中绘制当前每秒帧数
def draw_fps(frame, fps):
    cv2.rectangle(frame, (0, 0), (100, 30), color=(0, 0, 0), thickness=-1)
    cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (5, 20), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 255, 0))
    return frame


# 在图形区域中绘制文本
def draw_graph_text(text, color, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    text_size, text_base = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
    text_x = int((graph_width - text_size[0]) / 2)
    text_y = int((graph_height / 2 + text_base))
    cv2.putText(graph, text, (text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color,
                thickness=1)
    return graph


# 计算每分钟的脉搏(BPM)
def compute_bpm(filtered_values, fps, buffer_size, last_bpm):
    # 计算FFT
    fft = np.abs(np.fft.rfft(filtered_values))
    # 生成与FFT值相对应的频率列表
    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)
    # 滤除FFT中不在[MIN_HZ, MAX_HZ]范围内的全部峰值，因为无BPM值可对应它们
    while True:
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < MIN_HZ or bps > MAX_HZ:
            if DEBUG_MODE:
                print('BPM of {0} was discarded.'.format(bps * 60.0))
            fft[max_idx] = 0
        else:
            bpm = bps * 60.0
            break
    # 在样本之间，心率变化不可能超过10%，所以使用加权平均来使BPM与最后一个BPM平滑
    if last_bpm > 0:
        bpm = (last_bpm * 0.9) + (bpm * 0.1)
    return bpm


# 过滤信号数据
def filter_signal_data(values, fps):
    # 确保数组没有无限或NaN值
    values = np.array(values)
    np.nan_to_num(values, copy=False)
    # 通过消除趋势和贬低来使信号平滑
    detrended = signal.detrend(values, type='linear')
    demeaned = sliding_window_demean(detrended, 15)
    # 用butterworth带通滤波器对信号进行滤波
    filtered = butterworth_filter(demeaned, MIN_HZ, MAX_HZ, fps, order=5)
    return filtered


# 获取感兴趣区域的平均值，并在周围画一个绿色的矩形
def get_roi_avg(frame, view, face_points, draw_rect=True):
    # 得到感兴趣的区域
    fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
    nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)
    # 在感兴趣区域(ROI)周围绘制绿色矩形
    if draw_rect:
        cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
        cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)
    # 将感兴趣的区域(ROI)平均分割
    fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
    nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
    return get_avg(fh_roi, nose_roi)


# 主要方法
def run_pulse_observer(detector, predictor, webcam, window):
    roi_avg_values = []
    graph_values = []
    times = []
    last_bpm = 0
    graph_height = 200
    graph_width = 0
    bpm_display_width = 0
    # 当窗口被用户关闭 cv2.getWindowProperty()返回-1
    while cv2.getWindowProperty(window, 0) == 0:
        ret_val, frame = webcam.read()
        # 如果无法从网络摄像头读取 ret_val == False
        if not ret_val:
            print("ERROR:  Unable to read from webcam.  Was the webcam disconnected?  Exiting.")
            shut_down(webcam)
        # 在画框之前先把画框复制一份，我们将在GUI中显示副本，原始帧将用于计算心率
        view = np.array(frame)
        # 心率图占窗口宽度的75%，BPM获得25%
        if graph_width == 0:
            graph_width = int(view.shape[1] * 0.75)
            if DEBUG_MODE:
                print('Graph width = {0}'.format(graph_width))
        if bpm_display_width == 0:
            bpm_display_width = view.shape[1] - graph_width
        # 使用dlib检测人脸
        faces = detector(frame, 0)
        if len(faces) == 1:
            face_points = predictor(frame, faces[0])
            roi_avg = get_roi_avg(frame, view, face_points, draw_rect=True)
            roi_avg_values.append(roi_avg)
            times.append(time.time())
            # Buffer已经满了，从顶部弹出值来删除它
            if len(times) > BUFFER_MAX_SIZE:
                roi_avg_values.pop(0)
                times.pop(0)
            curr_buffer_size = len(times)
            # 在有最小帧数之前，不要计算脉搏
            if curr_buffer_size > MIN_FRAMES:
                # 计算相关的次数
                time_elapsed = times[-1] - times[0]
                fps = curr_buffer_size / time_elapsed  # frames per second
                # 清理信号数据
                filtered = filter_signal_data(roi_avg_values, fps)
                graph_values.append(filtered[-1])
                if len(graph_values) > MAX_VALUES_TO_GRAPH:
                    graph_values.pop(0)
                # 绘制脉搏图
                graph = draw_graph(graph_values, graph_width, graph_height)
                # 计算并显示BPM
                bpm = compute_bpm(filtered, fps, curr_buffer_size, last_bpm)
                bpm_display = draw_bpm(str(int(round(bpm))), bpm_display_width, graph_height)
                last_bpm = bpm
                # 显示FPS
                if DEBUG_MODE:
                    view = draw_fps(view, fps)
            else:
                # 如果没有足够的数据来计算脉搏，则显示一个带有加载文本和BPM占位符的空图
                pct = int(round(float(curr_buffer_size) / MIN_FRAMES * 100.0))
                loading_text = 'Computing pulse: ' + str(pct) + '%'
                graph = draw_graph_text(loading_text, (0, 255, 0), graph_width, graph_height)
                bpm_display = draw_bpm('--', bpm_display_width, graph_height)

        else:
            # 没有检测到脸，所以我们必须清除值和时间列表，否则，当再次检测到人脸时，时间就会出现空白。
            del roi_avg_values[:]
            del times[:]
            graph = draw_graph_text('No face detected', (0, 0, 255), graph_width, graph_height)
            bpm_display = draw_bpm('--', bpm_display_width, graph_height)

        graph = np.hstack((graph, bpm_display))
        view = np.vstack((view, graph))

        cv2.imshow(window, view)

        key = cv2.waitKey(1)
        # 如果用户按转义键退出
        if key == 27:
            shut_down(webcam)


# 关闭
def shut_down(webcam):
    webcam.release()
    cv2.destroyAllWindows()
    exit(0)


def main():
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    except RuntimeError as e:
        print('ERROR:  \'shape_predictor_68_face_landmarks.dat\' was not found in current directory.   '\
              'Download it from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2')
        return

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print('ERROR:  Unable to open webcam.  Verify that webcam is connected and try again.  Exiting.')
        webcam.release()
        return

    cv2.namedWindow(WINDOW_TITLE)
    run_pulse_observer(detector, predictor, webcam, WINDOW_TITLE)

    # 当用户关闭窗口时，Run_pulse_observer()返回，关闭摄像
    shut_down(webcam)


if __name__ == '__main__':
    main()