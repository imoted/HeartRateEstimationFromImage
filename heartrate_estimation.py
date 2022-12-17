import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy.fft import fft, fftfreq
import time
from collections import deque

LENGTH_FFT = 256
RRI_FFT = 32

RRI_PEAK_THRETHOLD = 8
HR_FFT_PEAK_THRETHOLD = 5

enable_plot = True
enable_video = True

resolution = [640, 480]

W_B = resolution[0] // 2 - resolution[0] // 8
W_E = resolution[0] // 2 + resolution[0] // 8
H_B = resolution[1] // 2 - resolution[1] // 8
H_E = resolution[1] // 2 + resolution[1] // 8
RES_SIZE = resolution[0] * resolution[1]

FPS = 30
average_window = 3

HR_MIN = 50
HR_MAX = 150
RP_MIN = 10
RP_MAX = 50


def moving_average(x, w) -> np.ndarray:
    # return np.convolve(x, np.ones(w), 'same') / w  # 数が変わってしまう為エラーになる
    func_ret = np.array([])
    for i in range(len(x)):
        temp_sum = 0
        for j in range(w):
            if i - j < 0:
                temp_sum += x[0]
            else:
                temp_sum += x[i - j]
        func_ret = np.append(func_ret, temp_sum / w)
    return func_ret


if __name__ == "__main__":
    cap = cv2.VideoCapture("/dev/video4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)

    index = 0
    x = np.empty(0)
    y = np.empty(0)
    last_time = time.time()
    one_sec_count = 0
    last_fps = 22

    hr_list = []
    hr_mean_list = deque()
    respiration_list = []
    respiration_mean_list = deque()

    mean_x_list = deque()
    time_list = deque()

    plt.figure(figsize=[6, 10])
    ax1 = plt.subplot(511)  # 緑の積算値のグラフ
    ax2 = plt.subplot(512)  # 上記のFFTのグラフ
    ax3 = plt.subplot(513)  # FFTした周波数から、脈拍と呼吸を推定したグラフ
    ax4 = plt.subplot(514)  # 緑の積算値のピークから脈と推定されるピーク間隔のグラフ
    ax5 = plt.subplot(515)  # 上記のFFTのグラフ
    plt.subplots_adjust(hspace=0.4)

    last_index = 0
    peak_index_list = np.empty(0)

    rri_mean_list = np.empty(0)
    rri_index_list = np.empty(0)
    rri_count = 0

    while True:
        ret, frame = cap.read()

        # 1secのフレーム数を計算
        time_list.append(time.time())
        if time_list[0] < time_list[-1] - 1:
            time_list.popleft()
            one_sec_count = len(time_list)
            # print("FPS: ", fps)

        # 緑の積算値を計算
        green_sum = frame[H_B:H_E, W_B:W_E].sum()
        x = np.append(x, index)
        y = np.append(y, green_sum / RES_SIZE)

        if len(x) > LENGTH_FFT:
            x = np.delete(x, 0)
            y = np.delete(y, 0)

        y_ave = moving_average(y, average_window)

        if len(x) > average_window:
            if enable_plot:
                ax1.cla()
                ax1.plot(x, y)
                ax1.plot(x, y_ave)

                ax1.grid()
                # ax1.set_title("green channel")
                ax1.set_xlabel("time")
                ax1.set_ylabel("green value")

        # 1秒毎に、移動平均値を、ピーク検出
        if y_ave.size > 0 and index >= last_index + one_sec_count:
            last_index = index
            prominence = (y_ave.max() - y_ave.mean()) / RRI_PEAK_THRETHOLD
            peaks, _ = signal.find_peaks(y_ave, prominence=prominence, distance=5)
            if enable_plot:
                ax1.scatter(x[peaks], y_ave[peaks], color="red")

            new_peaks = peaks[peaks > x.size - one_sec_count]
            # print("peak_index_list", peak_index_list)

            # ピークの差分を出して、異常値を省く
            if new_peaks.size > 0:
                peak_index_list = np.append(peak_index_list, x[new_peaks])
                if peak_index_list.size > 2:
                    peak_diff_list = np.diff(peak_index_list) / one_sec_count * 60
                    # print("peak_diff_list", peak_diff_list)
                    peak_diff_list = peak_diff_list[peak_diff_list < 150]
                    peak_diff_list = peak_diff_list[peak_diff_list > 50]

                    # mean をグラフ表示
                    # print("peak_diff_list", peak_diff_list)
                    # if peak_diff_list.size > 0:
                    #     peak_diff_mean = peak_diff_list.mean()
                    #     rri_mean_list = np.append(rri_mean_list, peak_diff_mean)
                    #     rri_count += 1
                    #     rri_index_list = np.append(rri_index_list, rri_count)
                    #     if enable_plot:
                    #         ax4.cla()
                    #         ax4.plot(rri_index_list, rri_mean_list)
                    #         ax4.grid()
                    #         ax4.set_ylabel("heart rate[beats/min]")

                    # 新しく取得できた値をそのままグラフ表示
                    rri_count += 1
                    rri_index_list = np.arange(peak_diff_list.size)
                    if enable_plot:
                        ax4.cla()
                        ax4.plot(rri_index_list, peak_diff_list)
                        ax4.grid()
                        ax4.set_ylabel("heart rate[beats/min]")

                    # FFT して LF /HF算出のためのRRIの周波数解析
                    if peak_diff_list.size > RRI_FFT:
                        peak_diff_list = peak_diff_list[-RRI_FFT:]
                        peak_diff_fft = fft(peak_diff_list)
                        amp_fft = 2.0 / RRI_FFT * np.abs(peak_diff_list[: RRI_FFT // 2])
                        peak_diff_mean = peak_diff_list.mean()
                        freq = fftfreq(
                            RRI_FFT, d=1.0 / (peak_diff_mean / one_sec_count)
                        )[: RRI_FFT // 2]

                        if enable_plot:
                            ax5.cla()
                            ax5.plot(freq, amp_fft)

                            # ax5.set_xlim(0, 0.6)
                            ax5.set_xlabel("Frequency [Hz]")
                            ax5.set_ylabel("Amplitude [a.u.]")
                            ax5.grid()
            if peak_index_list.size > RRI_FFT * 2:
                peak_index_list = peak_index_list[-RRI_FFT * 2 :]

        # FFTして、脈拍と呼吸数の解析
        if len(x) >= LENGTH_FFT and time.time() - last_time > 1:
            last_time = time.time()

            y_fft = fft(y_ave)
            amp_fft = 2.0 / LENGTH_FFT * np.abs(y_fft[3 : LENGTH_FFT // 2])
            freq = fftfreq(LENGTH_FFT, d=1.0 / one_sec_count)[3 : LENGTH_FFT // 2]
            prominence = max(amp_fft) / HR_FFT_PEAK_THRETHOLD
            peaks, _ = signal.find_peaks(amp_fft, prominence=prominence)
            if enable_plot:
                ax2.cla()
                ax2.plot(freq, amp_fft)
                ax2.scatter(freq[peaks], amp_fft[peaks], color="red")

                ax2.set_xlim(0, 3)
                ax2.set_xlabel("Frequency [Hz]")
                ax2.set_ylabel("Amplitude [a.u.]")
                ax2.grid()

            # 脈拍数の平均化及び、異常値の排除
            for i in freq[peaks]:
                rate = i * 60
                if HR_MIN < rate < HR_MAX:
                    hr_list.append(rate)
            if hr_list:
                hr_mean_list.append(np.mean(hr_list))
            else:  # 正常に取得出来ないときは、0にする
                hr_mean_list.append(0)
            if len(hr_list) > 20:
                hr_list = hr_list[-20:]

            # 呼吸数の平均化及び、異常値の排除
            for i in freq[peaks]:
                rate = i * 60
                if RP_MIN < rate < RP_MAX:
                    respiration_list.append(rate)
            if respiration_list:
                respiration_mean_list.append(np.mean(respiration_list))
            else:  # 正常に取得出来ないときは、0にする
                respiration_mean_list.append(0)
            if len(respiration_list) > 10:
                respiration_list = respiration_list[-10:]
            # print("length of respiration list", len(respiration_list))
            # print("length of heart rate list", len(hr_list))

            # X軸の生成
            mean_x_list.append(index)
            if len(mean_x_list) > LENGTH_FFT:
                hr_mean_list.popleft()
                respiration_mean_list.popleft()
                mean_x_list.popleft()

            if enable_plot:
                ax3.cla()
                ax3.plot(mean_x_list, hr_mean_list)
                ax3.plot(mean_x_list, respiration_mean_list)
                ax3.set_xlabel("Time [s]")
                ax3.set_ylabel("Rate [bpm]")
                ax3.legend(["Heart Rate", "Respiration Rate"])
                ax3.grid()
            else:
                print(
                    "Heart Rate: ",
                    hr_mean_list[-1],
                    "Respiration Rate: ",
                    respiration_mean_list[-1],
                )

        one_sec_count += 1

        if enable_plot:
            plt.pause(0.01)

        if enable_video:
            cv2.rectangle(frame, (W_B, H_B), (W_E, H_E), (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        index += 1
