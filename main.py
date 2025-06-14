import cv2 as cv
import numpy as np
from math import sqrt
from math import sqrt


class DistanceCalculator:
    """
    Класс для вычисления расстояний между точками, от точек до линий,
    а также вычисления расстояний в сантиметрах от метки до начала координат.
    """

    @staticmethod
    def dist(x1: float, y1: float, x2: float, y2: float) -> float:
        "Вычисляет расстояние между двумя точками (x1, y1) и (x2, y2)."
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def dist_line(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> float:
        "Вычисляет расстояние от точки (x0, y0) до линии, проходящей через точки (x1, y1) и (x2, y2)."

        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - x1 * y2)
        denominator = DistanceCalculator.dist(x1, y1, x2, y2)
        return numerator / denominator

    @staticmethod
    def calculate_cm_distance(x: list[float], y: list[float]) -> tuple[float, float]:
        "Вычисляет расстояния в сантиметрах от метки  до начала координат."

        rx = (DistanceCalculator.dist_line(x[3], y[3], x[0], y[0], x[2], y[2]) * 500 / DistanceCalculator.dist_line(
            x[1], y[1], x[0], y[0], x[2], y[2]))
        ry = (DistanceCalculator.dist_line(x[3], y[3], x[0], y[0], x[1], y[1]) * 500 / DistanceCalculator.dist_line(
            x[2], y[2], x[0], y[0], x[1], y[1]))

        return int(rx) / 10, int(ry) / 10


def metki(frame):
    "распознаёт метки, опредеяет их положение и возвращает координаты их центров в определённом порядке"
    xc = []
    yc = []
    x = [0, 0, 0, 0]
    y = [0, 0, 0, 0]

    mask = cv.inRange(frame, (0, 0, 0), (100, 100, 100))
    mask_m = cv.inRange(frame, (0, 160, 0), (35, 200, 35))
    mask0 = cv.inRange(frame, (230, 230, 230), (255, 255, 255))
    mask_y = cv.inRange(frame, (135, 35, 20), (255, 75, 50))
    mask_x = cv.inRange(frame, (220, 0, 230), (255, 20, 255))

    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
    contours_m = cv.findContours(mask_m, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]

    if contours:
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        (x0, y0, w0, h0) = cv.boundingRect(contours[0])
        (x1, y1, w1, h1) = cv.boundingRect(contours[1])
        (x2, y2, w2, h2) = cv.boundingRect(contours[2])

        xc.append(x0 + w0 // 2)
        yc.append(y0 + h0 // 2)
        xc.append(x1 + w1 // 2)
        yc.append(y1 + h1 // 2)
        xc.append(x2 + w2 // 2)
        yc.append(y2 + h2 // 2)

        cv.rectangle(frame, (xc[0] - 20, yc[0] - 20), (xc[0] + 20, yc[0] + 20), (255, 0, 0), 2)
        cv.rectangle(frame, (xc[1] - 20, yc[1] - 20), (xc[1] + 20, yc[1] + 20), (0, 255, 0), 2)
        cv.rectangle(frame, (xc[2] - 20, yc[2] - 20), (xc[2] + 20, yc[2] + 20), (0, 0, 255), 2)

    if contours_m:
        contours_m = sorted(contours_m, key=cv.contourArea, reverse=True)
        (xx, yy, w, h) = cv.boundingRect(contours_m[0])
        x[3] = xx + w // 2
        y[3] = yy + h // 2
        cv.rectangle(frame, (x[3] - 20, y[3] - 20), (x[3] + 20, y[3] + 20), (255, 0, 255), 2)

    for i in range(3):
        s0 = np.sum(mask0[yc[i] - 20: yc[i] + 20, xc[i] - 20: xc[i] + 20])
        sx = np.sum(mask_x[yc[i] - 20: yc[i] + 20, xc[i] - 20: xc[i] + 20])
        sy = np.sum(mask_y[yc[i] - 20: yc[i] + 20, xc[i] - 20: xc[i] + 20])

        if s0 == max(s0, sx, sy):
            x[0] = xc[i]
            y[0] = yc[i]
        if sx == max(s0, sx, sy):
            x[1] = xc[i]
            y[1] = yc[i]
        if sy == max(s0, sx, sy):
            x[2] = xc[i]
            y[2] = yc[i]

    return x, y, frame


def main():
    """читаем файл с изображением,
    обрабатываем его для получение координат точек,
    переводим координаты точек в пиксилях в координаты метки в см,
    выводим координаты метки и изображение с выделенными метками"""
    frame = cv.imread("3.jpg")
    frame = cv.resize(frame, (800, 800))
    x, y, frame = metki(frame)
    rx, ry = DistanceCalculator.calculate_cm_distance(x, y)
    cv.putText(frame, f"X: {rx:.1f} cm", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv.putText(frame, f"Y: {ry:.1f} cm", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print("x (cm)", rx)
    print("y (cm)", ry)

    while True:
        cv.imshow("frame", frame)
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
