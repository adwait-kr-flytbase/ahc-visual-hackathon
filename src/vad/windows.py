"""Window planner. Short clips -> one window. Long videos -> sliding windows."""


def plan(duration, win=20.0, hop=10.0, single_max=30.0):
    if duration <= single_max:
        return [(0.0, round(duration, 3))]
    out, t = [], 0.0
    while t < duration:
        t1 = min(t + win, duration)
        out.append((round(t, 3), round(t1, 3)))
        if t1 >= duration:
            break
        t += hop
    return out
