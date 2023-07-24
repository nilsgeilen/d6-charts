from collections import Counter
from functools import reduce
from itertools import combinations
from typing import Dict, Set, List

import matplotlib.pyplot as plt
import os


def chart(title, data: Dict[int, float], xlabel=None):
    plt.figure(figsize=(7, 4))
    plt.xticks(range(min(data), max(data) + 1))
    plt.bar(data.keys(), data.values())
    plt.ylabel('chance [%]')
    if xlabel:
        plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(os.path.join('repo', title), dpi=200)
    plt.close()


def multi_chart(title, data: Dict[str, Dict[int, float]], bar=False, ylabel='chance [%]', xlabel=None):
    plt.figure(figsize=(7, 4))
    all_xvals = [key for series in data.values() for key in series.keys()]
    plt.xticks(range(min(all_xvals), max(all_xvals) + 1))

    if bar:
        width = 0.8 / len(data)
        for i, label in enumerate(data):
            xvals = [key + (i + 0.5) * width - 0.4 for key in data[label].keys()]
            plt.bar(xvals, data[label].values(), label=label, width=width)
    else:
        for label in data:
            xvals = sorted(data[label])
            yvals = [data[label][key] for key in xvals]
            plt.plot(xvals, yvals, label=label, linewidth=3, marker='o')

    plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join('repo', title), dpi=200)
    plt.close()


def combine(n):
    dice = [x + 1 for x in range(6)] * n
    return set(combinations(dice, n))


def percentage(f, combos: Set[int]) -> Dict[int, float]:
    val_cnt = Counter(map(f, combos))
    sigma = sum(val_cnt.values())
    return {key: val / sigma * 100.0 for key, val in val_cnt.items()}


def cumulative(percentage: Dict[int, float]) -> Dict[int, float]:
    sigma = 0.0
    result = {}
    for key in reversed(sorted(percentage)):
        sigma += percentage[key]
        result[key] = sigma
    return result


def combine_percentages(l: List[Dict[int, float]]) -> Dict[int, float]:
    def combine2(a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
        counter = Counter()
        for keya in a:
            for keyb in b:
                counter[keya + keyb] += a[keya] * b[keyb] / 100.0
        return counter

    return reduce(combine2, l)


def prune_percentage(percentage: Dict[int, float], threshold=0.1) -> Dict[int, float]:
    return {key: val for key, val in percentage.items() if val >= threshold}


def expected_value(percentage: Dict[int, float]) -> float:
    result = 0.0
    for val, chance in percentage.items():
        result += val * chance
    return result / 100.0


delta = lambda l: max(l) - min(l)


def hits(threshold: int):
    return lambda l: sum(1 if x >= threshold else 0 for x in l)


def explosive_die(threshold: int, computational_depth: int):
    sequences = combine(computational_depth)

    def explode(sequence):
        i = 0
        for roll in sequence:
            if roll >= threshold:
                i += 1
            if roll != 6:
                break
        return i

    return percentage(explode, sequences)


chart('Probability of 1 d6', percentage(sum, combine(1)), xlabel='Σ(1 d6)')
chart('Probability of the Sum of 2 d6', percentage(sum, combine(2)), xlabel='Σ(2 d6)')
chart('Probability of the Sum of 3 d6', percentage(sum, combine(3)), xlabel='Σ(3 d6)')
chart('Probability of the Sum of 4 d6', percentage(sum, combine(4)), xlabel='Σ(4 d6)')

multi_chart('Cumulative Probability of the Sum of n d6',
            {f'Σ({i} d6)': cumulative(percentage(sum, combine(i))) for i in range(2, 5)},
            ylabel='cumulative chance [%]')

multi_chart('Probability of the Maximum of n d6', {f'max({i} d6)': percentage(max, combine(i)) for i in range(2, 5)},
            bar=True)

multi_chart('Probability of the Minimum of 2 d6', {
    'min(2 d6)': percentage(min, combine(2)),
    'min(2 d6 w/o  doubles)': percentage(min, {x for x in combine(2) if list(x)[0] != list(x)[1]})
}, bar=True)

chart('Probability of the Difference between 2 d6', percentage(delta, combine(2)), xlabel='Δ(2 d6)')
chart('Π(2 d6)', percentage(lambda l: l[0] * l[1], combine(2)))

for i in range(1, 7):
    multi_chart(f'Probability of Hits with {i} d6',
                {f'{threshold}+, μ = {round(expected_value(percentage), 1)}': percentage for threshold, percentage in
                 [(k, percentage(hits(k), combine(i))) for k in range(2, 7)]}, xlabel='hits')

    multi_chart(f'Cumulative Probability of Hits with {i} d6',
                {f'{threshold}+': cumulative(percentage(hits(threshold), combine(i))) for threshold in range(2, 7)},
                ylabel='cumulative chance [%]', xlabel='hits')

    for threshold in range(2, 7):
        c5d6e = explosive_die(threshold, 7)
    for i in range(1, 7):
        p1 = percentage(hits(threshold), combine(i))
    mu1 = round(expected_value(p1), 1)
    p2 = combine_percentages([c5d6e] * i)
    mu2 = round(expected_value(p2), 1)
    data = {f'{threshold}+, μ = {mu1}': p1,
            f'{threshold}+ explosive, μ = {mu2}': prune_percentage(p2)}
    multi_chart(f'Probability of Hits with {i} d6 ({threshold}+)', data, bar=True, xlabel='hits')

for i in range(2, 7):
    multi_chart(f'Probability of Hits with n d6 ({i}+)',
                {f'{n} d6, μ = {round(expected_value(percentage), 1)}': percentage for n, percentage in
                 [(n, percentage(hits(i), combine(n))) for n in range(1, 7)]}, xlabel='hits')