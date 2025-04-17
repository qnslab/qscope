import numpy as np


def gen_linear_sweep_list(low: int | float, high: int | float, num_steps: int):
    return np.linspace(low, high, num_steps)


def gen_centred_sweep_list(centre: int | float, span: int | float, num_points: int):
    return np.linspace(centre - span / 2, centre + span / 2, num_points)


def gen_gauss_sweep_list(centre: int | float, span: int | float, num_steps: int):
    # Calculate the standard deviation from the span
    std_dev = span / 6  # Assuming 99.7% of data within the span (3 stdev each side)
    # Generate normally distributed frequencies with 10 times more points than steps
    s_list = np.random.normal(loc=centre, scale=std_dev, size=10 * num_steps)
    # Order the frequencies
    s_list.sort()
    # Take every 5th point
    s_list = s_list[::10]
    return s_list


def gen_multigauss_sweep_list(centres, spans, steps_per_section):
    if isinstance(spans, (int, float)):
        spans = [spans] * len(centres)

    if len(centres) != len(spans):
        raise ValueError("The length of centre_freqs and spans must be the same.")

    rf_list = []
    for centre_freq, span in zip(centres, spans):
        std_dev = span / 6  # Assuming 99.7% of data within the span (3 stdev each side)
        # generate normally distributed frequencies with 10 times more points than steps
        new_section = np.random.normal(
            loc=centre_freq, scale=std_dev, size=10 * steps_per_section
        )
        # order the frequencies
        new_section.sort()
        # take every 10th point
        rf_list.extend(new_section[5::10])

    return np.array(rf_list)


def gen_multicentre_sweep_list(centres, spans, steps_per_section):
    if isinstance(spans, (int, float)):
        spans = [spans] * len(centres)

    if len(centres) != len(spans):
        raise ValueError("The length of centre_freqs and spans must be the same.")

    rf_list = []
    for centre_freq, span in zip(centres, spans):
        step_size = span / steps_per_section
        if centres[0] > centres[1]:
            rf_list.extend(
                [
                    centre_freq + span / 2 - step_size * i
                    for i in range(steps_per_section)
                ]
            )
        else:
            rf_list.extend(
                [
                    centre_freq - span / 2 + step_size * i
                    for i in range(steps_per_section)
                ]
            )

    return np.array(rf_list)


def gen_exp_tau_list(low, high, num_steps):
    return np.logspace(np.log10(low), np.log10(high), num_steps)


def gen_exp_centered_list(center, span, num_steps):
    return np.logspace(
        np.log10(center - span / 2), np.log10(center + span / 2), num_steps
    )
