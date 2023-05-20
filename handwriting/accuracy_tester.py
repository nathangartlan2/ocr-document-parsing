import numpy as np
import math


# Assesses the CER of a single image and label
# LIMITATION: does not behave well when the
# label and inference values are of different
# lengths
def character_error_rate(ground_truth, inference_label):

    num_errors = 0
    ground_truth_chars = len(ground_truth)
    inference_chars = len(inference_label)

    len_difference = abs(ground_truth_chars - inference_chars)
    num_errors += len_difference

    if ground_truth_chars > inference_chars:
        ground_truth = ground_truth[:inference_chars]
    elif inference_chars > ground_truth_chars:
        inference_label = inference_label[:ground_truth_chars]

    assert(len(ground_truth) == len(inference_label))

    for i in range(0, len(ground_truth)):
        if ground_truth[i] != inference_label[i]:
            num_errors += 1

    rate = num_errors/ground_truth_chars

    if(num_errors > ground_truth_chars):
        rate = 1
    elif ground_truth_chars == 0:
        rate = 1

    if np.isnan(rate):
        print("found an issue")

    return rate


def full_set_CER(
        ground_truths,
        inference_label,
        func_CER):
    lenght_truth = len(ground_truths)
    errors = []

    for i in range(lenght_truth):
        error_rate = func_CER(ground_truths[i], inference_label[i])
        errors.append(error_rate)
    return np.mean(np.array(errors))
