import numpy as np


def assert_equality(np_out1, np_out2, atol=1e-05):
    def max_abs_diff(np_out1, np_out2):
        return [
            np.max(np.absolute(arr1.flatten() - arr2.flatten()))
            for arr1, arr2 in zip(np_out1, np_out2)
        ]

    def correct_rate(np_out1, np_out2):
        def do_correct_rate(arr1, arr2):
            correct = np.isclose(
                arr1.flatten(), arr2.flatten(), atol=atol).sum()
            total = arr1.flatten().size
            return f'{correct}/{total}'
        return [
            do_correct_rate(arr1, arr2)
            for arr1, arr2 in zip(np_out1, np_out2)
        ]
    if not all(
        arr1.size == arr2.size
        for arr1, arr2 in zip(np_out1, np_out2)
    ):
        raise AssertionError(f"Outputs dims doesn't match.\n"
            f"{len(np_out1)=} "
            f"{len(np_out2)=}\n"
            f"{[x.size for x in np_out1]=}\n"
            f"{[x.size for x in np_out2]=}\n"
            )

    res = max_abs_diff(np_out1, np_out2)
    if not all(
        np.allclose(arr1.flatten(), arr2.flatten(), atol=atol)
        for arr1, arr2 in zip(np_out1, np_out2)
    ):
        raise AssertionError("Output of np_out1 and np_out2 doesn't match.\n"
            f"Correct rate: {correct_rate(np_out1, np_out2)}\n"
            f"{np_out1=}\n"
            f"{np_out2=}\n"
            f"Max abs diff: {res}"
        )
    return res
