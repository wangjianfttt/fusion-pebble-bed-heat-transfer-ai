from check_hccb_p418_gpu_training_ready import assess


def test_gpu_readiness_requires_cuda_backward_and_measured_memory():
    assert assess(True, 20.0, 12.7, True)["ready"]
    assert not assess(False, 20.0, 12.7, True)["ready"]
    assert not assess(True, 12.0, 12.7, True)["ready"]
    assert not assess(True, 20.0, 12.7, False)["ready"]
