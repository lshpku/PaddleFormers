import queue

import paddle
from paddle import Tensor

_pinned_tensor_queue = queue.deque()
_cudart = paddle.cuda.cudart()


def wait_to_device():
    while _pinned_tensor_queue:
        _, event = _pinned_tensor_queue.popleft()
        event.synchronize()


def to_device(data, dtype=None) -> Tensor:
    if isinstance(data, Tensor) and data.place.is_cuda_pinned_place():
        pin_data = data
    else:
        pin_data = paddle.to_tensor(
            data, dtype=dtype, place=paddle.CUDAPinnedPlace()
        )
    gpu_data = paddle.empty_like(pin_data)

    err = _cudart.cudaMemcpyAsync(
        gpu_data.data_ptr(),
        pin_data.data_ptr(),
        pin_data.size * pin_data.itemsize,
        _cudart.cudaMemcpyHostToDevice,
        paddle.device.current_stream().stream_base.cuda_stream,
    )
    assert err == _cudart.cudaError.success, f"cudaMemcpyAsync failed: {err}"

    event = paddle.device.Event()
    event.record()
    _pinned_tensor_queue.append((pin_data, event))

    while _pinned_tensor_queue:
        _, event = _pinned_tensor_queue[0]
        if event.query():
            _pinned_tensor_queue.popleft()
        else:
            break

    return gpu_data


def to_host(data: Tensor):
    pin_data = paddle.zeros_like(data, device="cpu").pin_memory()

    err = _cudart.cudaMemcpyAsync(
        pin_data.data_ptr(),
        data.data_ptr(),
        data.size * data.itemsize,
        _cudart.cudaMemcpyDeviceToHost,
        paddle.device.current_stream().stream_base.cuda_stream,
    )
    assert err == _cudart.cudaError.success, f"cudaMemcpyAsync failed: {err}"

    return pin_data
