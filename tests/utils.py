import torch
import numpy as np


# Testing utils copied from tt-metal/tests/ttnn/utils_for_testing.py
def comp_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        # logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        # logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        # logger.error("One tensor is all zero")
        return False, 0.0

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, 1.0

    return cal_pcc >= pcc, cal_pcc


def construct_pcc_assert_message(
    message, expected_pytorch_result, actual_pytorch_result
):
    messages = []
    messages.append(message)
    # messages.append("Expected")
    # messages.append(str(expected_pytorch_result))
    # messages.append("Actual")
    # messages.append(str(actual_pytorch_result))
    messages = [str(m) for m in messages]
    return "\n".join(messages)


def assert_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    pcc_passed, pcc_message = comp_pcc(
        expected_pytorch_result, actual_pytorch_result, pcc
    )
    assert pcc_passed, construct_pcc_assert_message(
        pcc_message, expected_pytorch_result, actual_pytorch_result
    )
    return pcc_passed, pcc_message


def check_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = comp_pcc(
        expected_pytorch_result, actual_pytorch_result, pcc
    )
    return pcc_passed, construct_pcc_assert_message(
        pcc_message, expected_pytorch_result, actual_pytorch_result
    )


def calculate_accuracy(original_outputs, compiled_outputs):
    if isinstance(original_outputs, dict) and isinstance(compiled_outputs, dict):
        # Handle case where outputs can be converted to dictionaries
        original_outputs = dict(original_outputs)
        compiled_outputs = dict(compiled_outputs)
        output_pccs = []
        for key in original_outputs.keys() & compiled_outputs.keys():
            _, pcc = comp_pcc(original_outputs[key], compiled_outputs[key])
            output_pccs.append(pcc)
        accuracy = torch.mean(torch.tensor(output_pccs)).item()
    elif isinstance(original_outputs, torch.Tensor) and isinstance(
        compiled_outputs, torch.Tensor
    ):
        # Handle case where outputs are Pytorch Tensors
        _, accuracy = comp_pcc(original_outputs, compiled_outputs)
    else:
        accuracy = None
    return accuracy
