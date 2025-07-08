from enum import Enum, auto
from pathlib import Path
import time
import numpy as np
# from typing import Annotated, Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import onnxruntime as ort
import torch
import typer

from depth_anything_v2.config import Encoder, Metric
from depth_anything_v2.dpt import DepthAnythingV2


class ExportFormat(str, Enum):
    onnx = auto()
    pt2 = auto()


class InferenceDevice(str, Enum):
    cpu = auto()
    cuda = auto()


app = typer.Typer()


@app.callback()
def callback():
    """Depth-Anything Dynamo CLI"""


def multiple_of_14(value: int) -> int:
    if value % 14 != 0:
        raise typer.BadParameter("Value must be a multiple of 14.")
    return value


# @app.command()
# def export(
#     encoder: Annotated[Encoder, typer.Option()] = Encoder.vitb,
#     metric: Annotated[
#         Optional[Metric], typer.Option(help="Export metric depth models.")
#     ] = None,
#     output: Annotated[
#         Optional[Path],
#         typer.Option(
#             "-o",
#             "--output",
#             dir_okay=False,
#             writable=True,
#             help="Path to save exported model.",
#         ),
#     ] = None,
#     format: Annotated[
#         ExportFormat, typer.Option("-f", "--format", help="Export format.")
#     ] = ExportFormat.onnx,
#     batch_size: Annotated[
#         int,
#         typer.Option(
#             "-b",
#             "--batch-size",
#             min=0,
#             help="Batch size of exported ONNX model. Set to 0 to mark as dynamic.",
#         ),
#     ] = 1,
#     height: Annotated[
#         int,
#         typer.Option(
#             "-h",
#             "--height",
#             min=0,
#             help="Height of input image. Set to 0 to mark as dynamic.",
#             callback=multiple_of_14,
#         ),
#     ] = 518,
#     width: Annotated[
#         int,
#         typer.Option(
#             "-w",
#             "--width",
#             min=0,
#             help="Width of input image. Set to 0 to mark as dynamic.",
#             callback=multiple_of_14,
#         ),
#     ] = 518,
#     opset: Annotated[
#         int,
#         typer.Option(
#             max=17,
#             help="ONNX opset version of exported model. Defaults to 17.",
#         ),
#     ] = 17,
#     use_dynamo: Annotated[
#         bool,
#         typer.Option(
#             help="Use TorchDynamo (Beta) for ONNX export. Only supports static shapes and opset 18."
#         ),
#     ] = False,
# ):
#     """Export Depth-Anything V2 using TorchDynamo."""
#     if encoder == Encoder.vitg:
#         raise NotImplementedError("Depth-Anything-V2-Giant is coming soon.")

#     if torch.__version__ < "2.3":
#         typer.echo(
#             "Warning: torch version is lower than 2.3, export may not work properly."
#         )

#     if output is None:
#         output = Path(f"weights/depth_anything_v2_{encoder}_{opset}.{format}")

#     config = encoder.get_config(metric)
#     model = DepthAnythingV2(
#         encoder=encoder.value,
#         features=config.features,
#         out_channels=config.out_channels,
#         max_depth=20
#         if metric == Metric.indoor
#         else 80
#         if metric == Metric.outdoor
#         else None,
#     )
#     model.load_state_dict(torch.hub.load_state_dict_from_url(config.url))

#     if format == ExportFormat.onnx:
#         if use_dynamo:
#             typer.echo(
#                 "Exporting to ONNX using TorchDynamo (Beta). Only supports static shapes and opset 18."
#             )
#             onnx_program = torch.onnx.dynamo_export(
#                 model, torch.randn(batch_size or 1, 3, height or 518, width or 518)
#             )
#             onnx_program.save(str(output))
#         else:  # Use TS exporter.
#             typer.echo("Exporting to ONNX using legacy JIT tracer.")
#             dynamic_axes = {}
#             if batch_size == 0:
#                 dynamic_axes[0] = "batch_size"
#             if height == 0:
#                 dynamic_axes[2] = "height"
#             if width == 0:
#                 dynamic_axes[3] = "width"
#             torch.onnx.export(
#                 model,
#                 torch.randn(batch_size or 1, 3, height or 140, width or 140),
#                 str(output),
#                 input_names=["image"],
#                 output_names=["depth"],
#                 opset_version=opset,
#                 dynamic_axes={"image": dynamic_axes, "depth": dynamic_axes},
#             )
#     elif format == ExportFormat.pt2:
#         batch_dim = torch.export.Dim("batch_size")
#         export_program = torch.export.export(
#             model.eval(),
#             (torch.randn(2, 3, height or 518, width or 518),),
#             dynamic_shapes={
#                 "x": {0: batch_dim},
#             },
#         )
#         torch.export.save(export_program, output)


# @app.command()
# def infer(
#     model_path: Annotated[
#         Path,
#         typer.Argument(
#             exists=True, dir_okay=False, readable=True, help="Path to ONNX model."
#         ),
#     ],
#     image_path: Annotated[
#         Path,
#         typer.Option(
#             "-i",
#             "--img",
#             "--image",
#             exists=True,
#             dir_okay=False,
#             readable=True,
#             help="Path to input image.",
#         ),
#     ],
#     height: Annotated[
#         int,
#         typer.Option(
#             "-h",
#             "--height",
#             min=14,
#             help="Height at which to perform inference. The input image will be resized to this.",
#             callback=multiple_of_14,
#         ),
#     ] = 518,
#     width: Annotated[
#         int,
#         typer.Option(
#             "-w",
#             "--width",
#             min=14,
#             help="Width at which to perform inference. The input image will be resized to this.",
#             callback=multiple_of_14,
#         ),
#     ] = 518,
#     device: Annotated[
#         InferenceDevice, typer.Option("-d", "--device", help="Inference device.")
#     ] = InferenceDevice.cuda,
#     output_path: Annotated[
#         Optional[Path],
#         typer.Option(
#             "-o",
#             "--output",
#             dir_okay=False,
#             writable=True,
#             help="Path to save output depth map. If not given, show visualization.",
#         ),
#     ] = None,
# ):

def onnx_predict(session, input_data) -> np.ndarray:
    binding = session.io_binding()
    ort_input = session.get_inputs()[0].name
    # print(ort_input, session.get_outputs()[0])
    binding.bind_cpu_input(ort_input, input_data)
    binding.bind_output('depth', 'cuda')
    session.run_with_iobinding(binding)  # Actual inference happens here.
    output = binding.get_outputs()[0].numpy()
    return output


def preprocess(image, depth_shape):
    h, w = depth_shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)[None].astype("float32")
    return image

def normalize(depth):
    return (depth - depth.min()) / (depth.max() - depth.min()) #* 255.0
def postprocess(depth, original_shape):
    h, w = original_shape
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.transpose(1, 2, 0).astype("uint8")
    # depth = cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
    
    cmap = plt.get_cmap('Spectral_r')
    depth_map = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype("uint8")
    return depth_map

import click
@click.command()
@click.argument('model_path')
@click.argument('image_path')
@click.option('--output_path', default=None)
def infer(model_path, image_path, output_path, depth_shape=(518, 518), device=0):
    """Depth-Anything V2 inference using ONNXRuntime. No dependency on PyTorch."""
    
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitb' # or 'vits', 'vitb'
    torch_model = DepthAnythingV2(**{**model_configs[encoder]}).cuda() # , 'max_depth': max_depth}
    torch_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', 
                                           map_location='cpu'))
    torch_model.eval()
    # depth_torch = torch_model.infer_image(image)

    # Inference
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = False
    # sess_options.log_severity_level=1
    # For inspecting applied ORT-optimizations:
    # sess_options.optimized_model_filepath = "weights/optimized.onnx"
    print("Available providers:", ort.get_available_providers())
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider", ]  # 'TensorrtExecutionProvider'
    # if device == InferenceDevice.cuda:
    #     providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers
    )


    # Preprocessing, implement this part in your chosen language:
    raw_image = cv2.imread(str(image_path))
    # h, w = raw_image.shape[:2]  # h, w
    h, w = (224, 384)
    image = preprocess(raw_image, depth_shape)

    # binding = session.io_binding()
    # ort_input_name = session.get_inputs()[0].name
    # binding.bind_cpu_input(ort_input_name, image)
    # binding.bind_output(ort_input_name, 'cuda')
    # session.run_with_iobinding(binding)  # Actual inference happens here.
    # output = binding.get_outputs()[0].numpy()
    
    
    # ort_inputs = {session.get_inputs()[0].name: image}
    # _ = session.run(None, ort_inputs)[0]  # warm up
    prev_depth = None
    while True:
        st = time.time()
        # depth = session.run(None, ort_inputs)[0] #.numpy()
        depth_ort = onnx_predict(session, image)[0]
        
        depth_ort = normalize(depth_ort)
        depth_ort = cv2.resize(depth_ort, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        onnx_time = time.time() - st
        print('onnx_time', onnx_time, 'depth_ort', depth_ort, depth_ort.shape)
        
        st = time.time()
        # depth = session.run(None, ort_inputs)[0] #.numpy()
        # print(image.shape)
        depth_torch = torch_model.infer_image(raw_image)

        
        depth_torch = normalize(depth_torch)
        depth_torch = cv2.resize(depth_torch, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        torch_time = time.time() - st
        print('torch_time', torch_time, 'depth_torch', depth_torch, depth_torch.shape)
        
        print(np.mean(depth_ort - depth_torch), np.max(depth_ort - depth_torch), np.min(depth_ort - depth_torch))
        np.testing.assert_allclose(depth_torch, depth_ort, rtol=1e-02, atol=1e-02)
        
        # Postprocessing, implement this part in your chosen language:
        depth_ort_map = postprocess(depth_ort, (h, w))
        depth_torch_map = postprocess(depth_torch, (h, w))

        if output_path is None:
            split_region = np.ones((depth_ort_map.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([depth_torch_map, split_region, depth_ort_map])
            cv2.imshow("combined_result", combined_result)
            if cv2.waitKey(10) == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            cv2.imwrite(str(output_path), combined_result)
            break


if __name__ == "__main__":
    # app()
    infer()
